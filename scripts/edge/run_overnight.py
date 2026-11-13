"""Unattended, resumable overnight orchestrator: runs the rest of the edge project with NO AI involvement.

    python -m scripts.edge.run_overnight [--only S4 S5] [--dry-run] [--parallel N] [--out DIR]

Stages (each writes ``<out>/<stage>.done.json`` on success and is skipped on re-run; a failed stage is logged and the
next independent stage still runs; every stage appends one line to ``<out>/STATUS.md``):
  S1 collect   launch_phase2 --collect into p2 (429: backoff, 5 retries)
  S2 curves    learning curves per run, convergence flag, retrain unconverged learners with --budget big
  S3 tune      pick SV-DPP kwargs from the tune shards by val CTR@D -> tuned.json (+ tuned_<variant>.json)
  S4 semantic  decision_trace of every analytic policy + learned seed-0 checkpoints (laptop, parallel) + review bundle
  S5 headers   header proofs on val (DECISIVE), PASS/FAIL line in STATUS.md
  S6 compare   val sweep: priority set on Kaggle (both accounts), the rest locally; merge; report; MILP bound
  S7 test      the frozen set on the test split, ONCE (state file protects against re-pushing)
  S8 live      live 2xT4 check (codec v1 kernel attached); an error tail is recorded in STATUS.md
  S9 report    FINAL_REPORT.md
``--dry-run`` writes everything under ``<out>/dry_run`` and runs no subprocess, no Kaggle call and no network.
Assumptions: a learner is "not converged" when at least half of its seeds are; learned checkpoints are seed 0; a
learned policy is evaluated with the receiver variant it was trained with (nofusion<->no fusion).
"""
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(r"D:\Bunker\BaseCamp\_claude_scratch")
SIM = SCRATCH / "simtable"
OUT_ROOT = SCRATCH / "overnight"
TMP = SCRATCH / "tmp"
VARIANTS = ("nofusion", "fusion")
WORKERS = (1, 2, 4, 6, 8)
RHOS = (0.8, 1.0, 1.2, 1.5)
REALISATIONS = 10
PRIORITY_ANALYTIC = ("sv_dpp", "semcover", "edf", "spi_first_edf", "whittle_abandon", "gen_cmu_h", "edgeserving",
                     "mpc_rollout", "fifo", "genie_sv_dpp")
PRIORITY_LEARNED = ("scope_ppo", "scope_dqn", "scope_ltr", "semcover", "bcedge")
CKPT_RE = re.compile(r"([a-z_]+)_seed(\d+)\.(\w+)")
TERMINAL = ("complete", "error", "cancel")
PUSH_RETRY_S, POLL_S = 300, 600


class StageFailed(Exception):
    pass


# ----------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, out, dry=False, parallel=None, sleep=time.sleep, scratch=SCRATCH, branch="main",
                 launch=None):
        self.scratch = Path(scratch)
        self.dry = dry
        self.out = Path(out) / "dry_run" if dry else Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.parallel = int(parallel or max(1, (os.cpu_count() or 4) - 2))
        self.sleep, self.branch = sleep, branch
        sim = self.scratch / "simtable"
        self.table, self.profile, self.fusion = sim / "simtable_v1.jsonl", sim / "latency_v1.json", sim / "sts_fusion.json"
        self.codec_quality = sim / "codec_quality.json"
        self.p2, self.dsa = self.scratch / "p2", self.scratch / "dsA"
        self.tmp = self.scratch / "tmp"
        self._launch = launch
        self._hub = None
        (self.out / "logs").mkdir(exist_ok=True)

    # -- logging / status
    def log(self, *a):
        line = "%s %s" % (datetime.now().strftime("%H:%M:%S"), " ".join(str(x) for x in a))
        print(line, flush=True)
        return line

    def status(self, stage, result, text):
        line = "- %s | %s | %s | %s" % (datetime.now().strftime("%Y-%m-%d %H:%M"), stage, result, text)
        with open(self.out / "STATUS.md", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.log(line)

    # -- state (resume info for long Kaggle stages)
    def state(self, name):
        p = self.out / "state" / (name + ".json")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def save_state(self, name, d):
        p = self.out / "state" / (name + ".json")
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")

    # -- subprocesses
    def env(self):
        e = dict(os.environ, TMP=str(self.tmp), TEMP=str(self.tmp), TMPDIR=str(self.tmp), PYTHONUNBUFFERED="1",
                 OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                 MPLCONFIGDIR=str(self.tmp / "mpl"), PYTHONPYCACHEPREFIX=str(self.tmp / "pycache"))
        return e

    def py(self, module, *args):
        return [sys.executable, "-u", "-m", module, *[str(a) for a in args]]

    def sh(self, argv, name):
        """Run one command (cwd = repo), log to logs/<name>.log. Dry-run only records the command."""
        argv = [str(a) for a in argv]
        if self.dry:
            with open(self.out / "dry_commands.log", "a", encoding="utf-8") as f:
                f.write("[%s] %s\n" % (name, " ".join(argv)))
            return 0
        self.tmp.mkdir(parents=True, exist_ok=True)
        with open(self.out / "logs" / (_safe(name) + ".log"), "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(argv) + "\n")
            f.flush()
            return subprocess.run(argv, cwd=str(REPO), env=self.env(), stdout=f, stderr=subprocess.STDOUT).returncode

    def pool(self, jobs, n=None):
        """jobs = [(name, argv, skip_if_exists_path|None)] -> {name: rc}; runs up to ``n`` (default parallel) at once."""
        res = {}

        def one(j):
            name, argv, skip = j
            if skip and Path(skip).exists():
                return name, 0
            return name, self.sh(argv, name)
        with ThreadPoolExecutor(max_workers=max(1, int(n or self.parallel))) as ex:
            for name, rc in ex.map(one, jobs):
                res[name] = rc
        return res

    def launch(self, argv):
        """launch_phase2.main in-process (so per-notebook return codes are visible). Dry-run: record only."""
        if self.dry:
            self.sh(["launch_phase2"] + list(argv), "launch_phase2")
            return {}
        if self._launch:
            return self._launch(list(argv))
        from scripts.edge import launch_phase2
        return launch_phase2.main(list(argv), out=self.log) or {}

    @property
    def hub(self):
        if self._hub is None:
            self._hub = Hub(self)
        return self._hub


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-100:]


# ----------------------------------------------------------------------------- Kaggle hub
class Hub:
    """Push / status / pull of custom notebooks on both accounts (launch_phase2.Launcher underneath)."""

    def __init__(self, ctx):
        self.ctx, self.la = ctx, None
        self.nb_dir = ctx.out / "nb"

    def _l(self):
        if self.la is None:
            from scripts.edge import build_notebook as bn
            from scripts.edge import launch_phase2 as lp
            self.la = lp.Launcher(bn.load_kaggle_config(), dry_run=False, out=self.ctx.log, out_dir=self.nb_dir)
        return self.la

    def slug(self, step, account):
        if self.ctx.dry:
            return "%s/%s" % (account, step)
        return self._l().ops[account].slug(step)

    def push(self, step, account, nb, meta):
        if self.ctx.dry:
            self.ctx.sh(["kaggle", "kernels", "push", account, step], "push_" + step)
            return 0
        from scripts.edge import build_notebook as bn
        bn.write_outputs(nb, meta, step, self.nb_dir)
        return self._l().ops[account].push(step, self.nb_dir)

    def status(self, step, account):
        if self.ctx.dry:
            return "complete"
        return self._l().ops[account].status_text(step).strip()

    def pull(self, step, account, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        if self.ctx.dry:
            return 0
        ops = self._l().ops[account]
        delay = 60
        for _ in range(6):
            rc, text = ops.run(["kernels", "output", ops.slug(step), "-p", str(dest), "-o"])
            if rc == 0 or not ("429" in text or "too many requests" in text.lower()):
                return rc
            self.ctx.sleep(delay)
            delay = min(delay * 2, 1800)
        return 1


def push_with_retry(ctx, step, account, nb, meta, max_s=6 * 3600):
    """Session limits make a push fail while the account is full: retry every 5 min."""
    t0 = time.monotonic()
    while True:
        rc = ctx.hub.push(step, account, nb, meta)
        if rc == 0:
            return True
        if time.monotonic() - t0 > max_s:
            return False
        ctx.log("push of %s failed (rc=%s); retry in %ds" % (step, rc, PUSH_RETRY_S))
        ctx.sleep(PUSH_RETRY_S)


def wait_all(ctx, items, max_s, poll=POLL_S):
    """items = [(step, account)] -> {step: 'complete'|'error'|'timeout'}; polls every ``poll`` s."""
    t0, final = time.monotonic(), {}
    while True:
        for step, acc in items:
            if step in final:
                continue
            text = ctx.hub.status(step, acc).lower()
            for t in TERMINAL:
                if t in text:
                    final[step] = "complete" if t == "complete" else "error"
                    break
        if len(final) == len(items):
            return final
        if time.monotonic() - t0 > max_s:
            return {s: final.get(s, "timeout") for s, _ in items}
        ctx.sleep(poll)


# ----------------------------------------------------------------------------- discovery
def analytic_policies():
    from Scheduler.edge import registry
    return sorted(registry.all_policies())


def learned_checkpoints(p2):
    """[{slug, variant, algo, seed, path}] for every ``<algo>_seed<k>.<ext>`` under p2/<slug>/ (not *.history.json)."""
    out = []
    p2 = Path(p2)
    if not p2.exists():
        return out
    for f in sorted(p2.rglob("*")):
        m = CKPT_RE.fullmatch(f.name) if f.is_file() else None
        if not m:
            continue
        slug = f.relative_to(p2).parts[0]
        out.append(dict(slug=slug, variant="fusion" if slug.endswith("-fusion") else "nofusion", algo=m.group(1),
                        seed=int(m.group(2)), path=str(f)))
    return out


def spec_of(c):
    return "learned:%s:%s" % (c["algo"], c["path"])


def seed0(ckpts):
    return [c for c in ckpts if c["seed"] == 0]


def train_account_map():
    from scripts.edge import launch_phase2 as lp
    return {e.step: e.account for e in lp.plan()}


# ----------------------------------------------------------------------------- S1 collect
def collect(ctx, only=None):
    args = ["--collect", str(ctx.p2)]
    if ctx.dry:
        ctx.launch(args + (["--only", *only] if only else []))
        return dict(bad=[])
    todo, delay, bad = list(only or []), 60, []
    for attempt in range(6):
        rcs = ctx.launch(args + (["--only", *todo] if todo else []))
        bad = sorted(k for k, v in rcs.items() if v)
        if not bad or attempt == 5:
            break
        ctx.log("collect failed for %s; backoff %ds" % (bad, delay))
        ctx.sleep(delay)
        delay, todo = min(delay * 2, 1800), bad
    return dict(bad=bad)


def s1(ctx):
    r = collect(ctx)
    if ctx.dry:
        return "dry-run: would collect into %s" % ctx.p2
    n = len(learned_checkpoints(ctx.p2))
    nh = len(list(ctx.p2.rglob("*.history.json")))
    line = "collected into %s: %d checkpoints, %d histories" % (ctx.p2, n, nh)
    if r["bad"]:
        line += "; FAILED " + ",".join(r["bad"])
    if r["bad"]:
        raise StageFailed(line)
    return line


# ----------------------------------------------------------------------------- S2 curves
def load_histories(p2):
    out = []
    for f in sorted(Path(p2).rglob("*.history.json")):
        m = re.search(r"([a-z_]+)_seed(\d+)", f.name)
        try:
            hist = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(hist, list) and hist:
            out.append(dict(slug=f.relative_to(p2).parts[0], file=str(f), algo=m.group(1) if m else f.stem,
                            seed=int(m.group(2)) if m else 0, hist=hist))
    return out


def series(hist, key="mean_reward"):
    return [float(h[key]) for h in hist if isinstance(h, dict) and isinstance(h.get(key), (int, float))
            and math.isfinite(h[key])]


def tail_gain(y, frac=0.2):
    """(gain, range) : mean of the last ``frac`` of iterations minus the mean of the window before it."""
    n = len(y)
    if n < 4:
        return 0.0, (max(y) - min(y)) if y else 0.0
    k = max(2, math.ceil(frac * n))
    k = min(k, n // 2)
    last, prev = sum(y[-k:]) / k, sum(y[-2 * k:-k]) / k
    return last - prev, max(y) - min(y)


def not_converged(y, frac=0.2, tol=0.05):
    gain, rng = tail_gain(y, frac)
    return bool(rng > 0 and gain > tol * rng)


def curve_rows(histories):
    rows = []
    for h in histories:
        y = series(h["hist"])
        if not y:
            rows.append(dict(learner=h["slug"], seed=h["seed"], n_iters=len(h["hist"]), note="no mean_reward"))
            continue
        gain, rng = tail_gain(y)
        rows.append(dict(learner=h["slug"], seed=h["seed"], n_iters=len(y), first=y[0], last=y[-1], best=max(y),
                         range=rng, tail_gain=gain, tail_gain_frac=(gain / rng) if rng else 0.0,
                         not_converged=int(not_converged(y)), note=""))
    return rows


def unconverged_learners(rows):
    """Learner slugs where at least half of the seeds (with a curve) are not converged."""
    by = {}
    for r in rows:
        if "not_converged" in r:
            by.setdefault(r["learner"], []).append(r["not_converged"])
    return sorted(k for k, v in by.items() if sum(v) * 2 >= len(v) and sum(v) > 0)


def plot_curves(histories, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    for h in histories:
        by.setdefault(h["slug"], []).append(h)
    paths = []
    for slug, hs in by.items():
        has_tier = any(isinstance(x, dict) and isinstance(x.get("tier_success"), list) for h in hs for x in h["hist"])
        fig, axes = plt.subplots(1, 2 if has_tier else 1, figsize=(10 if has_tier else 5, 3.6), squeeze=False)
        ax = axes[0][0]
        for h in sorted(hs, key=lambda x: x["seed"]):
            y = series(h["hist"])
            if y:
                ax.plot(range(len(y)), y, label="seed %d" % h["seed"])
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean_reward")
        ax.set_title(slug)
        ax.legend(fontsize=6)
        if has_tier:
            ax2 = axes[0][1]
            for t in range(4):
                per = []
                for h in hs:
                    v = [x["tier_success"][t] for x in h["hist"] if isinstance(x, dict) and isinstance(
                        x.get("tier_success"), list) and len(x["tier_success"]) > t]
                    if v:
                        per.append(v)
                if per:
                    m = min(len(v) for v in per)
                    ax2.plot([sum(v[i] for v in per) / len(per) for i in range(m)], label="tier %d" % t)
            ax2.set_xlabel("iteration")
            ax2.set_ylabel("tier success (mean over seeds)")
            ax2.legend(fontsize=6)
        fig.tight_layout()
        p = out / (_safe(slug) + ".png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))
    return paths


def write_csv(path, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def analyse_curves(ctx, tag=""):
    hs = load_histories(ctx.p2)
    rows = curve_rows(hs)
    write_csv(ctx.out / ("curves%s.csv" % tag), rows)
    try:
        plot_curves(hs, ctx.out / "curves")
    except Exception as e:  # plotting must never kill the stage
        ctx.log("plot failed: %r" % e)
    return hs, rows


def retrain(ctx, slugs, seeds):
    """Push the unconverged learners with the big budget, wait, re-collect. Returns a record dict."""
    st = ctx.state("S2")
    base = ["--budget", "big", "--seeds", str(seeds), "--only", *slugs]
    if not st.get("pushed"):
        pending, t0 = list(slugs), time.monotonic()
        while pending and time.monotonic() - t0 < 6 * 3600:
            rcs = ctx.launch(["--go", "--budget", "big", "--seeds", str(seeds), "--only", *pending])
            pending = sorted(k for k in pending if rcs.get(k, 0)) if rcs else []
            if pending:
                ctx.log("push failed for %s; retry in %ds" % (pending, PUSH_RETRY_S))
                ctx.sleep(PUSH_RETRY_S)
        st["pushed"] = sorted(set(slugs) - set(pending))
        st["push_failed"] = pending
        ctx.save_state("S2", st)
    waiting, t0, final = list(st["pushed"]), time.monotonic(), {}
    while waiting and time.monotonic() - t0 < 4 * 3600:
        res = ctx.launch(["--status", "--only", *waiting]) or {}
        final = {k: v for k, v in res.items() if any(t in str(v).lower() for t in TERMINAL)}
        if len(final) >= len(waiting) or ctx.dry:
            break
        ctx.sleep(POLL_S)
    collect(ctx, st["pushed"] or None)
    return dict(slugs=slugs, pushed=st["pushed"], push_failed=st["push_failed"], final=final)


def s2(ctx):
    if ctx.dry:
        retrain(ctx, ["edge-train-ppo"], 5)
        return "dry-run: would plot curves, flag unconverged learners and retrain them with --budget big"
    hs, rows = analyse_curves(ctx, "_before")
    bad = unconverged_learners(rows)
    n_ok = sum(1 for r in rows if "not_converged" in r)
    line = "%d runs with curves, %d learners not converged: %s; csv=%s plots=%s" % (
        n_ok, len(bad), ",".join(bad) or "-", ctx.out / "curves_before.csv", ctx.out / "curves")
    if bad:
        seeds = max([h["seed"] for h in hs if h["slug"] in bad] or [2]) + 1
        rec = retrain(ctx, bad, seeds)
        (ctx.out / "retrained.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        analyse_curves(ctx)
        line += "; RETRAINED %s (push_failed=%s)" % (",".join(rec["pushed"]), rec["push_failed"] or "-")
    else:
        shutil.copy(ctx.out / "curves_before.csv", ctx.out / "curves.csv")
    return line


# ----------------------------------------------------------------------------- S3 tune
def pick_tuned(files):
    """[(variant, json path)] of tune shards -> {variant: {best_kwargs, best_ctr, n_points, n_shards}}."""
    by = {}
    for variant, f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        by.setdefault(variant, []).append(d)
    out = {}
    for variant, ds in by.items():
        pts = [p for d in ds for p in d.get("sv_dpp_grid", [])
               if isinstance(p.get("ctr_ungated"), (int, float)) and math.isfinite(p["ctr_ungated"])]
        if not pts:
            continue
        best = max(pts, key=lambda p: p["ctr_ungated"])
        out[variant] = dict(best_kwargs=best["kwargs"], best_ctr=best["ctr_ungated"], n_points=len(pts),
                            n_shards=len(ds))
    return out


def tune_files(p2):
    out = []
    for f in sorted(Path(p2).rglob("tune*.json")):
        slug = f.relative_to(p2).parts[0]
        if slug.startswith("edge-sweep-tune"):
            out.append(("fusion" if slug.endswith("-fusion") else "nofusion", f))
    return out


def tuned_for(ctx, variant):
    """Tuned sv_dpp kwargs of a variant (falls back to the other variant, then to None = config defaults)."""
    p = ctx.out / "tuned.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8")).get("variants", {})
    for v in (variant, *[x for x in VARIANTS if x != variant]):
        if v in d:
            return d[v]["best_kwargs"]
    return None


def s3(ctx):
    if ctx.dry:
        return "dry-run: would read tune shards under %s and write tuned.json" % ctx.p2
    res = pick_tuned(tune_files(ctx.p2))
    if not res:
        raise StageFailed("no usable tune shard (sv_dpp_grid) under %s" % ctx.p2)
    tuned = dict(best_kwargs=(res.get("nofusion") or next(iter(res.values())))["best_kwargs"], variants=res,
                 untuned_defaults=["tie_tol_frac", "relevance_beta"], note="kwargs for sv_dpp; no config edited")
    (ctx.out / "tuned.json").write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    for v, d in res.items():
        (ctx.out / ("tuned_%s.json" % v)).write_text(json.dumps(dict(best_kwargs=d["best_kwargs"]), indent=1),
                                                      encoding="utf-8")
    return "tuned %s; wrote %s" % ("; ".join("%s=%s (CTR@D %.3f)" % (v, d["best_kwargs"], d["best_ctr"])
                                             for v, d in res.items()), ctx.out / "tuned.json")


# ----------------------------------------------------------------------------- S4 semantic audit
def semantic_jobs(ctx, analytic, ckpts):
    """[(variant, label, policy_spec)]: analytic x both variants; learned seed 0 with its own variant."""
    jobs = [(v, p, p) for p in analytic for v in VARIANTS]
    for c in seed0(ckpts):
        jobs.append((c["variant"], "learned_" + c["slug"], spec_of(c)))
    return jobs


def trace_cmd(ctx, variant, spec, out):
    argv = ctx.py("Evaluation.edge.decision_trace", "--table", ctx.table, "--profile", ctx.profile, "--split", "val",
                  "--policies", spec, "--workers", 2, "--rho", 1.2, "--subsample", 400, "--arrival", "trace",
                  "--out", out)
    if variant == "fusion":
        argv += ["--sts-fusion", ctx.fusion]
    if ctx.dsa.exists():
        argv += ["--images-root", ctx.dsa / "images"]
    return argv


def image_lookup(root):
    idx = {}
    if root and Path(root).exists():
        idx = {p.stem: p for p in Path(root).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    def find(image_id):
        tail = str(image_id).split(":", 1)[-1]
        if root and (Path(root) / tail).is_file():
            return str(Path(root) / tail)
        p = idx.get(Path(tail).stem)
        return str(p) if p else None
    return find


def review_sample(trace_file, find, n=30, max_skipped=15, seed=0):
    """30 sampled non-idle decisions with picked / skipped image paths and TRUE tiers."""
    decs = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "decision" and r.get("batch"):
                decs.append(r)
    picks = sorted(random.Random(seed).sample(range(len(decs)), min(n, len(decs))))
    out = []
    for i in picks:
        d = decs[i]
        b = set(d["batch"])

        def view(j):
            return dict(id=j["id"], image_id=j["image_id"], image_path=find(j["image_id"]), true_tier=j["true_tier"],
                        age=j["age"], slack_hard=j["slack_hard"], spi=j["spi"], score=j.get("score"))
        sk = sorted((j for j in d["queue"] if j["id"] not in b), key=lambda j: (j["true_tier"] is None, j["true_tier"],
                                                                                 -j["age"]))
        out.append(dict(t=d["t"], queue_size=d["queue_size"], picked=[view(j) for j in d["queue"] if j["id"] in b],
                        skipped_total=len(sk), skipped=[view(j) for j in sk[:max_skipped]]))
    return out


REVIEW_README = """# Semantic review bundle

One JSON per policy and receiver variant (`<variant>/<label>.json`). `decisions` = 30 sampled non-idle scheduling
decisions on val (2 workers, rho 1.2, trace arrivals): `picked` and `skipped` jobs with image paths and TRUE tiers
(0 Critical, 1 High, 2 Medium, 3 Low). Policies never saw the true tier. `metrics` = the policy's own semantic
metrics (semantic_precision, inversions_per_100, kendall_tau, waits). Question for the reviewer, per policy: does it
pick the more important image first (verdict: orders semantically / partly / not), with 2-3 example decisions.
Skipped lists are capped (most important first); `skipped_total` is the full count.
"""


def aggregate_semantic(ctx, jobs):
    rows = []
    find = image_lookup(ctx.dsa / "images")
    bundle = ctx.out / "review_bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "README.md").write_text(REVIEW_README, encoding="utf-8")
    for variant, label, spec in jobs:
        d = ctx.out / "semantic" / variant / _safe(label)
        s = d / "summary.csv"
        if not s.exists():
            continue
        with open(s, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(variant=variant, label=label, **r))
        tr = sorted(d.glob("*_trace.jsonl"))
        mets = sorted(d.glob("*_metrics.json"))
        if tr:
            (bundle / variant).mkdir(exist_ok=True)
            (bundle / variant / (_safe(label) + ".json")).write_text(json.dumps(dict(
                policy=label, variant=variant, spec=spec,
                metrics=json.loads(mets[0].read_text(encoding="utf-8")) if mets else None,
                decisions=review_sample(tr[0], find)), indent=1), encoding="utf-8")
    write_csv(ctx.out / "semantic_summary.csv", rows)
    return rows


def s4(ctx):
    ckpts = [] if ctx.dry else learned_checkpoints(ctx.p2)
    jobs = semantic_jobs(ctx, analytic_policies(), ckpts)
    cmds = []
    for variant, label, spec in jobs:
        d = ctx.out / "semantic" / variant / _safe(label)
        cmds.append(("trace_%s_%s" % (variant, label), trace_cmd(ctx, variant, spec, d), d / "summary.csv"))
    res = ctx.pool(cmds)
    bad = sorted(k for k, v in res.items() if v)
    if ctx.dry:
        return "dry-run: %d trace jobs (%d analytic x2 + %d learned)" % (len(jobs), len(jobs) - len(seed0(ckpts)),
                                                                          len(seed0(ckpts)))
    rows = aggregate_semantic(ctx, jobs)
    line = "%d traces ok, %d failed (%s); %s ; review bundle %s" % (
        len(jobs) - len(bad), len(bad), ",".join(bad[:5]) or "-", ctx.out / "semantic_summary.csv",
        ctx.out / "review_bundle")
    if not rows:
        raise StageFailed(line)
    return line


# ----------------------------------------------------------------------------- S5 header proofs
def ci_of(cell, key="ctr_ungated"):
    ci = (cell.get("ci95") or {}).get(key)
    return tuple(ci) if ci and ci[0] is not None and ci[1] is not None else None


def header_verdict(ablation, policy="sv_dpp", rho=1.2):
    """PASS iff, at every worker count, CTR@D full > spi > none with non-overlapping 95% CIs at ``rho``."""
    by = {}
    for c in ablation:
        if c["policy"] == policy and abs(float(c["rho"]) - rho) < 1e-9 and c["variant"] in (
                "cum:full", "cum:spi", "cum:none"):
            by[(int(c["workers"]), c["variant"])] = c
    ws = sorted({w for w, _ in by})
    if not ws:
        return False, "no %s cells at rho %s" % (policy, rho)
    ok, parts = True, []
    for w in ws:
        try:
            f, s, n = by[(w, "cum:full")], by[(w, "cum:spi")], by[(w, "cum:none")]
        except KeyError:
            ok = False
            parts.append("c=%d missing" % w)
            continue
        cf, cs, cn = ci_of(f), ci_of(s), ci_of(n)
        good = (f["ctr_ungated"] > s["ctr_ungated"] > n["ctr_ungated"] and cf and cs and cn
                and cf[0] > cs[1] and cs[0] > cn[1])
        ok = ok and bool(good)
        parts.append("c=%d full %.3f [%.3f,%.3f] spi %.3f [%.3f,%.3f] none %.3f [%.3f,%.3f] %s" % (
            w, f["ctr_ungated"], *(cf or (math.nan,) * 2), s["ctr_ungated"], *(cs or (math.nan,) * 2),
            n["ctr_ungated"], *(cn or (math.nan,) * 2), "ok" if good else "no"))
    return ok, "; ".join(parts)


def merge_header_results(results):
    base = dict(results[0])
    base["ablation"] = [c for r in results for c in r["ablation"]]
    base["dedup"] = {k: v for r in results for k, v in r.get("dedup", {}).items()}
    base["config"] = dict(base["config"], policies=sorted({p for r in results for p in r["config"]["policies"]}))
    return base


def s5(ctx):
    policies, jobs = ("sv_dpp", "semcover"), []
    for v in VARIANTS:
        for p in policies:
            d = ctx.out / "s5_header" / v / p
            argv = ctx.py("Evaluation.edge.header_proofs", "--table", ctx.table, "--profile", ctx.profile, "--split",
                          "val", "--policies", p, "--workers", 1, 2, "--rhos", 0.8, 1.2, 1.5, "--realisations", 10,
                          "--arrival", "trace", "--out-dir", d)
            if v == "fusion":
                argv += ["--sts-fusion", ctx.fusion]
            jobs.append(("hp_%s_%s" % (v, p), argv, d / "header_proofs.json"))
    res = ctx.pool(jobs)
    if ctx.dry:
        ctx.sh(ctx.py("Evaluation.edge.report", "header-proofs", "--csv", "<merged>", "--out", "<report>"), "report_hp")
        return "dry-run: %d header-proof jobs; PASS/FAIL line would be written" % len(jobs)
    lines, any_data = [], False
    from Evaluation.edge import header_proofs as hp
    for v in VARIANTS:
        parts = []
        for p in policies:
            f = ctx.out / "s5_header" / v / p / "header_proofs.json"
            if f.exists():
                parts.append(json.loads(f.read_text(encoding="utf-8")))
        if not parts:
            lines.append("%s: no data (%s)" % (v, [k for k, rc in res.items() if rc and v in k]))
            continue
        any_data = True
        merged = merge_header_results(parts)
        d = ctx.out / "s5_header" / v
        hp.save(merged, str(d))
        ctx.sh(ctx.py("Evaluation.edge.report", "header-proofs", "--csv", d / "header_proofs.csv", "--out",
                      ctx.out / "report_header" / v), "report_hp_" + v)
        ok, detail = header_verdict(merged["ablation"])
        lines.append("HEADER-PROOF %s [%s] sv_dpp rho 1.2: %s" % ("PASS" if ok else "FAIL", v, detail))
    if not any_data:
        raise StageFailed("header proofs produced no data: " + "; ".join(lines))
    (ctx.out / "header_verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return " || ".join(lines) + "; csv under %s" % (ctx.out / "s5_header")


# ----------------------------------------------------------------------------- S6 / S7 sweeps
def split_names(ctx, ckpts):
    """Priority and remaining policy sets (analytic names + learned specs) per variant."""
    registry = set(analytic_policies())
    prio = [p for p in PRIORITY_ANALYTIC if p in registry]
    rest = [p for p in analytic_policies() if p not in prio]
    plan = {}
    for v in VARIANTS:
        mine = [c for c in seed0(ckpts) if c["variant"] == v]
        kaggle_learned = [c for c in mine if c["algo"] in PRIORITY_LEARNED]
        local_learned = [c for c in mine if c["algo"] not in PRIORITY_LEARNED]
        plan[v] = dict(analytic=prio, learned=kaggle_learned, local=rest + [spec_of(c) for c in local_learned])
    return plan


def kaggle_notebooks(ctx, plan, stage, accounts_of):
    """[(step, account, variant, policies, kernel_slugs)] : 2 variants x 2 accounts."""
    nbs = []
    accs = ("wsl", "win")
    for v in VARIANTS:
        by = {a: dict(policies=[], kernels=[]) for a in accs}
        for i, p in enumerate(plan[v]["analytic"]):
            by[accs[i % 2]]["policies"].append(p)
        for c in plan[v]["learned"]:
            acc = accounts_of.get(c["slug"], "wsl")
            by[acc]["policies"].append("learned:%s:%s" % (c["algo"], Path(c["path"]).name))
            by[acc]["kernels"].append(c["slug"])
        for a in accs:
            if by[a]["policies"]:
                step = "edge-sweep-%s-%s%s" % (stage, a, "-fusion" if v == "fusion" else "")
                nbs.append((step, a, v, by[a]["policies"], sorted(set(by[a]["kernels"]))))
    return nbs


def local_shard_cmds(ctx, stage, plan, outdir):
    cmds = []
    for v in VARIANTS:
        pols = plan[v]["local"]
        n = max(1, min(ctx.parallel, len(pols)))
        for i in range(n):
            d = outdir / "local" / v
            argv = ctx.py("Evaluation.edge.sweep_stages", "--stage", stage, "--table", ctx.table, "--profile",
                          ctx.profile, "--out-dir", d, "--shard", "%d/%d" % (i, n), "--policies", *pols,
                          "--workers", *WORKERS, "--rhos", *RHOS, "--realisations", REALISATIONS)
            if v == "fusion":
                argv += ["--sts-fusion", ctx.fusion]
            if stage == "test":
                argv.append("--final-test")
            if tuned_for(ctx, v):
                tf = ctx.out / ("tuned_%s.json" % v)
                if tf.exists():
                    argv += ["--tuned", tf]
            cmds.append(("%s_local_%s_%d" % (stage, v, i), argv, d / ("%s.shard%dof%d.json" % (stage, i, n))))
    return cmds


def merge_runs(files):
    """Union of run_sweep result files (cells deduped on policy/workers/rho; later files win)."""
    base, cells = None, {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if "cells" not in d:
            continue
        base = base or d
        for c in d["cells"]:
            cells[(c["policy"], int(c["workers"]), float(c["rho"]))] = c
    if base is None:
        return None
    base = dict(base, cells=list(cells.values()))
    base["config"] = dict(base["config"], policies=sorted({k[0] for k in cells}),
                          workers=sorted({k[1] for k in cells}), rhos=sorted({k[2] for k in cells}))
    return base


def sweep_stage(ctx, split):
    stage = "compare" if split == "val" else "test"
    tag = "S6" if split == "val" else "S7"
    ckpts = [] if ctx.dry else learned_checkpoints(ctx.p2)
    plan = split_names(ctx, ckpts)
    outdir = ctx.out / ("s6_compare" if split == "val" else "s7_test")
    st = ctx.state(tag)
    nbs = kaggle_notebooks(ctx, plan, stage, {} if ctx.dry else train_account_map())
    # 1. push the Kaggle notebooks (never twice: S7 is a one-shot)
    from scripts.edge import make_scheduler_sweeps_notebook as sw
    for step, acc, v, pols, kernels in nbs:
        if st.get("pushed", {}).get(step):
            continue
        nb = meta = None
        if not ctx.dry:
            ks = [ctx.hub.slug(k, acc) for k in kernels]
            nb, meta = sw.build(stage, acc, ctx.branch, "", pols, split == "test", step, v == "fusion", ks,
                                ["--workers", *map(str, WORKERS), "--rhos", *map(str, RHOS), "--realisations",
                                 str(REALISATIONS)], dict(best_kwargs=tuned_for(ctx, v)) if tuned_for(ctx, v) else None)
        if push_with_retry(ctx, step, acc, nb, meta):
            st.setdefault("pushed", {})[step] = acc
            ctx.save_state(tag, st)
        else:
            ctx.log("could not push %s" % step)
    # 2. local analytic remainder (parallel) while Kaggle runs
    res = ctx.pool(local_shard_cmds(ctx, stage, plan, outdir))
    # 3. wait + pull
    pushed = list(st.get("pushed", {}).items())
    final = wait_all(ctx, pushed, 8 * 3600) if pushed else {}
    for step, acc in pushed:
        if final.get(step) == "complete" and not (outdir / "kaggle" / step).exists():
            ctx.hub.pull(step, acc, outdir / "kaggle" / step)
    if ctx.dry:
        return "dry-run: %d Kaggle notebooks, %d local shard jobs" % (len(nbs), len(res))
    # 4. merge + report per variant
    summary = []
    for v in VARIANTS:
        files = list((outdir / "local" / v).glob("%s*.json" % stage))
        for step, acc in pushed:
            if (step.endswith("-fusion")) == (v == "fusion"):
                files += list((outdir / "kaggle" / step).rglob("%s*.json" % stage))
        merged = merge_runs(files)
        if not merged:
            summary.append("%s: no results" % v)
            continue
        name = "%s_all_%s.json" % (stage, v)
        (outdir / name).write_text(json.dumps(merged), encoding="utf-8")
        ctx.sh(ctx.py("Evaluation.edge.report", "results", "--runs", outdir / name, "--out",
                      outdir / "report" / v, "--split", split), "report_%s_%s" % (stage, v))
        summary.append("%s: %d cells, %d policies" % (v, len(merged["cells"]), len(merged["config"]["policies"])))
    first = outdir / ("%s_all_nofusion.json" % stage)
    if first.exists():
        shutil.copy(first, outdir / ("%s_all.json" % stage))
    bad = [s for s, f in final.items() if f != "complete"]
    line = "; ".join(summary) + "; kaggle %s; failed local jobs %s; out %s" % (
        {s: f for s, f in final.items()}, sorted(k for k, v in res.items() if v) or "-", outdir)
    if not any("cells" in x for x in summary):
        raise StageFailed(line)
    return line + ("; KAGGLE PROBLEMS: %s" % bad if bad else "")


def milp_run(ctx, n_jobs=200, seeds=3):
    import numpy as np
    from Evaluation.edge import run_sweep as rs
    from Evaluation.edge import tables
    from Evaluation.edge.env import EdgeEnv
    from Evaluation.edge.milp_bound import clairvoyant_bound, greedy_upper_bound
    from Evaluation.edge.receiver import Receiver
    lat = tables.latency_from_profile_json(str(ctx.profile))
    rows, spec, _ = rs._setup(tables.load_rows(str(ctx.table)), lat, "val", None, 1)
    out, path = [], ctx.out / "milp_bound.json"
    for seed in range(seeds):
        keep = np.sort(np.random.default_rng(rs._seed(seed, 555)).choice(len(rows), min(n_jobs, len(rows)), replace=False))
        sub = [rows[i] for i in keep]
        for w in (1, 2):
            for rho in (1.0, 1.2):
                try:
                    jobs, _ = rs._make_jobs(sub, lat, tables, w, rho, 0, seed, 1, None, 0.0, {})
                    opt, _ = clairvoyant_bound(jobs, lat, spec, w, 1)
                    rec = dict(seed=seed, workers=w, rho=rho, n_jobs=len(jobs), milp=opt,
                               greedy_ub=greedy_upper_bound(jobs, spec), policies={})
                    truth = {j.id: j.truth for j in jobs}
                    for p in ("fifo", "edf", "sv_dpp", "semcover", "genie_sv_dpp"):
                        res = EdgeEnv(jobs, lat, spec, w, 1, Receiver(), rs._seed(seed, w, 1)).run(
                            rs._build_policy(p, {}, truth))
                        rec["policies"][p] = float(sum(o.get("value", 0.0) for o in res.outcomes))
                    out.append(rec)
                except Exception as e:
                    out.append(dict(seed=seed, workers=w, rho=rho, error=repr(e)))
                path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def s6(ctx):
    line = sweep_stage(ctx, "val")
    if not ctx.dry:
        try:
            m = milp_run(ctx)
            ok = [r for r in m if "milp" in r]
            line += "; MILP bound %d/%d cases (%s)" % (len(ok), len(m), ctx.out / "milp_bound.json")
        except Exception as e:
            line += "; MILP FAILED %r" % e
    else:
        ctx.sh(ctx.py("Evaluation.edge.milp_bound", "(in-process)"), "milp")
    return line


def s7(ctx):
    return sweep_stage(ctx, "test")


# ----------------------------------------------------------------------------- S8 live
def s8(ctx):
    from scripts.edge import make_live_t4_notebook as live
    from Evaluation.edge.edge_config import load_config
    step, acc = live.STEP, "wsl"
    kernel = load_config("codec")["codecs"]["v1"].get("kaggle_kernel")
    st = ctx.state("S8")
    if not st.get("pushed"):
        nb = meta = None
        if not ctx.dry:
            nb, meta = live.build(acc, ctx.branch, 1000, ("fifo", "edf", "sv_dpp"))
            if kernel:   # attach the v1 codec kernel output; drop the (possibly unpublished) fallback dataset
                fallback = load_config("codec")["codecs"]["v1"].get("kaggle_dataset")
                meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d != fallback]
                meta["kernel_sources"] = list(meta.get("kernel_sources", [])) + [kernel]
        if not push_with_retry(ctx, step, acc, nb, meta):
            raise StageFailed("could not push %s" % step)
        st["pushed"] = True
        ctx.save_state("S8", st)
    final = wait_all(ctx, [(step, acc)], 6 * 3600).get(step)
    dest = ctx.out / "s8_live"
    ctx.hub.pull(step, acc, dest)
    if ctx.dry:
        return "dry-run: would push %s with kernel %s, poll, collect live_check.json" % (step, kernel)
    found = sorted(dest.rglob("live_check.json"))
    if final == "complete" and found:
        shutil.copy(found[0], ctx.out / "live_check.json")
        return "live check collected: %s" % (ctx.out / "live_check.json")
    tail = ""
    for lg in sorted(dest.rglob("*.log")):
        tail = "\n".join(lg.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
    (ctx.out / "s8_error_tail.txt").write_text(tail, encoding="utf-8")
    st["pushed"] = False   # a re-run pushes a fresh version
    ctx.save_state("S8", st)
    raise StageFailed("live run %s, no live_check.json; log tail: %s" % (final, " | ".join(tail.splitlines()[-8:])[:900]))


# ----------------------------------------------------------------------------- S9 report
def md_table(rows, cols=None, limit=60):
    if not rows:
        return "_(none)_\n"
    cols = cols or list(rows[0])
    fmt = lambda v: ("%.4g" % v) if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows[:limit]]
    return "\n".join(out) + "\n"


def read_csv(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_report(ctx):
    o = ctx.out
    status = (o / "STATUS.md").read_text(encoding="utf-8") if (o / "STATUS.md").exists() else ""
    parts = ["# Overnight final report\n", "Generated %s. Stage log: STATUS.md\n" % datetime.now().isoformat(timespec="minutes"),
             "## Stage status\n```\n%s\n```\n" % status]
    parts.append("## Header proofs (decisive)\n%s\n" % ((o / "header_verdict.txt").read_text(encoding="utf-8")
                                                        if (o / "header_verdict.txt").exists() else "_not run_"))
    for v in VARIANTS:
        parts.append("### Ablation table (%s)\n%s" % (v, md_table(read_csv(o / "report_header" / v / "ablation_table.csv"))))
    parts.append("## Codec quality (simtable codec_quality.json)\n")
    if ctx.codec_quality.exists():
        d = json.loads(ctx.codec_quality.read_text(encoding="utf-8"))
        parts.append("```\n%s\n```\n" % json.dumps({k: d[k] for k in ("levels", "gate", "notes") if k in d}, indent=1)[:4000])
    else:
        parts.append("_missing_\n")
    for title, sub in (("Compare (val)", "s6_compare"), ("Test (frozen)", "s7_test")):
        for v in VARIANTS:
            parts.append("## %s main table (%s)\n%s" % (title, v, md_table(read_csv(o / sub / "report" / v / "main_table.csv"), limit=40)))
    parts.append("## Semantic summary (decision trace, val, c=2, rho 1.2)\n%s" % md_table(read_csv(o / "semantic_summary.csv"), limit=120))
    for name in ("tuned.json", "retrained.json", "milp_bound.json", "live_check.json"):
        p = o / name
        parts.append("## %s\n```\n%s\n```\n" % (name, p.read_text(encoding="utf-8")[:3000] if p.exists() else "(missing)"))
    (o / "FINAL_REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    return o / "FINAL_REPORT.md"


def s9(ctx):
    return "wrote %s" % build_report(ctx)


# ----------------------------------------------------------------------------- driver
STAGES = (("S1", "collect", s1), ("S2", "curves", s2), ("S3", "tune", s3), ("S4", "semantic", s4),
          ("S5", "headers", s5), ("S6", "compare", s6), ("S7", "test", s7), ("S8", "live", s8), ("S9", "report", s9))


def run_stages(ctx, only=None, stages=STAGES):
    results = {}
    for sid, name, fn in stages:
        if only and sid not in only:
            continue
        marker = ctx.out / ("%s.done.json" % sid)
        if marker.exists():
            ctx.status("%s %s" % (sid, name), "SKIP", "already done (%s)" % marker)
            results[sid] = "skip"
            continue
        t0 = time.time()
        ctx.log("=== %s %s ===" % (sid, name))
        try:
            text = fn(ctx)
            marker.write_text(json.dumps(dict(stage=sid, name=name, seconds=round(time.time() - t0, 1), line=text,
                                              dry_run=ctx.dry, finished=datetime.now().isoformat()), indent=1),
                              encoding="utf-8")
            ctx.status("%s %s" % (sid, name), "OK", text)
            results[sid] = "ok"
        except Exception as e:
            tail = (str(e) if isinstance(e, StageFailed) else traceback.format_exc().strip().splitlines()[-1])
            if not isinstance(e, StageFailed):
                (ctx.out / "logs" / ("%s.traceback.txt" % sid)).write_text(traceback.format_exc(), encoding="utf-8")
            ctx.status("%s %s" % (sid, name), "FAIL", tail[:1200])
            results[sid] = "fail"
    return results


def main(argv=None, **kw):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, metavar="S", help="stage ids, e.g. S4 S5")
    ap.add_argument("--dry-run", action="store_true", help="no subprocess / Kaggle / network; outputs in <out>/dry_run")
    ap.add_argument("--parallel", type=int, default=None, help="laptop processes (default cpu_count-2)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--branch", default="main")
    a = ap.parse_args(argv)
    os.environ.update(TMP=str(TMP), TEMP=str(TMP))
    ctx = Ctx(a.out, a.dry_run, a.parallel, branch=a.branch, **kw)
    ctx.status("orchestrator", "START", "stages=%s dry=%s parallel=%d out=%s" % (
        ",".join(a.only) if a.only else "all", a.dry_run, ctx.parallel, ctx.out))
    res = run_stages(ctx, set(a.only) if a.only else None)
    ctx.status("orchestrator", "END", " ".join("%s=%s" % kv for kv in res.items()))
    return res


if __name__ == "__main__":
    main()
