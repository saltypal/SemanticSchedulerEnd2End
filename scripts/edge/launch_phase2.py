"""Launch every phase-2 notebook (learner training + sharded tune sweep) across both Kaggle accounts.

Default: build all notebooks and PRINT the exact kaggle commands (dry run). ``--go`` really pushes.
``--status`` prints the state of every planned notebook on both accounts; ``--collect DIR`` downloads only small
result files (json/csv/checkpoints) into DIR/<step>/ (HTTP 429: wait 60 s, retry once).

Plan: one CPU notebook per learner (slug ``edge-train-<algo>``, 5 seeds each; none needs a GPU, so no GPU session
is used), plus the ``tune`` sweep split in two shards (``edge-sweep-tune-0of2``/``1of2``). Notebooks are dealt
alternately to ``wsl`` (satyapaladuguplus) and ``win`` (satyapaladugu). The sim-table dataset must be readable by
both accounts (make it public before the first ``--account win`` run).
"""
import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.edge import make_scheduler_sweeps_notebook as sw
from scripts.edge import make_scheduler_train_notebook as tr
from scripts.edge import build_notebook as bn
from scripts.edge.kaggle_ops import ACCOUNTS, Ops, real_runner

TRAIN_ALGOS = (("ppo", None), ("scope_ppo", None), ("bc", None), ("dagger", None), ("dqn", None),
               ("scope_dqn", None), ("ltr", "lambdamart"), ("ltr", "ranknet"), ("scope_ltr", None),
               ("semcover", None), ("bcedge", None))
TUNE_SHARDS = 2
VARIANTS = ("nofusion", "fusion")     # fusion = pass --sts-fusion; slug gets the suffix -fusion
MAX_GPU_PER_ACCOUNT = 2
COLLECT_PATTERN = r".*\.(json|csv|pt|pkl|bin)$"
COLLECT_EXT = (".json", ".csv", ".pt", ".pkl", ".bin")
MAX_COLLECT_MB = 50
RETRY_WAIT_S = 60


@dataclass(frozen=True)
class Entry:
    step: str
    kind: str            # train | sweep
    algo: str            # learner name or stage
    account: str
    gpu: bool = False
    ranker: str = None
    shard: str = ""
    variant: str = "nofusion"

    @property
    def machine(self):
        return "gpu" if self.gpu else "cpu (no accelerator)"


def train_step(algo, ranker):
    return "edge-train-" + algo.replace("_", "-") + ("-" + ranker if ranker else "")


def plan(accounts=ACCOUNTS[::-1], variants=VARIANTS):
    """Ordered entries, dealt alternately to the accounts (wsl first); tune shards interleaved."""
    items = [("train", a, r, "") for a, r in TRAIN_ALGOS] + \
            [("sweep", "tune", None, "%d/%d" % (i, TUNE_SHARDS)) for i in range(TUNE_SHARDS)]
    items.sort(key=lambda it: it[0] == "sweep")           # trainers first, then tune
    entries = []
    n = 0
    for kind, algo, ranker, shard in items:
        base = train_step(algo, ranker) if kind == "train" else "edge-sweep-%s-%sof%s" % (algo, *shard.split("/"))
        for v in variants:
            entries.append(Entry(base + ("-fusion" if v == "fusion" else ""), kind, algo,
                                 accounts[n % len(accounts)], False, ranker, shard, v))
            n += 1
    return entries


def check_limits(entries):
    for acc in {e.account for e in entries}:
        if sum(e.gpu for e in entries if e.account == acc) > MAX_GPU_PER_ACCOUNT:
            raise ValueError("more than %d GPU sessions on account %s" % (MAX_GPU_PER_ACCOUNT, acc))
    slugs = [e.step for e in entries]
    if len(slugs) != len(set(slugs)):
        raise ValueError("duplicate notebook slugs")


def build_entry(e, out_dir, cfg, branch="main", seeds=5):
    """Write the notebook + metadata for one entry (owner = its account); returns the metadata dict."""
    if e.kind == "train":
        nb, meta = tr.build(e.algo, e.account, branch, seeds, e.ranker or "lambdamart", step=e.step,
                             sts_fusion=e.variant == "fusion")
    else:
        nb, meta = sw.build(e.algo, e.account, branch, e.shard, step=e.step, sts_fusion=e.variant == "fusion")
    meta["enable_gpu"] = e.gpu
    if not re.fullmatch(r"[a-z0-9-]+", meta["id"].split("/")[1]) or \
            re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") != meta["id"].split("/")[1]:
        raise ValueError("title %r does not slugify to %r" % (meta["title"], meta["id"]))
    bn.write_outputs(nb, meta, e.step, out_dir)
    return meta


def _pull_filter(folder, max_mb=MAX_COLLECT_MB):
    """Keep only small result files: drop everything else the kernel output contained."""
    for p in Path(folder).rglob("*"):
        if p.is_file() and (p.suffix not in COLLECT_EXT or p.stat().st_size > max_mb * 1024 * 1024):
            p.unlink()


class Launcher:
    def __init__(self, cfg, runner=real_runner, dry_run=True, out=print, out_dir=bn.OUT_DIR, sleep=time.sleep,
                 log_path=None):
        self.cfg, self.dry_run, self.out, self.out_dir, self.sleep = cfg, dry_run, out, Path(out_dir), sleep
        kw = {"log_path": log_path} if log_path else {}
        self.ops = {a: Ops(cfg, a, runner, dry_run, out, stage_root=self.out_dir, **kw) for a in ACCOUNTS}

    def build_all(self, entries, branch="main", seeds=5):
        check_limits(entries)
        return {e.step: build_entry(e, self.out_dir, self.cfg, branch, seeds) for e in entries}

    def push_all(self, entries):
        rcs = {}
        for e in entries:
            self.out("# %s -> account %s, %s" % (e.step, e.account, e.machine))
            rcs[e.step] = self.ops[e.account].push(e.step, self.out_dir)
        return rcs

    def status_all(self, entries):
        res = {}
        for e in entries:
            text = self.ops[e.account].status_text(e.step).strip().splitlines()
            res[e.step] = text[-1] if text else ""
            self.out("%-28s %-4s %s" % (e.step, e.account, res[e.step]))
        return res

    def collect_all(self, entries, dest):
        done = {}
        for e in entries:
            ops, folder = self.ops[e.account], Path(dest) / e.step
            args = ["kernels", "output", ops.slug(e.step), "-p", str(folder), "--file-pattern", COLLECT_PATTERN, "-o"]
            if not self.dry_run:
                folder.mkdir(parents=True, exist_ok=True)
            rc, text = ops.run(args)
            if rc and ("429" in text or "too many requests" in text.lower()):
                self.out("429 for %s: waiting %ds, one retry" % (e.step, RETRY_WAIT_S))
                self.sleep(RETRY_WAIT_S)
                rc, text = ops.run(args)
            if not self.dry_run:
                _pull_filter(folder)
            done[e.step] = rc
        return done


def main(argv=None, runner=real_runner, cfg=None, out=print, sleep=time.sleep):
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--go", action="store_true", help="really push (default: dry run, print commands)")
    g.add_argument("--status", action="store_true")
    g.add_argument("--collect", metavar="DIR")
    ap.add_argument("--dry-run", action="store_true", help="with --status/--collect: print commands only")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma list of nofusion,fusion (default both; fusion passes --sts-fusion, slug suffix -fusion)")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these notebook slugs")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    cfg = cfg or bn.load_kaggle_config()
    variants = tuple(v for v in a.variants.split(",") if v)
    if not variants or set(variants) - set(VARIANTS):
        ap.error("--variants must be a subset of %s" % ",".join(VARIANTS))
    entries = [e for e in plan(variants=variants) if not a.only or e.step in a.only]
    dry = not (a.go or a.status or a.collect) or a.dry_run
    la = Launcher(cfg, runner, dry, out, a.out_dir, sleep)
    if a.status:
        return la.status_all(entries)
    if a.collect:
        return la.collect_all(entries, a.collect)
    la.build_all(entries, a.branch, a.seeds)
    return la.push_all(entries)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
