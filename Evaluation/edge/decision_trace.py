"""Inspectable scheduling: per-decision trace, semantic-ordering metrics, text/HTML replay.

``TracingPolicy`` wraps any policy. It logs, at every ``select``, every queued job (image id, TRUE tier for
auditing only, SPI probs, relevance, mission, age, slack, dup_prob), the chosen batch and the policy's own
scores when it exposes ``last_scores`` ({job id: float}). The wrapped policy only ever sees the ``State``;
Truth stays in the wrapper. ``semantic_metrics`` derives, from the trace and TRUE tiers:
  semantic_precision  fraction of non-idle decisions whose best chosen tier is >= (as important as) the best
                      skipped job that could still meet its hard deadline (no such job counts as precise)
  inversions_per_100  decisions with a Medium/Low job chosen while a feasible Critical waited, per 100 decisions
  kendall_tau         mean over decisions of tau between the policy order (chosen in batch order, then the
                      unchosen tied) and true value order (tier weight); ties excluded, tau=(C-D)/(C+D)
  wait_critical/low   mean seconds from arrival to being picked, true Critical / Low
  ctr                 CTR@D from the run (metrics.ctr, measured jobs)
"""
import argparse
import csv
import html
import json
import os
import re
from pathlib import Path

import numpy as np

from Evaluation.edge import metrics as edge_metrics
from Evaluation.edge import run_sweep
from Evaluation.edge.contracts import Policy, TIERS
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver

TIER_COLORS = ("#d62728", "#ff7f0e", "#bcbd22", "#9aa5b1")


class TracingPolicy(Policy):
    def __init__(self, inner, truth_by_id, image_by_id=None):
        self.inner, self.truth, self.images = inner, dict(truth_by_id), image_by_id or {}
        self.name, self.uses_truth = inner.name, False   # truth is never forwarded
        self.records = []

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.records = []
        self.inner.reset(spec, rng)

    def _job(self, j, spec, scores):
        k = self.truth[j.id].tier if j.id in self.truth else None
        h = j.header
        sc = None if scores is None else scores.get(j.id)
        return dict(id=j.id, image_id=self.images.get(j.id, str(j.id)), true_tier=k,
                    spi=[round(float(p), 4) for p in h.tier_probs], relevance=float(h.relevance),
                    mission=int(h.mission), age=float(j.age), dup_prob=float(j.dup_prob),
                    arrival=float(h.arrival_time),
                    slack_soft=None if k is None else float(spec.soft_deadline[k] - j.age),
                    slack_hard=None if k is None else float(spec.hard_deadline(k) - j.age),
                    score=None if sc is None else float(sc))

    def select(self, state):
        dec = self.inner.select(state)
        raw = getattr(self.inner, "last_scores", None)
        scores = raw if isinstance(raw, dict) else None
        self.records.append(dict(
            kind="decision", t=float(state.now), queue_size=len(state.queue),
            min_service=float(state.latency.quantile(state.latency.batch_sizes[0], 0.0)),
            queue=[self._job(j, state.spec, scores) for j in state.queue],
            batch=list(dec.batch), purge=list(dec.purge), has_scores=scores is not None))
        return dec

    def observe(self, event, payload):
        if event == "complete":
            for i, d in zip(payload["ids"], payload["delays"]):
                k = self.truth[i].tier if i in self.truth else None
                on = None if k is None else bool(d <= self.spec.soft_deadline[k])
                self.records.append(dict(kind="complete", t=float(payload["time"]), id=i,
                                         image_id=self.images.get(i, str(i)), true_tier=k,
                                         delay=float(d), on_time=on))
        self.inner.observe(event, payload)


def _decisions(records):
    return [r for r in records if r["kind"] == "decision"]


def _feasible(j, min_service):
    return j["slack_hard"] is not None and j["slack_hard"] > min_service


def kendall_tau(queue, batch):
    """Tau between selection order (chosen in order, unchosen tied last) and true value order."""
    pos = {i: p for p, i in enumerate(batch)}
    items = [(pos.get(j["id"], len(batch)), -j["true_tier"]) for j in queue]
    c = d = 0
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (pa, va), (pb, vb) = items[a], items[b]
            if pa == pb or va == vb:
                continue
            if (pa < pb) == (va > vb):
                c += 1
            else:
                d += 1
    return (c - d) / (c + d) if c + d else None


def semantic_metrics(records, ctr=None):
    decs = [r for r in _decisions(records) if r["batch"]]
    precise = inv = 0
    taus, waits, tier_of = [], {}, {}
    for r in decs:
        by = {j["id"]: j for j in r["queue"]}
        chosen = [by[i] for i in r["batch"]]
        picked = set(r["batch"])
        skipped = [j for j in r["queue"] if j["id"] not in picked and j["true_tier"] is not None
                   and _feasible(j, r["min_service"])]
        best_chosen = min(j["true_tier"] for j in chosen)
        best_skipped = min((j["true_tier"] for j in skipped), default=None)
        precise += best_skipped is None or best_chosen <= best_skipped
        inv += any(j["true_tier"] >= 2 for j in chosen) and any(j["true_tier"] == 0 for j in skipped)
        tau = kendall_tau(r["queue"], r["batch"])
        if tau is not None:
            taus.append(tau)
        for j in chosen:
            waits.setdefault(j["id"], r["t"] - j["arrival"])
            tier_of[j["id"]] = j["true_tier"]
    n = len(decs)

    def wait(k):
        w = [v for i, v in waits.items() if tier_of[i] == k]
        return float(np.mean(w)) if w else None
    return dict(n_decisions=n,
                semantic_precision=precise / n if n else None,
                inversions_per_100=100.0 * inv / n if n else None,
                kendall_tau=float(np.mean(taus)) if taus else None,
                wait_critical=wait(0), wait_low=wait(3), ctr=ctr)


def _spi_txt(j):
    k = int(np.argmax(j["spi"]))
    return "SPI %s=%.2f" % ("CHML"[k], j["spi"][k])


def _tier_name(k):
    return "?" if k is None else TIERS[k]


def replay_lines(records):
    out = []
    for r in _decisions(records):
        if not r["batch"]:
            continue
        by = {j["id"]: j for j in r["queue"]}
        picked = "; ".join("%s [TRUE %s, %s]" % (by[i]["image_id"], _tier_name(by[i]["true_tier"]), _spi_txt(by[i]))
                           for i in r["batch"])
        sk = [j for j in r["queue"] if j["id"] not in set(r["batch"]) and j["true_tier"] is not None
              and _feasible(j, r["min_service"])]
        best = "[TRUE %s]" % _tier_name(min(j["true_tier"] for j in sk)) if sk else "none"
        crit = sum(j["true_tier"] == 0 for j in sk)
        out.append("t=%.1fs queue=%d | picked: %s | skipped best: %s | Critical waiting: %d"
                   % (r["t"], r["queue_size"], picked, best, crit))
    return out


def _thumb(image_id, root, out_dir):
    if not root:
        return None
    p = Path(root) / image_id.split(":", 1)[-1]
    if not p.exists():
        return None
    try:
        return os.path.relpath(p, out_dir).replace("\\", "/")
    except ValueError:
        return p.as_uri()


_PAGE = """<!doctype html><meta charset=utf-8><title>__T__</title><style>body{font:13px sans-serif;margin:12px}
.row{display:flex;align-items:center;margin:1px 0}.lab{width:110px;color:#555;flex:none}
.c{width:11px;height:11px;margin:1px;flex:none;box-sizing:border-box}.p{outline:2px solid #000;z-index:1}
#tip{position:fixed;background:#fff;border:1px solid #333;padding:4px;display:none;pointer-events:none;max-width:360px}
#tip img{max-width:160px;display:block}.lg span{display:inline-block;padding:2px 8px;color:#fff;margin-right:4px}</style>
<h3>__T__</h3><div class=lg id=lg></div>
<p>One row per decision; each cell is a queued job coloured by TRUE tier; black outline = picked.</p>
<div id=g></div><div id=tip></div>
<script>
var D=__D__;var g=document.getElementById('g'),tip=document.getElementById('tip');
D.tiers.forEach(function(n,i){var s=document.createElement('span');s.textContent=n;s.style.background=D.colors[i];document.getElementById('lg').appendChild(s)});
D.rows.forEach(function(r){var d=document.createElement('div');d.className='row';
 var l=document.createElement('div');l.className='lab';l.textContent='t='+r.t+'s q='+r.q;d.appendChild(l);
 r.c.forEach(function(c){var m=D.imgs[c[0]],e=document.createElement('div');
  e.className='c'+(c[1]?' p':'');e.style.background=D.colors[m.tier==null?3:m.tier];
  e.onmousemove=function(ev){tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.textContent='';
   var t=document.createElement('div');t.textContent=m.id+' | TRUE '+D.tiers[m.tier]+' | SPI '+m.spi.map(function(x){return x.toFixed(2)}).join('/')+' | rel '+m.rel;tip.appendChild(t);
   if(m.th){var i=document.createElement('img');i.src=m.th;tip.appendChild(i)}};
  e.onmouseleave=function(){tip.style.display='none'};d.appendChild(e)});
 g.appendChild(d)});
</script>
"""


def write_html(records, path, title, images_root=None):
    decs = [r for r in _decisions(records) if r["batch"]]
    imgs, idx, rows = [], {}, []
    for r in decs:
        picked, cells = set(r["batch"]), []
        for j in r["queue"]:
            if j["image_id"] not in idx:
                idx[j["image_id"]] = len(imgs)
                imgs.append(dict(id=j["image_id"], tier=j["true_tier"], spi=j["spi"], rel=round(j["relevance"], 3),
                                 th=_thumb(j["image_id"], images_root, Path(path).parent)))
            cells.append([idx[j["image_id"]], int(j["id"] in picked)])
        rows.append(dict(t=round(r["t"], 2), q=r["queue_size"], c=cells))
    data = json.dumps(dict(imgs=imgs, rows=rows, tiers=list(TIERS), colors=TIER_COLORS)).replace("</", "<\\/")
    Path(path).write_text(_PAGE.replace("__T__", html.escape(title)).replace("__D__", data), encoding="utf-8")


class _Capture:
    """tables shim that remembers the rows handed to the job builder (job id -> row)."""

    def __init__(self, tables):
        self.t, self.rows = tables, None

    def rows_to_jobs(self, rows, *a, **k):
        self.rows = rows
        return self.t.rows_to_jobs(rows, *a, **k)

    def rows_to_jobs_ordered(self, rows, *a, **k):
        self.rows = rows
        return self.t.rows_to_jobs_ordered(rows, *a, **k)

    def __getattr__(self, n):
        return getattr(self.t, n)


def trace_run(rows, latency, policy, workers=2, rho=1.2, seed=0, arrival="trace", split="val", subsample=None,
              sts_fusion=None, kwargs=None, tables=None):
    """One traced realisation; returns (records, metrics dict)."""
    if tables is None:
        from Evaluation.edge import tables
    rows, spec, max_batch = run_sweep._setup(rows, latency, split, None, None)
    if subsample and subsample < len(rows):
        keep = np.sort(np.random.default_rng(run_sweep._seed(seed, 777)).choice(len(rows), int(subsample), replace=False))
        rows = [rows[i] for i in keep]
    cap = _Capture(tables)
    jobs, _ = run_sweep._make_jobs(rows, latency, cap, workers, rho, 0, seed, max_batch, None, 0.0,
                                   {"arrival": arrival})
    truth = {j.id: j.truth for j in jobs}
    images = {j.id: cap.rows[j.id]["image_id"] for j in jobs}
    inner = run_sweep._build_policy(policy, kwargs or {}, truth)
    tp = TracingPolicy(inner, truth, images)
    env = EdgeEnv(jobs, latency, spec, workers, max_batch, Receiver(fusion=sts_fusion),
                  run_sweep._seed(seed, workers, round(rho * 1000), 0, 2))
    res = env.run(tp)
    m = semantic_metrics(tp.records, ctr=edge_metrics.ctr(res, False))
    m.update(policy=policy, workers=workers, rho=rho, seed=seed, arrival=arrival)
    return tp.records, m


def _slug(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-60:]


def write_outputs(records, m, out_dir, name, images_root=None):
    out, s = Path(out_dir), _slug(name)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / (s + "_trace.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(run_sweep._clean(r)) + "\n")
    (out / (s + "_metrics.json")).write_text(json.dumps(run_sweep._clean(m), indent=1), encoding="utf-8")
    (out / (s + "_replay.txt")).write_text("\n".join(replay_lines(records)) + "\n", encoding="utf-8")
    write_html(records, out / (s + "_replay.html"), name, images_root)


COLS = ("policy", "n_decisions", "semantic_precision", "inversions_per_100", "kendall_tau",
        "wait_critical", "wait_low", "ctr")


def main(argv=None):
    from Evaluation.edge import tables
    ap = argparse.ArgumentParser(description="Per-decision scheduling trace + semantic-ordering metrics")
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rho", type=float, default=1.2)
    ap.add_argument("--subsample", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arrival", choices=["poisson", "mmpp", "trace"], default="trace")
    ap.add_argument("--sts-fusion", default=None)
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, lat = tables.load_rows(a.table), tables.latency_from_profile_json(a.profile)
    fusion = run_sweep._load_fusion(a.sts_fusion)
    summary = []
    for p in run_sweep.resolve_policies(a.policies):
        recs, m = trace_run(rows, lat, p, a.workers, a.rho, a.seed, a.arrival, a.split, a.subsample, fusion,
                            tables=tables)
        write_outputs(recs, m, a.out, p, a.images_root)
        summary.append(m)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for m in summary:
            w.writerow([m.get(c) for c in COLS])
    return summary


if __name__ == "__main__":
    main()
