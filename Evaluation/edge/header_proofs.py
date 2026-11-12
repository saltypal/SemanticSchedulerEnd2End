"""Header-proof runner (checklist I9): does each 64-byte header field earn its bytes?

(a) Field drop-out: the same policy on the same paired traces with header fields neutralised - SPI ->
    uniform prior, relevance -> constant, keys -> independent random keys per job (no dedup possible),
    STS -> empty - one at a time (``full_minus_<field>``) and cumulatively
    (none -> spi -> spi+relevance -> spi+relevance+keys -> full).
(b) Equal-bytes baseline ``equal_bytes``: the 64 bytes are kept but carry NO semantic signal. No thumbnail
    data exists in the table, so this is the "payload size + random bytes" variant: payload_bytes is kept,
    SPI is the uniform prior, and relevance, both keys and the 13 STS values are filled with random bytes
    (decoded through the same dequantisation). Information, not byte count, must explain any gain.
(c) Receiver dedup precision/recall against ``dup_group``: all jobs of a realisation are delivered in
    arrival order; a job is "predicted duplicate" when ``Receiver.dup_prob >= threshold`` and truly a
    duplicate when its group was already delivered. Reported for each threshold.
(d) Overhead 64 / (64 + payload_bytes) per row (mean, min, max). NOTE the table's payload_bytes is a
    placeholder until codec v1 exists.

Output: tidy CSV (experiment, variant, policy, workers, rho, metric, value) and a JSON summary.
"""
import argparse
import csv
import dataclasses
import os

import numpy as np

from Evaluation.edge import regimes, run_sweep
from Evaluation.edge.receiver import Receiver

HEADER_BYTES = 64
FIELDS = ("spi", "relevance", "keys", "sts")
CUMULATIVE = (("none", ()), ("spi", ("spi",)), ("spi+relevance", ("spi", "relevance")),
              ("spi+relevance+keys", ("spi", "relevance", "keys")), ("full", FIELDS))
METRICS = ("ctr_ungated", "value", "value_per_gpu_s", "low_starvation", "redundant_per_100")
THRESHOLDS = (0.5, 0.95)
UNIFORM = (0.25, 0.25, 0.25, 0.25)
CONST_RELEVANCE = 0.5


def _rng(job_id, salt):
    return np.random.default_rng(np.random.SeedSequence([int(job_id) & 0xFFFFFFFF, salt]))


def neutralise(header, fields, job_id):
    """Return ``header`` with the named fields ('spi','relevance','keys','sts') neutralised."""
    kw = {}
    if "spi" in fields:
        kw["tier_probs"] = UNIFORM
    if "relevance" in fields:
        kw["relevance"] = CONST_RELEVANCE
    if "keys" in fields:
        r = _rng(job_id, 1)
        kw["sem_key"], kw["perc_key"] = (int(x) for x in r.integers(0, 2 ** 63, size=2, dtype=np.int64))
    if "sts" in fields:
        kw["sts"] = ()
    return dataclasses.replace(header, **kw)


def equal_bytes(header, job_id):
    """Same 64 bytes, no semantics: uniform SPI, random relevance / keys / STS (see module docstring)."""
    r = _rng(job_id, 2)
    h = neutralise(header, ("spi", "keys"), job_id)
    return dataclasses.replace(h, relevance=float(r.integers(0, 256)) / 255.0,
                               sts=tuple(float(x) / 255.0 for x in r.integers(0, 256, size=13)))


def variants():
    """Ordered ``{name: header_mask or None}``: cumulative levels, leave-one-out, equal-bytes."""
    out = {}
    for name, kept in CUMULATIVE:
        drop = tuple(f for f in FIELDS if f not in kept)
        out["cum:" + name] = (lambda h, i, d=drop: neutralise(h, d, i)) if drop else None
    for f in FIELDS:
        out["drop:" + f] = lambda h, i, f=f: neutralise(h, (f,), i)
    out["equal_bytes"] = equal_bytes
    return out


def dedup_pr(jobs, thresholds=THRESHOLDS, size=512):
    """Precision / recall of receiver duplicate detection vs ``dup_group`` over arrival-ordered jobs."""
    jobs = sorted(jobs, key=lambda j: (j.header.arrival_time, j.id))
    rx, seen = Receiver(size=size), set()
    pred = {t: [0, 0, 0] for t in thresholds}   # tp, fp, fn
    n_dup = 0
    for j in jobs:
        p = rx.dup_prob(j.header)
        g = j.truth.dup_group
        truth = g is not None and g in seen
        n_dup += truth
        for t in thresholds:
            hit = p >= t
            pred[t][0] += hit and truth
            pred[t][1] += hit and not truth
            pred[t][2] += (not hit) and truth
        rx.deliver(j.header)
        if g is not None:
            seen.add(g)
    out = {}
    for t, (tp, fp, fn) in pred.items():
        out[t] = dict(precision=tp / (tp + fp) if tp + fp else None,
                      recall=tp / (tp + fn) if tp + fn else None, tp=tp, fp=fp, fn=fn,
                      n_true_dups=n_dup)
    return out


def overhead(rows):
    """Per-row overhead 64 / (64 + payload_bytes): mean, min, max."""
    o = np.array([HEADER_BYTES / (HEADER_BYTES + float(r["payload_bytes"])) for r in rows])
    return dict(mean=float(o.mean()), min=float(o.min()), max=float(o.max()), n=int(len(o)))


def run_header_proofs(rows, latency, split, policies, workers, rhos, realisations, base_seed=0,
                      tables=None, scenario_names=None, arrival=None, subsample=None, n_boot=50,
                      max_batch=None, policy_kwargs=None, sts_fusion=None):
    """Run (a)-(d); returns ``{'config', 'ablation': [cell...], 'dedup': {...}, 'overhead': {...}}``."""
    if tables is None:
        from Evaluation.edge import tables
    scen = regimes.resolve_scenarios(scenario_names)
    if arrival:
        scen["arrival"] = arrival
    split_rows = [r for r in rows if r.get("split") == split] if split else list(rows)
    ablation = []
    for name, mask in variants().items():
        res = run_sweep.run_sweep(rows, latency, split, policies, workers, rhos, realisations,
                                  base_seed, max_batch, n_boot=n_boot, tables=tables, scenario=scen,
                                  subsample=subsample, header_mask=mask, policy_kwargs=policy_kwargs,
                                  sts_fusion=sts_fusion)
        for c in res["cells"]:
            ablation.append(dict(c, variant=name))
    # dedup P/R on the first cell's paired traces (receiver-only, policy independent)
    sub = run_sweep._setup(rows, latency, split, None, max_batch)[0]
    if subsample and subsample < len(sub):
        keep = np.sort(np.random.default_rng(run_sweep._seed(base_seed, 777)).choice(len(sub), int(subsample), replace=False))
        sub = [sub[i] for i in keep]
    b = max([x for x in latency.batch_sizes if x <= (max_batch or max(latency.batch_sizes))]
            or [latency.batch_sizes[0]])
    dedup = {}
    for w in workers:
        for rho in rhos:
            jobs, _ = run_sweep._make_jobs(sub, latency, tables, w, rho, 0, base_seed, b, None, 0.0, scen)
            dedup["w%s_rho%s" % (w, rho)] = dedup_pr(jobs)
    return dict(config=dict(split=split, policies=list(policies), workers=list(workers),
                            rhos=list(rhos), realisations=realisations, base_seed=base_seed,
                            scenario=scen, n_images=len(sub), git_sha=run_sweep.git_sha()),
                ablation=ablation, dedup=dedup, overhead=overhead(split_rows))


def tidy_rows(result):
    """Flatten a result into tidy ``(experiment, variant, policy, workers, rho, metric, value)`` rows."""
    out = []
    for c in result["ablation"]:
        exp = "equal_bytes" if c["variant"] == "equal_bytes" else "ablation"
        for m in METRICS:
            out.append((exp, c["variant"], c["policy"], c["workers"], c["rho"], m, c[m]))
    for key, per_t in result["dedup"].items():
        w, rho = key[1:].split("_rho")
        for t, d in per_t.items():
            for m in ("precision", "recall"):
                out.append(("dedup", "thr=%s" % t, "receiver", int(w), float(rho), m, d[m]))
    for m, v in result["overhead"].items():
        out.append(("overhead", "64/(64+payload)", "", "", "", m, v))
    return out


def save(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "header_proofs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("experiment", "variant", "policy", "workers", "rho", "metric", "value"))
        for r in tidy_rows(result):
            w.writerow(["" if v is None or (isinstance(v, float) and v != v) else v for v in r])
    run_sweep.save_results(result, os.path.join(out_dir, "header_proofs.json"))


def main(argv=None):
    from Evaluation.edge import tables
    ap = argparse.ArgumentParser(description="Header-field proofs (drop-out, equal bytes, dedup P/R, overhead)")
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--policies", nargs="+", default=["sv_dpp"])
    ap.add_argument("--workers", nargs="+", type=int, default=[2])
    ap.add_argument("--rhos", nargs="+", type=float, default=[1.0])
    ap.add_argument("--realisations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arrival", choices=["poisson", "mmpp", "trace"])
    ap.add_argument("--scenario", nargs="+", default=[])
    ap.add_argument("--subsample", type=int)
    ap.add_argument("--sts-fusion", default=None, help="sts_fusion.json (D19)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    res = run_header_proofs(tables.load_rows(a.table), tables.latency_from_profile_json(a.profile),
                            a.split, a.policies, a.workers, a.rhos, a.realisations, a.seed, tables,
                            a.scenario, a.arrival, a.subsample,
                            **({"sts_fusion": run_sweep._load_fusion(a.sts_fusion)} if a.sts_fusion else {}))
    save(res, a.out_dir)
    return res


if __name__ == "__main__":
    main()
