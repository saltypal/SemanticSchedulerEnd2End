"""Experiment runner: policies x workers x rho x realisations on a per-image table (plan sec. 9).

Library + CLI. Reuses ``tables`` (rows -> jobs, latency profile), ``arrivals`` (rate_for_rho,
poisson/mmpp times), ``env.EdgeEnv``, ``metrics`` (summaries, bootstrap CI) and
``Scheduler.edge.registry``. Traces are paired: a (workers, rho, realisation) cell uses the same
seed for every policy, derived from ``base_seed`` through ``SeedSequence``. Policy groups from
config/edge/policies.yaml may be given by name. Arrival modes: poisson (default), mmpp, trace (v3 sec. 9,
see ``trace.py``; kappa is recorded per cell). Named scenarios from config/edge/sim.yaml (``regimes.py``:
misspecified regimes and sensitivity rows) are selectable with ``--scenario``. ``val_tuning`` grid-searches the V of SV-DPP.
"""
import argparse
import dataclasses
import json
import math
import subprocess
import time

import numpy as np

from Evaluation.edge import arrivals, metrics, regimes, trace
from Evaluation.edge.contracts import ValueSpec
from Evaluation.edge.edge_config import load_config, repo_root
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry

SCALARS = ("ctr_ungated", "value", "value_per_gpu_s", "low_starvation",
           "redundant_per_100")
NAN = float("nan")


def resolve_policies(names):
    """Expand group names (main_table, ...) to policy names; raise on unknown names."""
    groups, known = load_config("policies"), registry.all_policies()
    out = []
    for n in names:
        for m in (groups[n] if n in groups else [n]):
            if m not in known:
                raise ValueError("unknown policy %r; known: %s" % (m, sorted(known)))
            if m not in out:
                out.append(m)
    return out


def _seed(base, *parts):
    return int(np.random.SeedSequence([int(base)] + [int(p) for p in parts]).generate_state(1)[0])


def _build_policy(name, kwargs, truth_by_id):
    factory = registry.all_policies()[name]
    if getattr(factory, "uses_truth", False):
        return factory(truth_by_id, **kwargs)
    return factory(**kwargs)


def _ctr_rows(rows, d0):
    crit = [r for r in rows if r["tier"] == 0]
    if not crit:
        return NAN
    return float(np.mean([r["status"] == "decoded" and r["delay"] <= d0
                          for r in crit]))


def _mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else NAN


def _make_jobs(rows, latency, tables, workers, rho, rep, base_seed, max_batch, mmpp, network_delay,
               scen=None, header_mask=None):
    """Return ``(jobs, info)`` for one realisation; identical for every policy of the cell (paired)."""
    scen = scen or {}
    b = max([x for x in latency.batch_sizes if x <= max_batch] or [latency.batch_sizes[0]])
    rate = arrivals.rate_for_rho(rho, latency, b, workers)
    cell_seed = (workers, round(rho * 1000), rep)
    rng = np.random.default_rng(_seed(base_seed, *cell_seed, 1))
    rrng = np.random.default_rng(_seed(base_seed, *cell_seed, 3))    # regime / sensitivity randomness
    mode = scen.get("arrival") or ("mmpp" if mmpp else "poisson")
    if mode not in ("poisson", "mmpp", "trace"):
        raise ValueError("unknown arrival mode %r" % (mode,))
    mmpp = scen.get("mmpp") or mmpp or (load_config("sim")["mmpp"] if mode == "mmpp" else None)
    rows = regimes.use_tier_key(rows, scen.get("tier_key"))
    rows = regimes.resample_critical(rows, scen.get("critical_share", 1.0), rrng)
    rows = regimes.inject_spi_noise(rows, scen.get("spi_noise", 0.0), rrng)
    info = dict(kappa=None, arrival=mode)
    if mode == "trace":
        rows, times, tinfo = trace.trace_arrivals(rows, rate, rng)
        info.update(tinfo)
        jobs = tables.rows_to_jobs_ordered(rows, times, network_delay=network_delay)
    else:
        n = len(rows)
        if mode == "mmpp":
            times = arrivals.mmpp_times(n, (rate * mmpp["low"], rate * mmpp["high"]),
                                        mmpp["switch_rate"], rng)
        else:
            times = arrivals.poisson_times(n, rate, rng)
        jobs = tables.rows_to_jobs(rows, times, rng, network_delay=network_delay)
    n0 = len(jobs)
    jobs = regimes.drop_arrivals(jobs, scen.get("loss", 0.0), rrng)
    info["n_lost"] = n0 - len(jobs)
    jobs = regimes.jitter_capture(jobs, scen.get("capture_jitter_s", 0.0), rrng)
    if header_mask is not None:   # I9 field drop-out: header_mask(header, job_id) -> header
        jobs = [dataclasses.replace(j, header=header_mask(j.header, j.id)) for j in jobs]
    return jobs, info


def run_cell(rows, latency, spec, policy, kwargs, workers, rho, realisations, base_seed,
             max_batch, mmpp=None, network_delay=0.0, n_boot=200, tables=None, scenario=None,
             header_mask=None):
    """Run one (policy, workers, rho) cell over all realisations; return the aggregate dict."""
    if tables is None:
        from Evaluation.edge import tables
    t0 = time.perf_counter()
    scen = scenario or {}
    spec = regimes.scale_weights(spec, scen.get("weights_scale", 1.0))
    seen = regimes.scale_latency_variance(latency, scen.get("sigma_w_scale", 1.0))
    summaries, pooled, infos = [], [], []
    for rep in range(realisations):
        jobs, info = _make_jobs(rows, latency, tables, workers, rho, rep, base_seed, max_batch,
                                mmpp, network_delay, scen, header_mask)
        infos.append(info)
        truth = {j.id: j.truth for j in jobs}
        env = EdgeEnv(jobs, latency, spec, workers, max_batch, Receiver(),
                      _seed(base_seed, workers, round(rho * 1000), rep, 2), policy_latency=seen)
        res = env.run(_build_policy(policy, kwargs, truth))
        summaries.append(metrics.summarize(res))
        pooled.extend(res.measured())
    d0, rng = spec.soft_deadline[0], np.random.default_rng(_seed(base_seed, 99))
    cell = dict(policy=policy, kwargs=kwargs, workers=workers, rho=rho, realisations=realisations,
                n_measured=len(pooled), scenario=scen)
    ks = [i["kappa"] for i in infos if i["kappa"] is not None]
    cell["kappa"] = float(np.mean(ks)) if ks else None
    cell["kappas"] = ks
    for key in SCALARS:
        cell[key] = _mean([s[key] for s in summaries])
    cell["false_purge_rate"] = _mean([s["false_purges"]["rate"] for s in summaries])
    cell["timely"] = {k: _mean([s["timely"][k]["rate"] for s in summaries])
                      for k in range(len(spec.weights))}
    cell["decision_time_mean_us"] = _mean([s["decision_time"]["mean_us"] for s in summaries])
    ci = {}
    if pooled:
        has_crit = any(r["tier"] == 0 for r in pooled)
        ci["ctr_ungated"] = (metrics.bootstrap_ci(pooled, lambda s: _ctr_rows(s, d0), n_boot, rng)
                             if has_crit else (NAN, NAN))
        ci["value_per_image"] = metrics.bootstrap_ci(
            pooled, lambda s: float(np.mean([r["value"] for r in s])), n_boot, rng)
    cell["ci95"] = {k: list(v) for k, v in ci.items()}
    cell["seconds"] = time.perf_counter() - t0
    return cell


def _setup(rows, latency, split, spec, max_batch):
    rows = [r for r in rows if r.get("split") == split] if split else list(rows)
    if not rows:
        raise ValueError("no rows for split %r" % (split,))
    spec = spec or ValueSpec.from_config(load_config("sim"))
    return rows, spec, int(max_batch or max(latency.batch_sizes))


def run_sweep(rows, latency, split, policies, workers, rhos, realisations, base_seed=0,
              max_batch=None, spec=None, mmpp=None, network_delay=0.0, n_boot=200,
              policy_kwargs=None, tables=None, scenario=None, scenario_names=None, arrival=None,
              subsample=None, header_mask=None):
    """Sweep policies x workers x rho x realisations; returns a JSON-ready dict.

    ``scenario`` is a knob dict, ``scenario_names`` names from config sim.yaml merged under it;
    ``arrival`` ('poisson'|'mmpp'|'trace') overrides the scenario's arrival mode; ``subsample`` keeps a
    seeded random subset (without replacement) of the split's rows.
    """
    names = resolve_policies(policies)
    rows, spec, max_batch = _setup(rows, latency, split, spec, max_batch)
    scen = dict(regimes.resolve_scenarios(scenario_names), **(scenario or {}))
    if arrival:
        scen["arrival"] = arrival
    if subsample and subsample < len(rows):
        keep = np.sort(np.random.default_rng(_seed(base_seed, 777)).choice(len(rows), int(subsample), replace=False))
        rows = [rows[i] for i in keep]
    kw = policy_kwargs or {}
    t0 = time.perf_counter()
    cells = [run_cell(rows, latency, spec, p, kw.get(p, {}), w, r, realisations, base_seed,
                      max_batch, mmpp, network_delay, n_boot, tables, scen, header_mask)
             for w in workers for r in rhos for p in names]
    return dict(config=dict(split=split, policies=names, workers=list(workers), rhos=list(rhos),
                            realisations=realisations, base_seed=base_seed, max_batch=max_batch,
                            mmpp=mmpp, network_delay=network_delay, n_boot=n_boot,
                            n_images=len(rows), scenario_names=list(scenario_names or []),
                            scenario=scen),
                git_sha=git_sha(), seconds=time.perf_counter() - t0, cells=cells)


def val_tuning(rows, latency, v_grid, workers=2, rho=1.0, realisations=3, base_seed=0,
               max_batch=None, spec=None, n_boot=200, split="val", tables=None):
    """Grid-search the V of SV-DPP on ``split``; best = highest CTR@D (ctr_ungated, the headline metric). Records every point."""
    rows, spec, max_batch = _setup(rows, latency, split, spec, max_batch)
    points = [run_cell(rows, latency, spec, "sv_dpp", dict(V=float(v)), workers, rho, realisations,
                       base_seed, max_batch, None, 0.0, n_boot, tables) for v in v_grid]
    best = max(points, key=lambda c: -math.inf if math.isnan(c["ctr_ungated"]) else c["ctr_ungated"])
    return dict(best_V=best["kwargs"]["V"], best=best, grid=points, workers=workers, rho=rho,
                split=split)


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root()),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def save_results(results, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_clean(results), handle, indent=1)


def main(argv=None):
    from Evaluation.edge import tables
    ap = argparse.ArgumentParser(description="Policy sweep on a per-image table")
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--policies", nargs="+", default=["main_table"])
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    ap.add_argument("--realisations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--mmpp", nargs=3, type=float, metavar=("LOW", "HIGH", "SWITCH_RATE"),
                    help="rate multipliers and switching rate for an MMPP arrival process")
    ap.add_argument("--arrival", choices=["poisson", "mmpp", "trace"],
                    help="arrival mode (default poisson; --mmpp implies mmpp); trace = v3 sec. 9")
    ap.add_argument("--scenario", nargs="+", default=[], metavar="NAME",
                    help="named scenarios from config/edge/sim.yaml (regimes / sensitivity rows)")
    ap.add_argument("--subsample", type=int, help="seeded random subset of the split's rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    mmpp = dict(low=a.mmpp[0], high=a.mmpp[1], switch_rate=a.mmpp[2]) if a.mmpp else None
    res = run_sweep(tables.load_rows(a.table), tables.latency_from_profile_json(a.profile),
                    a.split, a.policies, a.workers, a.rhos, a.realisations, a.seed,
                    mmpp=mmpp, n_boot=a.n_boot, tables=tables, scenario_names=a.scenario,
                    arrival=a.arrival, subsample=a.subsample)
    save_results(res, a.out)
    return res


if __name__ == "__main__":
    main()
