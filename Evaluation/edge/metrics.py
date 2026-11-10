"""Metrics computed from a ``RunResult`` (measured, post-warmup jobs only) plus bootstrap CIs.

Pure functions over the outcome rows produced by ``env.EdgeEnv``; no simulator state is touched.
"""
import numpy as np

NAN = float("nan")


def _mean(x):
    return float(np.mean(x)) if len(x) else NAN


def _decoded(rows):
    return [r for r in rows if r["status"] == "decoded"]


def timely_by_tier(result):
    """Per tier: P(decoded and delay <= H_k), the target eta_k and whether it is met."""
    spec, rows, out = result.spec, result.measured(), {}
    for k in range(len(spec.weights)):
        rk = [r for r in rows if r["tier"] == k]
        ok = [r["status"] == "decoded" and r["delay"] <= spec.hard_deadline(k) for r in rk]
        rate = _mean(ok)
        out[k] = dict(n=len(rk), rate=rate, eta=spec.eta[k], met=bool(rate >= spec.eta[k]))
    return out


def ctr(result, gated):
    """Critical timely rate: true-Critical decoded within D_C; ``gated`` also requires u == 1."""
    d0 = result.spec.soft_deadline[0]
    crit = [r for r in result.measured() if r["tier"] == 0]
    ok = [r["status"] == "decoded" and r["delay"] <= d0 and (r["u"] == 1 or not gated)
          for r in crit]
    return _mean(ok)


def normalised_value(result, reference=None):
    total = sum(r["value"] for r in result.measured())
    return total / reference if reference else total


def value_per_gpu_second(result):
    total = sum(r["value"] for r in result.measured())
    return total / result.busy_time if result.busy_time else NAN


def p95_sojourn(result):
    out = {}
    for k in range(len(result.spec.weights)):
        d = [r["delay"] for r in _decoded(result.measured()) if r["tier"] == k]
        out[k] = float(np.quantile(d, 0.95)) if d else NAN
    return out


def low_starvation(result):
    low = [r for r in result.measured() if r["tier"] == len(result.spec.weights) - 1]
    return _mean([r["status"] != "decoded" for r in low])


def redundant_per_100(result):
    dec = _decoded(result.measured())
    return 100.0 * sum(r["dup_flag"] for r in dec) / len(dec) if dec else NAN


def false_purges(result):
    """True-Critical jobs purged by the policy before their hard deadline H_0."""
    h0 = result.spec.hard_deadline(0)
    crit = [r for r in result.measured() if r["tier"] == 0]
    fp = [r for r in crit if r["status"] == "purged" and r["delay"] < h0]
    return dict(count=len(fp), n_critical=len(crit),
                rate=len(fp) / len(crit) if crit else NAN)


def breakdown(result, key):
    """Group measured rows by ``key`` (row field or callable): count, decoded, value, timely."""
    spec, groups = result.spec, {}
    for r in result.measured():
        groups.setdefault(key(r) if callable(key) else r[key], []).append(r)
    out = {}
    for g, rs in groups.items():
        timely = [r["status"] == "decoded" and r["delay"] <= spec.hard_deadline(r["tier"])
                  for r in rs]
        out[g] = dict(n=len(rs), decoded=len(_decoded(rs)),
                      mean_value=_mean([r["value"] for r in rs]), timely=_mean(timely))
    return out


def decision_time_summary(result):
    us = np.asarray(result.decision_times) * 1e6
    if not len(us):
        return dict(n=0, mean_us=NAN, p50_us=NAN, p95_us=NAN, max_us=NAN)
    return dict(n=len(us), mean_us=float(us.mean()), p50_us=float(np.median(us)),
                p95_us=float(np.quantile(us, 0.95)), max_us=float(us.max()))


def summarize(result, reference=None):
    return dict(timely=timely_by_tier(result), ctr_gated=ctr(result, True),
                ctr_ungated=ctr(result, False), value=normalised_value(result, reference),
                value_per_gpu_s=value_per_gpu_second(result), p95_sojourn=p95_sojourn(result),
                low_starvation=low_starvation(result), redundant_per_100=redundant_per_100(result),
                false_purges=false_purges(result), by_source=breakdown(result, "source"),
                by_event=breakdown(result, "event"), by_u=breakdown(result, "u"),
                decision_time=decision_time_summary(result))


def bootstrap_ci(rows, stat_fn, n_boot, rng, group=None, alpha=0.05):
    """Percentile CI of ``stat_fn(sample)``, resampling rows (or whole groups) with replacement.

    ``group`` is a row field name or callable giving the cluster (e.g. event) of a row.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to bootstrap")
    if group is None:
        units = [[r] for r in rows]
    else:
        cl = {}
        for r in rows:
            cl.setdefault(group(r) if callable(group) else r[group], []).append(r)
        units = list(cl.values())
    stats = []
    for _ in range(int(n_boot)):
        idx = rng.integers(len(units), size=len(units))
        stats.append(stat_fn([r for i in idx for r in units[i]]))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
