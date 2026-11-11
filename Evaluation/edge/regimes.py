"""Misspecified regimes (I10, plan v3 sec. 7) and sensitivity rows (I13, plan v3 sec. 9).

Every transform is seeded and acts only on what the policy sees or on the scenario itself:
``noisy_spi`` rewrites ``tier_probs`` (never ``tier``), ``wrong sigma_W`` swaps the latency profile
shown to the policy (the simulator keeps sampling the true one). ``tier_alt`` and ``critical_share``
deliberately change the truth: they are sensitivity rows, not regimes. Named bundles of knobs live in
config/edge/sim.yaml under ``scenarios``.

Knobs: arrival ('poisson'|'mmpp'|'trace'), mmpp {low, high, switch_rate}, spi_noise (rate in [0,1]),
sigma_w_scale (>0, scales the latency spread seen by the policy, mean kept), tier_key ('tier'|'tier_alt'),
critical_share (factor on the Critical share among crisis rows), weights_scale, loss (drop probability),
capture_jitter_s (capture earlier than tweet time by |N(0, s)|).
"""
from dataclasses import replace

import numpy as np

from Evaluation.edge.contracts import LatencyProfile
from Evaluation.edge.edge_config import load_config

KNOBS = ("arrival", "mmpp", "spi_noise", "sigma_w_scale", "tier_key", "critical_share",
         "weights_scale", "loss", "capture_jitter_s")


def resolve_scenarios(names):
    """Merge the named scenario bundles (later names win); unknown names raise."""
    known, out = load_config("sim").get("scenarios", {}), {}
    for n in names or ():
        if n not in known:
            raise ValueError("unknown scenario %r; known: %s" % (n, sorted(known)))
        out.update(known[n])
    bad = set(out) - set(KNOBS)
    if bad:
        raise ValueError("unknown scenario knobs: %s" % sorted(bad))
    return out


def is_crisis(row):
    return row.get("capture_time_s") is not None or bool(row.get("event"))


def inject_spi_noise(rows, rate, rng):
    """With probability ``rate`` per row, swap the mass of the argmax tier with a random other tier
    (a confidently wrong SPI label). Copies rows; ``tier`` and every other field are untouched."""
    out = []
    for r in rows:
        if rate > 0 and rng.random() < rate:
            p = list(r["tier_probs"])
            top = int(np.argmax(p))
            other = int(rng.choice([k for k in range(len(p)) if k != top]))
            p[top], p[other] = p[other], p[top]
            r = dict(r, tier_probs=p)
        out.append(r)
    return out


def use_tier_key(rows, key):
    """Use ``row[key]`` (e.g. tier_alt) as the truth tier."""
    return list(rows) if key in (None, "tier") else [dict(r, tier=int(r[key])) for r in rows]


def resample_critical(rows, factor, rng):
    """Change the Critical share among crisis rows by ``factor`` by dropping rows without replacement
    (factor>1 drops non-Critical crisis rows, factor<1 drops Critical ones). Everyday rows are kept."""
    rows = list(rows)
    if factor == 1:
        return rows
    crit = [i for i, r in enumerate(rows) if is_crisis(r) and r["tier"] == 0]
    rest = [i for i, r in enumerate(rows) if is_crisis(r) and r["tier"] != 0]
    n_c, n_r = len(crit), len(rest)
    if not n_c or not n_r:
        return rows
    share = min(0.95, n_c / (n_c + n_r) * factor)
    if factor > 1:   # keep all Critical, thin the rest to reach the share
        keep = min(n_r, int(round(n_c * (1 - share) / share)))
        drop = set(rng.choice(rest, size=n_r - keep, replace=False).tolist())
    else:
        keep = max(1, min(n_c, int(round(n_r * share / (1 - share)))))
        drop = set(rng.choice(crit, size=n_c - keep, replace=False).tolist())
    return [r for i, r in enumerate(rows) if i not in drop]


def drop_arrivals(jobs, p, rng):
    """Network loss: each job is lost independently with probability ``p`` (at least one is kept)."""
    if p <= 0:
        return list(jobs)
    keep = [j for j in jobs if rng.random() >= p]
    return keep or [jobs[0]]


def jitter_capture(jobs, sigma, rng):
    """Move each capture time earlier by |N(0, sigma)| (tweet time is after the true capture)."""
    if sigma <= 0:
        return list(jobs)
    return [replace(j, header=replace(j.header, capture_time=j.header.capture_time
                                      - abs(float(rng.normal(0.0, sigma))))) for j in jobs]


def scale_latency_variance(latency, factor):
    """Profile whose per-batch spread is widened (factor>1) or narrowed (<1); the mean is kept."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return latency
    out = {}
    for b, s in latency.samples.items():
        m = float(s.mean())
        x = (s / m) ** factor          # power transform keeps samples positive; spread grows with factor
        out[b] = x * (m / float(x.mean()))   # mean preserved exactly
    return LatencyProfile(out)


def scale_weights(spec, factor):
    return spec if factor == 1 else replace(spec, weights=tuple(w * factor for w in spec.weights))
