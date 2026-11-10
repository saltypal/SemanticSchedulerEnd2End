"""Offline bounds on achievable value for short traces (plan sec. 5.2 value, single-job batches).

``clairvoyant_bound``: time-indexed MILP (scipy.optimize.milp) using true values.
``greedy_upper_bound``: trivially valid upper bound, every job decoded instantly on arrival.
"""
import math

import numpy as np


def _zeta(spec, tier, first):
    return 1.0 if first else (spec.zeta_critical_copy if tier == 0 else 0.0)


def greedy_upper_bound(jobs, spec):
    """Sum of w*u*zeta if each job were decoded immediately (g = 1). Within a dup_group the
    best-valued copy is credited as the first (full) copy, the others get zeta (valid bound)."""
    total, groups = 0.0, {}
    for j in jobs:
        t = j.truth
        v = spec.weights[t.tier] * t.fidelity
        if t.dup_group is None:
            total += v
        else:
            groups.setdefault(t.dup_group, []).append((v, t.tier))
    for members in groups.values():
        best = max(members)
        total += best[0] + sum(v * _zeta(spec, k, False) for v, k in members) - best[0] * _zeta(spec, best[1], False)
    return float(total)


def clairvoyant_bound(jobs, latency, spec, workers, max_batch=1, dt=0.5):
    """Maximise the sum of true values. Returns (optimum, schedule[(job_id, start, finish)]).

    x[i,s] = 1 if job i starts at grid step s (not before its arrival, finishing before H_k);
    service = floor(mean(1)/dt) steps (>=1) and start = floor(arrival/dt), i.e. both rounded in the
    optimistic direction so the grid problem relaxes the continuous one (use dt <= mean(1)); at most ``workers`` jobs in service at any step; each job
    decoded at most once. Duplicates: one copy per dup_group may earn full value (flag f<=x),
    the rest earn zeta*value. The full-value copy is chosen freely rather than by decode order,
    so with duplicates the result is a (still valid) upper bound on the sequential optimum.
    """
    if max_batch != 1:
        raise ValueError("clairvoyant_bound supports max_batch=1 only")
    if len(jobs) > 200:
        raise ValueError("clairvoyant_bound is for traces of <= 200 jobs")
    if dt <= 0 or workers < 1:
        raise ValueError("dt must be > 0 and workers >= 1")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as e:  # pragma: no cover
        raise ImportError("clairvoyant_bound needs scipy>=1.9 (scipy.optimize.milp)") from e
    if not jobs:
        return 0.0, []
    svc = max(1, math.floor(latency.mean(1) / dt + 1e-9))   # round DOWN: relaxation
    var = []   # (job index, start step, value if full)
    for i, j in enumerate(jobs):
        k, t = j.truth.tier, j.truth
        s = int(math.floor(j.header.arrival_time / dt + 1e-9))   # round DOWN: relaxation
        while True:
            delay = (s + svc) * dt - j.header.capture_time
            if delay >= spec.hard_deadline(k):
                break
            var.append((i, s, spec.weights[k] * t.fidelity * spec.g(k, delay)))
            s += 1
    if not var:
        return 0.0, []
    nv = len(var)
    horizon = max(s for _, s, _ in var) + svc
    gi = {i: j.truth.dup_group for i, j in enumerate(jobs) if j.truth.dup_group is not None}
    groups = sorted(set(gi.values()))

    cost = np.zeros(2 * nv)
    ub = np.ones(2 * nv)
    for idx, (i, s, v) in enumerate(var):
        zc = _zeta(spec, jobs[i].truth.tier, False) if i in gi else 1.0
        cost[idx], cost[nv + idx] = -v * zc, -v * (1.0 - zc)
        if i not in gi:
            ub[nv + idx] = 0.0
    rows = len(jobs) + horizon + nv + len(groups)
    A = lil_matrix((rows, 2 * nv))
    hi = np.zeros(rows)
    r = 0
    for i in range(len(jobs)):
        for idx, (ii, _, _) in enumerate(var):
            if ii == i:
                A[r, idx] = 1
        hi[r] = 1
        r += 1
    for t in range(horizon):
        for idx, (_, s, _) in enumerate(var):
            if s <= t < s + svc:
                A[r, idx] = 1
        hi[r] = workers
        r += 1
    for idx in range(nv):
        A[r, nv + idx], A[r, idx] = 1, -1
        r += 1
    for g in groups:
        for idx, (i, _, _) in enumerate(var):
            if gi.get(i) == g:
                A[r, nv + idx] = 1
        hi[r] = 1
        r += 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), np.full(rows, -np.inf), hi),
               integrality=np.ones(2 * nv), bounds=Bounds(np.zeros(2 * nv), ub))
    if not res.success:
        raise RuntimeError("MILP failed: %s" % res.message)
    sched = [(jobs[i].id, s * dt, (s + svc) * dt)
             for idx, (i, s, _) in enumerate(var) if res.x[idx] > 0.5]
    sched.sort(key=lambda x: x[1])
    return float(-res.fun), sched
