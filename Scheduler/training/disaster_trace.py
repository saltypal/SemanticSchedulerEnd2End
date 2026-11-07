"""Reproducible offered-workload construction after jobs have been selected.

Synthetic arrival mechanics only; not real UE traffic or emergency deadlines.
"""
from __future__ import annotations

import numpy as np


LOAD_FACTORS = (0.5, 0.8, 1.0, 1.2, 2.0)


def synthetic_lifetimes_from_ready(ready_ms, service_ms, median_service_multiples: float = 5.) -> np.ndarray:
    """Deadline sensitivity only; no inferred urgency or native ground truth."""
    ready, service = np.asarray(ready_ms, float), np.asarray(service_ms, float)
    if ready.shape != service.shape or ready.ndim != 1 or not len(ready):
        raise ValueError("Invalid synthetic lifetime inputs")
    if (not np.isfinite(ready).all() or not np.isfinite(service).all() or
            np.any(service <= 0) or not np.isfinite(median_service_multiples) or
            median_service_multiples <= 0):
        raise ValueError("Invalid synthetic lifetime values")
    return ready + median_service_multiples * float(np.median(service))


def fifo_max_ready_backlog(ready_ms, service_ms) -> int:
    """Largest number of jobs ready at a FIFO service decision."""
    ready, service = np.asarray(ready_ms, float), np.asarray(service_ms, float)
    order = np.argsort(ready, kind="stable")
    now = float(ready[order[0]])
    maximum = 0
    for position, index in enumerate(order):
        now = max(now, float(ready[index]))
        maximum = max(maximum, int(np.sum(ready[order[position:]] <= now + 1e-9)))
        now += service[index]
    return maximum


def construct_trace(service_ms, encode_ms, header_bits, ue_ids, *, mode: str,
                    load_factor: float, seed: int, control_rate_bps: float = 100_000.):
    """Return captures/ready times and measured load after choosing the jobs.

    The dynamic trace uses deterministic interarrival spacing with a seeded
    burst every ten jobs. All policies subsequently see these exact times.
    """
    service, encode, headers = [np.asarray(values, float) for values in (service_ms, encode_ms, header_bits)]
    users = np.asarray(ue_ids, str)
    n = len(service)
    if mode not in {"static", "dynamic"} or n < 2 or any(len(v) != n for v in (encode, headers, users)):
        raise ValueError("Invalid trace inputs")
    if (not np.isfinite(service).all() or not np.isfinite(encode).all() or
            not np.isfinite(headers).all() or np.any(service <= 0) or
            np.any(encode < 0) or np.any(headers < 0) or
            not np.isfinite(load_factor) or load_factor <= 0 or control_rate_bps <= 0):
        raise ValueError("Invalid trace timing or rate")
    control_ms = headers / control_rate_bps * 1000
    if mode == "static":
        captures = np.zeros(n, dtype=float)
        ready = np.full(n, float(np.max(encode + control_ms)))
        realized = None  # all-ready batch has no meaningful arrival-rate load
    else:
        generator = np.random.default_rng(seed)
        base_spacing = float(np.mean(service)) / load_factor
        spacings = np.full(n - 1, base_spacing)
        # Explicit modest bursts, followed by compensating longer gaps.
        for start in range(9, n - 1, 10):
            spacings[start] = base_spacing * generator.uniform(.05, .25)
            if start + 1 < n - 1:
                spacings[start + 1] = 2 * base_spacing - spacings[start]
        captures = np.r_[0., np.cumsum(spacings)]
        ready = np.empty(n)
        ue_free: dict[str, float] = {}
        for i in range(n):
            ue = str(users[i])
            encoded_at = max(float(captures[i]), ue_free.get(ue, 0.)) + encode[i]
            ue_free[ue] = encoded_at
            ready[i] = encoded_at + control_ms[i]
        span = max(float(ready.max() - ready.min()) + float(np.mean(service)), 1e-9)
        realized = float(service.sum() / span)
    backlog = fifo_max_ready_backlog(ready, service)
    if mode == "dynamic" and load_factor >= 1.2 and n >= 10 and backlog < 2:
        raise ValueError("Overloaded trace has no actual ready-set contention")
    return {"capture_ms": captures, "ready_ms": ready, "target_load_factor": load_factor,
            "realized_load_factor": realized, "fifo_max_ready_backlog": backlog,
            "generation": "selected jobs first; then seeded deterministic spacing/bursts"}
