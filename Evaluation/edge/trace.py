"""Trace-driven arrivals (plan v3 sec. 9, checklist I8).

CrisisMMD rows (non-null ``capture_time_s``, i.e. tweet time) arrive in tweet-time order, time-compressed
by kappa so that the offered load equals the target rho (rho = lam * E[S_B] / (B * c)). Everyday rows (null
capture time) are interleaved as a Poisson process at their share of the rate. Every row is replayed at
most once (no resampling). A realisation = permutation of the everyday arrivals + kappa + seed.

Mission per UE stream: there is one stream per mission id. A crisis row stays in the stream of its event's
mission (the table's ``mission``); an everyday row is assigned, uniformly at random per realisation, to the
mission of the stream it is mixed into. In both cases the header relevance is
``relevance_by_mission[mission]`` (D15), never the relevance of another mission.
"""
import numpy as np

from Evaluation.edge import arrivals


def is_trace_row(row):
    return row.get("capture_time_s") is not None


def n_missions_of(rows, default=4):
    for r in rows:
        if r.get("relevance_by_mission"):
            return len(r["relevance_by_mission"])
    return default


def with_mission(row, mission):
    """Copy of ``row`` re-assigned to ``mission`` with the matching per-mission relevance."""
    out = dict(row)
    out["mission"] = int(mission)
    rbm = row.get("relevance_by_mission")
    if rbm:
        out["relevance"] = float(rbm[int(mission)])
    return out


def kappa_for_rho(span, n_trace, trace_rate):
    """kappa so that ``n_trace`` rows spread over ``span`` seconds arrive at ``trace_rate`` per second."""
    if n_trace < 2 or span <= 0 or trace_rate <= 0:
        return 1.0
    return float(trace_rate * span / n_trace)


def trace_arrivals(rows, rate, rng, n_missions=None):
    """Return ``(ordered_rows, times, info)``: rows in arrival order with mission-consistent relevance.

    ``rate`` is the total target arrival rate (jobs/s) from ``arrivals.rate_for_rho``; crisis rows get
    the share ``n_crisis / n`` of it through kappa, everyday rows the rest as Poisson.
    """
    rows = list(rows)
    n = len(rows)
    if n == 0:
        raise ValueError("no rows")
    crisis = sorted((r for r in rows if is_trace_row(r)), key=lambda r: float(r["capture_time_s"]))
    everyday = [r for r in rows if not is_trace_row(r)]
    nm = n_missions or n_missions_of(rows)
    out_rows, out_times = [], []
    kappa = 1.0
    if crisis:
        share = len(crisis) / n
        ts = np.array([float(r["capture_time_s"]) for r in crisis])
        kappa = kappa_for_rho(ts[-1] - ts[0], len(crisis), share * rate)
        out_times.extend(arrivals.compress_trace(ts, kappa).tolist())
        out_rows.extend(with_mission(r, r["mission"]) for r in crisis)
    if everyday:
        perm = rng.permutation(len(everyday))
        missions = rng.integers(nm, size=len(everyday))
        t = arrivals.poisson_times(len(everyday), len(everyday) / n * rate, rng)
        out_times.extend(t.tolist())
        out_rows.extend(with_mission(everyday[p], m) for p, m in zip(perm, missions))
    order = np.argsort(np.asarray(out_times), kind="stable")
    times = np.asarray(out_times)[order]
    info = dict(kappa=float(kappa), n_crisis=len(crisis), n_everyday=len(everyday),
                duration=float(times[-1]) if len(times) else 0.0)
    return [out_rows[i] for i in order], times, info
