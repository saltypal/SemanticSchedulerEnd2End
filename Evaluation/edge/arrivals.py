"""Arrival processes and trace-to-job assembly for the edge simulator.

Poisson and 2-state MMPP arrival times, tweet-trace compression, the rho <-> rate relation and
``build_jobs`` which assigns rows to arrivals sampling without replacement. Reuses ``contracts.Job``.
"""
import numpy as np

from Evaluation.edge.contracts import Job


def rate_for_rho(rho, latency, b, c):
    """Arrival rate lam (jobs/s) giving utilisation rho = lam * E[S_b] / (b * c)."""
    if rho <= 0 or b < 1 or c < 1:
        raise ValueError("rho, b and c must be positive")
    return rho * b * c / latency.mean(b)


def poisson_times(n, rate, rng):
    """n arrival times of a Poisson process of the given rate (starting after t=0)."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    return np.cumsum(rng.exponential(1.0 / rate, size=int(n)))


def mmpp_times(n, rates, switch_rate, rng):
    """n arrival times of a 2-state Markov-modulated Poisson process (symmetric switching)."""
    if len(rates) != 2 or min(rates) <= 0 or switch_rate <= 0:
        raise ValueError("need two positive rates and a positive switch rate")
    t, state, out = 0.0, 0, []
    while len(out) < n:
        gap = rng.exponential(1.0 / rates[state])
        sw = rng.exponential(1.0 / switch_rate)
        if gap <= sw:
            t += gap
            out.append(t)
        else:
            t += sw
            state = 1 - state
    return np.asarray(out)


def compress_trace(times, kappa):
    """Shift a real timestamp trace to start at 0 and divide time by kappa (kappa>1 = faster)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    t = np.sort(np.asarray(times, dtype=float))
    return (t - t[0]) / kappa


def build_jobs(rows, times, header_fn, truth_fn, rng, network_delay=0.0):
    """Assign distinct rows to arrival times (without replacement) and return sorted Jobs.

    ``header_fn(row, arrival_time, capture_time)`` -> Header, ``truth_fn(row)`` -> Truth.
    Job ids are the row indices. Duplicates exist only if the caller put them in ``rows``.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if len(times) > len(rows):
        raise ValueError("more arrivals (%d) than rows (%d): sampling is without replacement"
                         % (len(times), len(rows)))
    picks = rng.choice(len(rows), size=len(times), replace=False)
    jobs = []
    for idx, t in zip(picks, times):
        t = float(t)
        jobs.append(Job(int(idx), header_fn(rows[idx], t, t - network_delay), truth_fn(rows[idx])))
    return jobs
