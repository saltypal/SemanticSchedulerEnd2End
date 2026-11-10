"""Queueing-theory index baselines (plan sec. 8): generalised c-mu/h and a discrete Whittle index.

Reuses ``Evaluation.edge.contracts`` (Policy, State, Decision, top_batch). No policy reads truth;
the predicted tier distribution is ``header.tier_probs`` and indices are expectations over it.

Simplifications (documented on purpose):
* gen_cmu_h (after Long et al. 2020): holding-cost rate gamma_k(x) = w_k * (1 + x / D_k), abandonment
  hazard h_k(x) = 1 / max(H_k - x, eps) (deterministic hard deadline stands in for a patience law),
  index = E_k[gamma_k(age) * mu / (1 + h_k(age))] with mu = 1 / mean service time of batch size 1.
* whittle_abandon (after Aalto, arXiv 2609.19792): discrete-time Whittle index with geometric
  abandonment. Time is cut into slots of ``slot_s`` seconds (default 1.0). The service-time pmf is the
  histogram of the batch-size-1 latency samples in slots (ceil, min 1 slot). With b = 1 - theta,
  theta = slot_s / H_k, penalty d = w_k, slot age x, S the service time in slots:
      N(D) = sum_{i=1..D} b^(i-1) P(S=i) * (gamma(x + i + 1) / theta + d)
      T(D) = sum_{i=1..D} b^(i-1) P(S>=i)          W = max_D N(D) / T(D)
  and the policy index is E_k[W_k]. gamma is evaluated in seconds at ``age + (i + 1) * slot_s``.

Observe payloads: this module's policies keep no state and ignore ``observe``.
"""
import numpy as np

from Evaluation.edge.contracts import Decision, Policy, top_batch
from Scheduler.edge.classic import batch_size


def predicted_tier(tier_probs):
    """Argmax of the calibrated tier posterior."""
    return int(np.argmax(tier_probs))


def holding_cost(spec, k, age):
    """gamma_k(x) = w_k * (1 + x / D_k)."""
    return spec.weights[k] * (1.0 + age / spec.soft_deadline[k])


def abandon_hazard(spec, k, age, eps=0.1):
    """h_k(x) = 1 / max(H_k - x, eps)."""
    return 1.0 / max(spec.hard_deadline(k) - age, eps)


def rank_by_index(state, index_fn):
    """Queue ids sorted by descending ``index_fn(job)``; ties broken by older age then id."""
    ordered = sorted(state.queue, key=lambda j: (-index_fn(j), -j.age, j.id))
    return [j.id for j in ordered]


def service_pmf(samples, slot_s):
    """pmf over service slots 1..n from latency samples (ceil to slots, at least one slot)."""
    if slot_s <= 0:
        raise ValueError("slot_s must be positive")
    slots = np.maximum(1, np.ceil(np.asarray(samples, dtype=float) / slot_s - 1e-12).astype(int))
    return np.bincount(slots)[1:] / len(slots)


def whittle_index(pmf, age_slots, gamma, theta, penalty):
    """Discrete Whittle index W = max_D N(D)/T(D); ``gamma`` maps a slot age to a cost rate.

    ``pmf[i-1] = P(S=i)``. Vectorised over the service horizon with cumulative sums.
    """
    pmf = np.asarray(pmf, dtype=float)
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie in (0, 1)")
    i = np.arange(1, len(pmf) + 1)
    surv = 1.0 - np.concatenate(([0.0], np.cumsum(pmf)[:-1]))      # P(S >= i)
    disc = (1.0 - theta) ** (i - 1)
    cost = np.array([gamma(age_slots + n + 1) for n in i], dtype=float) / theta + penalty
    num = np.cumsum(disc * pmf * cost)
    den = np.cumsum(disc * surv)
    return float(np.max(num / den))


class GenCMuH(Policy):
    """Generalised c-mu/h with abandonment: serve the top-B by E_k[gamma*mu/(1+h)]."""
    name = "gen_cmu_h"

    def __init__(self, eps=0.1):
        self.eps = eps

    def select(self, state):
        mu = 1.0 / state.latency.mean(min(state.latency.batch_sizes))
        spec = state.spec

        def index(job):
            return sum(p * holding_cost(spec, k, job.age) * mu / (1.0 + abandon_hazard(spec, k, job.age, self.eps))
                       for k, p in enumerate(job.header.tier_probs))
        return Decision(batch=top_batch(rank_by_index(state, index), state, batch_size(state)))


class WhittleAbandon(Policy):
    """Discrete Whittle index with geometric abandonment (see module docstring)."""
    name = "whittle_abandon"

    def __init__(self, slot_s=1.0):
        self.slot_s = slot_s

    def select(self, state):
        spec, slot = state.spec, self.slot_s
        b1 = min(state.latency.batch_sizes)
        pmf = service_pmf(state.latency.samples[b1], slot)

        def index(job):
            total = 0.0
            for k, p in enumerate(job.header.tier_probs):
                if p <= 0:
                    continue
                theta = min(0.99, slot / spec.hard_deadline(k))
                gamma = lambda n, k=k: holding_cost(spec, k, n * slot)
                total += p * whittle_index(pmf, job.age / slot, gamma, theta, spec.weights[k])
            return total
        return Decision(batch=top_batch(rank_by_index(state, index), state, batch_size(state)))


POLICIES = {"gen_cmu_h": GenCMuH, "whittle_abandon": WhittleAbandon}
