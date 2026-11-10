"""EdgeServing stability-score policy and the completion-aware drift-plus-penalty (DPP) policies.

Reuses ``Evaluation.edge.contracts`` and ``config/edge/sim.yaml`` (sv_dpp.V). No policy reads truth;
the predicted tier is argmax(tier_probs) wherever a single tier is needed.

* edgeserving (after arXiv 2605.05527): f(w) = min(exp(w / tau - 1), C) with tau = hard deadline of the
  predicted tier and C = exp(2.3). Candidate batches are the top-B by EDF for each profiled B <= max_batch;
  remaining jobs age by p95(B), the score is the sum of f(age) over remaining jobs, argmin wins, ties go to
  larger B. No early exit.
* completion_dpp (after arXiv 2608.05537) and tier_dpp_no_gain: per-tier debt Z_k <- max(0, Z_k + eta_k*A_k
  - C_k), updated from ``observe``. completion_dpp scores a job by sum_k Z_k p_k (P_now_k - P_defer_k) +
  V*E[w g]; tier_dpp_no_gain replaces the gain with 1[on time if decoded now]. P_now =
  Phi((H_k - age - mean_service) / sigma), sigma = std(latency samples) + 1e-6, P_defer uses
  age + mean_service * rank / c (rank = EDF position, c = batch size). Jobs whose P_now is below
  ``p_floor`` (1e-9, our float stand-in for "== 0") for every tier with p_k > 0.05 are purged.

observe payloads (exactly as sent by the simulator):
  event="arrival":  {"id": int, "tier_probs": tuple}     -> A_k += tier_probs[k] (debt += eta_k * p_k)
  event="complete": {"ids": list, "tier_probs": list of tuples, "delays": list of floats}
                    -> C_k += p_k for jobs with delay <= H_k (on time = within hard deadline)
  event in ("expire", "purge"): {"ids": list}            -> ignored (the debt already counted them)
"""
import math

import numpy as np

from Evaluation.edge.contracts import Decision, Policy
from Evaluation.edge.edge_config import load_config
from Scheduler.edge.theory import predicted_tier

CAP = math.exp(2.3)


def stability(age, tau):
    """f(w) = min(exp(w / tau - 1), C)."""
    return min(math.exp(age / tau - 1.0), CAP)


def candidate_sizes(state):
    """Profiled batch sizes that fit the max batch and the current queue (at least one)."""
    n = len(state.queue)
    sizes = [b for b in state.latency.batch_sizes if b <= state.max_batch and b <= n]
    return sizes or [min(state.latency.batch_sizes)]


def _tau(spec, job):
    return spec.hard_deadline(predicted_tier(job.header.tier_probs))


def _edf(state):
    return sorted(state.queue, key=lambda j: (_tau(state.spec, j) - j.age, j.id))


class EdgeServing(Policy):
    """Pick the batch whose post-service queue has the lowest stability score."""
    name = "edgeserving"

    def select(self, state):
        edf = _edf(state)
        best = None
        for b in candidate_sizes(state):
            delta = state.latency.p95(b)
            score = sum(stability(j.age + delta, _tau(state.spec, j)) for j in edf[b:])
            key = (score, -b)
            if best is None or key < best[0]:
                best = (key, b)
        return Decision(batch=tuple(j.id for j in edf[:best[1]]))


def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


class DebtPolicy(Policy):
    """Shared debt-queue machinery; subclasses define ``job_term``."""
    name = "debt"

    def __init__(self, V=None, p_floor=1e-9):
        self.V = float(load_config("sim")["sv_dpp"]["V"] if V is None else V)
        self.p_floor = p_floor
        self.Z = [0.0] * 4

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.Z = [float(load_config("sim")["sv_dpp"]["debt_init"])] * 4

    def observe(self, event, payload):
        spec = getattr(self, "spec", None)
        if spec is None:
            return
        if event == "arrival":
            for k, p in enumerate(payload["tier_probs"]):
                self.Z[k] = max(0.0, self.Z[k] + spec.eta[k] * p)
        elif event == "complete":
            for probs, delay in zip(payload["tier_probs"], payload["delays"]):
                for k, p in enumerate(probs):
                    if delay <= spec.hard_deadline(k):
                        self.Z[k] = max(0.0, self.Z[k] - p)

    def job_term(self, k, p_now, p_defer, on_time_now):
        raise NotImplementedError

    def select(self, state):
        spec, lat = state.spec, state.latency
        b = max(candidate_sizes(state))
        mean, sigma = lat.mean(b), float(np.std(lat.samples[b])) + 1e-6
        scored, purge = [], []
        for rank, job in enumerate(_edf(state)):
            probs = job.header.tier_probs
            p_now = [_phi((spec.hard_deadline(k) - job.age - mean) / sigma) for k in range(4)]
            if all(p_now[k] < self.p_floor for k in range(4) if probs[k] > 0.05):
                purge.append(job.id)
                continue
            total = 0.0
            for k in range(4):
                p_def = _phi((spec.hard_deadline(k) - job.age - mean * rank / b) / sigma)
                on_time = 1.0 if job.age + mean <= spec.hard_deadline(k) else 0.0
                total += probs[k] * (self.Z[k] * self.job_term(k, p_now[k], p_def, on_time)
                                     + self.V * spec.weights[k] * spec.g(k, job.age + mean))
            scored.append((-total, -job.age, job.id))
        scored.sort()
        fit = [x for x in lat.batch_sizes if x <= min(b, len(scored))]   # keep a profiled size after purges
        return Decision(batch=tuple(s[2] for s in scored[:max(fit)]) if fit else (), purge=tuple(purge))


class CompletionDPP(DebtPolicy):
    name = "completion_dpp"

    def job_term(self, k, p_now, p_defer, on_time_now):
        return p_now - p_defer


class TierDPPNoGain(DebtPolicy):
    name = "tier_dpp_no_gain"

    def job_term(self, k, p_now, p_defer, on_time_now):
        return on_time_now


POLICIES = {"edgeserving": EdgeServing, "completion_dpp": CompletionDPP,
            "tier_dpp_no_gain": TierDPPNoGain}
