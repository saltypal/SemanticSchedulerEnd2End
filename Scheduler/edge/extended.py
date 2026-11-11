"""Extended header-only scheduling policies (checklist P0, items 1-8, 12, 13).

Every policy here sees only ``State`` / ``QueuedJob`` (header, age, queue, latency profile); none reads
``Truth`` or the fidelity estimate ``u_hat`` (fidelity is dropped from the project). Predicted tier =
argmax(tier_probs); expected value of a job = sum_k p_k w_k g_k(delay) (``expected_value``). Each class
docstring says how the published algorithm was adapted to one decoder queue with batching: the batch is
the top ``batch_size(state)`` ids of the policy's order (non-preemptive at batch granularity).
Batching policies (Clipper, Clockwork, Shepherd/Nexus) live in ``batching.py``.
Reuses: Scheduler.edge.classic (OrderPolicy, helpers), Evaluation.edge.contracts.
"""
import math

import numpy as np

from Evaluation.edge.contracts import Decision, Policy, top_batch
from Scheduler.edge.classic import (OrderPolicy, batch_size, by_key, by_score, d_pred, fifo_key,
                                    mean_service, pred_tier)


def h_pred(job, state):
    return state.spec.hard_deadline(pred_tier(job))


def expected_value(job, state, delay):
    """E over the SPI posterior of w_k * g_k(delay) (header-only)."""
    sp = state.spec
    return sum(p * sp.weights[k] * sp.g(k, delay) for k, p in enumerate(job.header.tier_probs))


class DOver(Policy):
    """D-over (Koren & Shasha 1995). Value density order; overload handling at the latest start time.

    Adapted: value = expected_value at completion; service time is the batch mean latency (same for all
    jobs, so density order = value order). A job with age + s > H_pred is hopeless and purged. A job at its
    latest start time (H_pred - age - s < s) is admitted to the front only if its value >= (1 + sqrt(kappa))
    x the value of the non-urgent job it would displace (the b-th ranked; 0 if a slot is free), else it is
    abandoned (purged). kappa = max(w) / min(w).
    """
    name = "d_over"

    def __init__(self, kappa=None):
        self.kappa = kappa

    def select(self, state):
        if not state.queue:
            return Decision()
        s, b = mean_service(state), batch_size(state)
        kappa = self.kappa or max(state.spec.weights) / min(state.spec.weights)
        factor = 1.0 + math.sqrt(kappa)
        val = {j.id: expected_value(j, state, j.age + s) for j in state.queue}
        purge, normal, urgent = [], [], []
        for j in sorted(state.queue, key=fifo_key):
            slack = h_pred(j, state) - j.age - s
            if slack < 0:
                purge.append(j.id)
            elif slack < s:
                urgent.append(j)
            else:
                normal.append(j)
        normal.sort(key=lambda j: -val[j.id])
        ref = val[normal[b - 1].id] if len(normal) >= b else 0.0
        admitted = []
        for j in sorted(urgent, key=lambda j: -val[j.id]):
            if val[j.id] >= factor * ref:
                admitted.append(j.id)
            else:
                purge.append(j.id)
        order = admitted + [j.id for j in normal]
        return Decision(batch=top_batch(order, state, b) if order else (), purge=tuple(purge))


class Hvdf(OrderPolicy):
    """Highest (expected) value density first: expected_value(age + s_j) / s_j.

    Adapted: s_j = mean batch service time scaled by payload_bytes / mean payload of the queue (decode cost
    assumed proportional to payload size; equal payloads reduce to plain value-density order).
    """
    name = "hvdf"

    def order(self, state):
        s = mean_service(state)
        mp = sum(j.header.payload_bytes for j in state.queue) / len(state.queue) or 1.0
        sj = {j.id: s * max(j.header.payload_bytes, 1) / mp for j in state.queue}
        return by_score(state, lambda j: expected_value(j, state, j.age + sj[j.id]) / sj[j.id])


class EdfTriage(Policy):
    """EDF with admission/triage: purge jobs that cannot meet H_pred, EDF (D_pred - age) on the rest.

    Adapted: a job is infeasible when age + the ``q``-quantile of the fastest profiled batch latency
    exceeds H_pred (optimistic, so only provably late jobs are dropped).
    """
    name = "edf_triage"

    def __init__(self, q=0.5):
        self.q = q

    def select(self, state):
        if not state.queue:
            return Decision()
        lat = state.latency.quantile(state.latency.batch_sizes[0], self.q)
        purge = tuple(j.id for j in sorted(state.queue, key=fifo_key) if j.age + lat > h_pred(j, state))
        dead = set(purge)
        live = [j for j in state.queue if j.id not in dead]
        order = [j.id for j in sorted(sorted(live, key=fifo_key), key=lambda j: d_pred(j, state) - j.age)]
        return Decision(batch=top_batch(order, state, batch_size(state)) if order else (), purge=purge)


class LcfsPriority(OrderPolicy):
    """LCFS with priority: predicted tier ascending, newest arrival first within a tier.

    Adapted: preemption is only possible at batch boundaries (non-preemptive at batch granularity).
    """
    name = "lcfs_priority"

    def order(self, state):
        return by_key(state, lambda j: (pred_tier(j), -j.header.arrival_time, -j.id))


class DrrTiers(Policy):
    """Deficit Round Robin over predicted-tier FIFO queues; quantum_k = w_k / min(w) jobs per round.

    Adapted: unit job cost; a batch is filled by continuing the round-robin visit across calls (the
    pointer, deficits and whether the current visit was already credited persist between selects).
    """
    name = "drr_tiers"

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.deficit, self.ptr, self.credited = [0.0] * len(spec.weights), 0, False

    def select(self, state):
        if not state.queue:
            return Decision()
        K, w = len(state.spec.weights), state.spec.weights
        quantum = [x / min(w) for x in w]
        q = [[] for _ in range(K)]
        for j in sorted(state.queue, key=fifo_key):
            q[pred_tier(j)].append(j.id)
        b, out = batch_size(state), []
        while len(out) < b and any(q):
            k = self.ptr
            if not q[k]:
                self.deficit[k], self.ptr, self.credited = 0.0, (k + 1) % K, False
                continue
            if not self.credited:
                self.deficit[k] += quantum[k]
                self.credited = True
            while self.deficit[k] >= 1.0 and q[k] and len(out) < b:
                out.append(q[k].pop(0))
                self.deficit[k] -= 1.0
            if len(out) >= b and q[k] and self.deficit[k] >= 1.0:
                break                                   # visit continues on the next select
            if not q[k]:
                self.deficit[k] = 0.0
            self.ptr, self.credited = (k + 1) % K, False
        return Decision(tuple(out))


class MaxWeight(Policy):
    """Max-Weight / backpressure over predicted-tier queues: repeatedly serve the tier maximising
    (remaining queue length) x w_k, taking its oldest job, until the batch is full."""
    name = "max_weight"

    def select(self, state):
        if not state.queue:
            return Decision()
        K, w = len(state.spec.weights), state.spec.weights
        q = [[] for _ in range(K)]
        for j in sorted(state.queue, key=fifo_key):
            q[pred_tier(j)].append(j.id)
        out = []
        for _ in range(batch_size(state)):
            k = max((k for k in range(K) if q[k]), key=lambda k: (len(q[k]) * w[k], -k))
            out.append(q[k].pop(0))
        return Decision(tuple(out))


class MaxAge(OrderPolicy):
    """Max-age-first (AoI): descending age = now - capture_time (ties FIFO)."""
    name = "max_age"

    def order(self, state):
        return by_score(state, lambda j: j.age)


class WhittleAoi(OrderPolicy):
    """Whittle-type index for Age of Information, per tier, expected over the posterior.

    Adapted (Kadota-style, reliable channel with service time s): resetting age a to s during one service
    saves ((a + s)^2 - s^2) / (2 s) age-seconds; here a, s are measured in units of D_k and scaled by w_k:
    index = sum_k p_k w_k a_k (a_k + 2 s_k) / (2 s_k), a_k = age / D_k, s_k = s / D_k, s = batch mean latency.
    """
    name = "whittle_aoi"

    def order(self, state):
        s, sp = mean_service(state), state.spec

        def index(j):
            tot = 0.0
            for k, p in enumerate(j.header.tier_probs):
                a, sk = j.age / sp.soft_deadline[k], s / sp.soft_deadline[k]
                tot += p * sp.weights[k] * a * (a + 2.0 * sk) / (2.0 * sk)
            return tot
        return by_score(state, index)


def cvar_lower(probs, weights, alpha):
    """Mean of the worst ``alpha`` probability mass of the discrete distribution (weights ~ probs)."""
    alpha = min(alpha, sum(probs))
    left, acc = alpha, 0.0
    for k in sorted(range(len(weights)), key=lambda k: weights[k]):
        m = min(probs[k], left)
        acc += m * weights[k]
        left -= m
        if left <= 1e-12:
            break
    return acc / alpha if alpha > 0 else 0.0


class CvarSpi(OrderPolicy):
    """Risk-averse SPI priority: lower-tail CVaR_alpha of the tier weight under the posterior (a
    conservative value, so uncertain headers rank below confident ones with the same mean), times
    (1 + age / D_pred) so waiting jobs rise. alpha = 1 gives the plain expected weight."""
    name = "cvar_spi"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def order(self, state):
        w = state.spec.weights
        return by_score(state, lambda j: cvar_lower(j.header.tier_probs, w, self.alpha)
                        * (1.0 + j.age / d_pred(j, state)))


class LinUcb(Policy):
    """Contextual-bandit priority learned online inside one run (no offline training, seeded).

    Linear reward model on header features x = [tier_probs(4), relevance, min(age / H_pred, 2),
    payload_kB, 1]. Score = theta.x + alpha sqrt(x^T A^-1 x) (LinUCB), or with ``thompson`` a draw
    theta ~ N(theta_hat, alpha^2 A^-1) from the policy rng. Adapted: one shared linear model over jobs; the
    observed reward of a decoded job is its header-only realised expected value sum_k p_k w_k g_k(delay)
    (true tier/fidelity are not observable); the update happens on the 'complete' event.
    """
    name = "linucb"
    DIM = 8

    def __init__(self, alpha=0.5, ridge=1.0, thompson=False):
        self.alpha, self.ridge, self.thompson = alpha, ridge, thompson

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.A, self.b, self.pending = self.ridge * np.eye(self.DIM), np.zeros(self.DIM), {}

    def features(self, job, state):
        h = job.header
        return np.array([*h.tier_probs, h.relevance, min(job.age / h_pred(job, state), 2.0),
                         h.payload_bytes / 1000.0, 1.0])

    def select(self, state):
        if not state.queue:
            return Decision()
        ainv = np.linalg.inv(self.A)
        theta = ainv @ self.b
        if self.thompson:
            theta = self.rng.multivariate_normal(theta, self.alpha ** 2 * (ainv + ainv.T) / 2.0)
        jobs = sorted(state.queue, key=fifo_key)
        X = np.array([self.features(j, state) for j in jobs])
        score = X @ theta
        if not self.thompson:
            score = score + self.alpha * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, ainv, X), 0.0))
        order = np.argsort(-score, kind="stable")
        batch = top_batch([jobs[i].id for i in order], state, batch_size(state))
        pos = {j.id: i for i, j in enumerate(jobs)}
        for i in batch:
            self.pending[i] = X[pos[i]]
        return Decision(batch)

    def observe(self, event, payload):
        if event != "complete":
            return
        sp = self.spec
        for i, probs, delay in zip(payload["ids"], payload["tier_probs"], payload["delays"]):
            x = self.pending.pop(i, None)
            if x is None:
                continue
            r = sum(p * sp.weights[k] * sp.g(k, delay) for k, p in enumerate(probs))
            self.A += np.outer(x, x)
            self.b += r * x


POLICIES = {
    "d_over": DOver, "hvdf": Hvdf, "edf_triage": EdfTriage, "lcfs_priority": LcfsPriority,
    "drr_tiers": DrrTiers, "max_weight": MaxWeight, "max_age": MaxAge, "whittle_aoi": WhittleAoi,
    "cvar_spi": CvarSpi, "linucb": LinUcb,
}
