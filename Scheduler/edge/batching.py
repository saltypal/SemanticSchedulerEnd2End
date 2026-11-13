"""Batching-aware header-only policies (checklist P0, items 9-11): Clipper AIMD, Clockwork, Shepherd/Nexus.

Same conventions as ``extended.py``: predicted tier = argmax(tier_probs), no Truth / u_hat. The batch size
is chosen from the profiled sizes <= min(max_batch, queue length) using ``latency.quantile``.
Reuses: Scheduler.edge.classic, Scheduler.edge.extended (expected_value, h_pred).
"""
import numpy as np

from Evaluation.edge.contracts import Decision, Policy, top_batch
from Scheduler.edge.classic import batch_size, d_pred, fifo_key, pred_tier
from Scheduler.edge.extended import expected_value, h_pred


def sizes(state, n):
    """Profiled batch sizes usable with ``n`` candidate jobs, ascending."""
    return [b for b in state.latency.batch_sizes if b <= min(state.max_batch, n)]


class ClipperAimd(Policy):
    """Clipper-style AIMD adaptive batching (Crankshaw et al. 2017) with SPI-first-EDF priority order.

    Adapted: the batch-size cap starts at 1, grows by ``add`` after a batch whose measured service time
    (completion time - start time seen in ``select``) is within the SLO, and is multiplied by ``beta`` after
    a violation. SLO = ``slo_frac`` x the smallest D_pred among the batch's jobs (from their tier_probs).
    The decoded size is the largest profiled size <= min(cap, max_batch, queue length).
    """
    name = "clipper_aimd"

    def __init__(self, add=1.0, beta=0.5, slo_frac=0.25):
        self.add, self.beta, self.slo_frac = add, beta, slo_frac

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.cap, self.start = 1.0, {}

    def select(self, state):
        if not state.queue:
            return Decision()
        ok = sizes(state, len(state.queue))
        cap = int(self.cap)
        use = [b for b in ok if b <= cap]
        b = max(use) if use else (min(ok) if ok else batch_size(state))
        order = [j.id for j in sorted(sorted(state.queue, key=fifo_key),
                                      key=lambda j: (pred_tier(j), d_pred(j, state) - j.age))]
        batch = top_batch(order, state, b)
        for i in batch:
            self.start[i] = state.now
        return Decision(batch)

    def observe(self, event, payload):
        if event != "complete":
            return
        ids = payload["ids"]
        t0 = min(self.start.pop(i, payload["time"]) for i in ids)
        slo = self.slo_frac * min(self.spec.soft_deadline[int(np.argmax(p))] for p in payload["tier_probs"])
        if payload["time"] - t0 > slo:
            self.cap = max(1.0, self.cap * self.beta)
        else:
            self.cap += self.add


class Clockwork(Policy):
    """Clockwork-style deadline-aware batch planner (Gujarati et al. 2020).

    Adapted: latency is the ``q``-quantile (default p95) of the profile. Jobs that cannot meet H_pred even in
    the fastest batch are rejected (purged). Remaining jobs are ordered by earliest H_pred - age; for each
    feasible size b (earliest-deadline job still completes by its H_pred) the plan value is
    sum of expected_value(age + c_b) over the first b jobs / c_b; the best plan wins (ties: larger b). If no
    size is feasible the smallest size is used.
    """
    name = "clockwork"

    def __init__(self, q=0.95):
        self.q = q

    def select(self, state):
        if not state.queue:
            return Decision()
        lat = state.latency
        c1 = lat.quantile(lat.batch_sizes[0], self.q)
        purge = tuple(j.id for j in sorted(state.queue, key=fifo_key) if j.age + c1 > h_pred(j, state))
        dead = set(purge)
        live = sorted(sorted((j for j in state.queue if j.id not in dead), key=fifo_key),
                      key=lambda j: h_pred(j, state) - j.age)
        if not live:
            return Decision(purge=purge)
        best, best_score = None, -1.0
        for b in sizes(state, len(live)):
            c = lat.quantile(b, self.q)
            if live[0].age + c > h_pred(live[0], state):
                continue
            score = sum(expected_value(j, state, j.age + c) for j in live[:b]) / c
            if score >= best_score:
                best, best_score = b, score
        if best is None:
            ok = sizes(state, len(live))
            best = min(ok) if ok else batch_size(state)
        return Decision(batch=tuple(j.id for j in live[:best]), purge=purge)


class ShepherdNexus(Policy):
    """Shepherd / Nexus-style SLO batch packing (Zhang et al. 2023; Shen et al. 2019).

    Adapted: the SLO of a job is its D_pred. For each profiled size b (descending) the eligible jobs are those
    with D_pred - age >= c_b (c_b = ``q``-quantile latency of size b); the largest b with at least b eligible
    jobs wins and the batch is the first b eligible jobs by priority (predicted tier, then least slack).
    If no job tolerates even the smallest batch, the highest-priority job is served alone (best effort).
    """
    name = "shepherd_nexus"

    def __init__(self, q=0.95):
        self.q = q

    def select(self, state):
        if not state.queue:
            return Decision()
        prio = sorted(sorted(state.queue, key=fifo_key),
                      key=lambda j: (pred_tier(j), d_pred(j, state) - j.age))
        for b in reversed(sizes(state, len(prio))):
            c = state.latency.quantile(b, self.q)
            elig = [j.id for j in prio if d_pred(j, state) - j.age >= c]
            if len(elig) >= b:
                return Decision(tuple(elig[:b]))
        ok = sizes(state, len(prio))
        b = min(ok) if ok else batch_size(state)
        return Decision(top_batch([j.id for j in prio], state, b))


UNIFORM4 = (0.25, 0.25, 0.25, 0.25)


def dead_mask(state):
    """F3: True for jobs that cannot finish before the hard deadline of any likely tier (p > 0.05) even as a
    batch of one - the same test SV-DPP uses to purge. Learners rank these last (queueing analogue of the
    work-conserving parametrisation of Che et al., arXiv:2409.03740)."""
    sp, lat = state.spec, state.latency
    b1 = lat.batch_sizes[0]
    out = []
    for j in state.queue:
        p = j.header.tier_probs
        live = [k for k in range(4) if p[k] > 0.05] or [int(np.argmax(p))]
        out.append(all(lat.cdf(b1, sp.hard_deadline(k) - j.age) == 0.0 for k in live))
    return np.asarray(out, dtype=bool)


def rank_alive_first(scores, state):
    """Stable descending order of ``scores`` with dead jobs (``dead_mask``) moved to the end."""
    s = np.asarray(scores, dtype=float).copy()
    s[dead_mask(state)] = -1e18
    return np.argsort(-s, kind="stable")


def ratio_batch_size(state, order, use_header=True):
    """Shared batch-size rule (fix plan F2): given a ranking ``order`` (indices into ``state.queue``), pick the
    profiled B maximising expected on-time value per second of decoding,
    sum_{i in top-B} sum_k p_ik w_k P(S_B <= H_k - age_i) / E[S_B]  (renewal-ratio rule, Neely 2013).
    ``use_header=False`` uses the uniform tier prior (no semantics) for the batching-only control."""
    ok = sizes(state, len(order))
    if not ok:
        return batch_size(state)
    if len(ok) == 1:
        return ok[0]
    sp, lat = state.spec, state.latency
    best = None
    for b in ok:
        tot = 0.0
        for i in order[:b]:
            j = state.queue[int(i)]
            p = j.header.tier_probs if use_header else UNIFORM4
            tot += sum(p[k] * sp.weights[k] * lat.cdf(b, sp.hard_deadline(k) - j.age) for k in range(4))
        r = tot / lat.mean(b)
        if best is None or r > best[0] + 1e-12:
            best = (r, b)
    return best[1]


class RatioBatched(Policy):
    """Control policies for the semantics-vs-batching question: an order rule plus ``ratio_batch_size``.
    ``edf_ratio``: EDF on the uniform-prior deadline (no header) + header-free batch rule (batching only).
    ``spi_edf_ratio``: SPI-first-EDF order + header batch rule (semantics through the order and the rule)."""
    def __init__(self, use_header):
        self.use_header = use_header

    def select(self, state):
        if not state.queue:
            return Decision()
        q = state.queue
        if self.use_header:
            key = [(-(j.header.tier_probs[0]), state.spec.soft_deadline[pred_tier(j)] - j.age, j.header.arrival_time)
                   for j in q]
        else:
            dbar = float(np.mean(state.spec.soft_deadline))
            key = [(0.0, dbar - j.age, j.header.arrival_time) for j in q]
        order = sorted(range(len(q)), key=lambda i: key[i])
        b = ratio_batch_size(state, order, self.use_header)
        return Decision(batch=tuple(q[i].id for i in order[:b]))


class EdfRatio(RatioBatched):
    name = "edf_ratio"

    def __init__(self):
        super().__init__(False)


class SpiEdfRatio(RatioBatched):
    name = "spi_edf_ratio"

    def __init__(self):
        super().__init__(True)


POLICIES = {"clipper_aimd": ClipperAimd, "clockwork": Clockwork, "shepherd_nexus": ShepherdNexus,
            "edf_ratio": EdfRatio, "spi_edf_ratio": SpiEdfRatio}
