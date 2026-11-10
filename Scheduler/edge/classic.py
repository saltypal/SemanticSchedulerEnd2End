"""Classic scheduling policies (plan v3.1 sec. 8).

Every policy here only orders the queue; none purges and none reads ``Truth`` (uses_truth=False).
Shared pieces: predicted tier = argmax(header.tier_probs), predicted soft deadline
D_pred = spec.soft_deadline[tier], hard deadline H_pred = spec.hard_deadline(tier). The batch is
``top_batch(order, state, batch_size(state))`` with the size clipped to the queue length.
Reuses: Evaluation.edge.contracts (Policy, Decision, State, top_batch).
"""
import math

from Evaluation.edge.contracts import Decision, Policy, top_batch


def pred_tier(job):
    """argmax of the calibrated tier posterior."""
    p = job.header.tier_probs
    return max(range(len(p)), key=lambda k: p[k])


def batch_size(state):
    """Largest profiled batch size <= min(max_batch, queue length); falls back to the queue length."""
    cap = min(state.max_batch, len(state.queue))
    ok = [b for b in state.latency.batch_sizes if b <= cap]
    return max(ok) if ok else max(cap, 1)


def mean_service(state):
    """Mean decode time of the batch size that would be used."""
    b = batch_size(state)
    return state.latency.mean(b if b in state.latency.samples else state.latency.batch_sizes[0])


def d_pred(job, state):
    return state.spec.soft_deadline[pred_tier(job)]


def w_pred(job, state):
    return state.spec.weights[pred_tier(job)]


def fifo_key(job):
    return (job.header.arrival_time, job.id)


def by_score(state, score):
    """Job ids sorted by descending score, ties broken FIFO."""
    jobs = sorted(state.queue, key=fifo_key)
    return [j.id for j in sorted(jobs, key=lambda j: -score(j))]


def by_key(state, key):
    """Job ids sorted by ascending key, ties broken FIFO."""
    jobs = sorted(state.queue, key=fifo_key)
    return [j.id for j in sorted(jobs, key=key)]


class OrderPolicy(Policy):
    """Base: subclasses implement ``order(state) -> list of ids``; decision is the top batch."""
    uses_truth = False

    def order(self, state):
        raise NotImplementedError

    def select(self, state):
        if not state.queue:
            return Decision()
        return Decision(batch=top_batch(self.order(state), state, batch_size(state)))


class Fifo(OrderPolicy):
    """Order by earliest arrival_time."""
    name = "fifo"

    def order(self, state):
        return by_key(state, fifo_key)


class RandomPolicy(OrderPolicy):
    """Uniform random permutation drawn from the policy rng."""
    name = "random"

    def order(self, state):
        ids = [j.id for j in sorted(state.queue, key=fifo_key)]
        return [ids[i] for i in self.rng.permutation(len(ids))]


class RoundRobinMission(OrderPolicy):
    """Cycle over mission ids (ascending) starting after the last served mission; FIFO within a mission."""
    name = "rr_mission"
    last = None

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.last = None

    def order(self, state):
        groups = {}
        for j in sorted(state.queue, key=fifo_key):
            groups.setdefault(j.header.mission, []).append(j.id)
        ms = sorted(groups)
        if self.last is not None:
            ms = [m for m in ms if m > self.last] + [m for m in ms if m <= self.last]
        out, i = [], 0
        while any(groups.values()):
            m = ms[i % len(ms)]
            if groups[m]:
                out.append(groups[m].pop(0))
            i += 1
        return out

    def select(self, state):
        dec = super().select(state)
        if dec.batch:
            by_id = {j.id: j for j in state.queue}
            self.last = by_id[dec.batch[-1]].header.mission
        return dec


class Edf(OrderPolicy):
    """Earliest deadline first: ascending (D_pred - age)."""
    name = "edf"

    def order(self, state):
        return by_key(state, lambda j: d_pred(j, state) - j.age)


class TufDasa(OrderPolicy):
    """Locke best-effort TUF/DASA: descending E[w_k g_k(age+s)] / s, E over tier_probs, s = mean service."""
    name = "tuf_dasa"

    def order(self, state):
        s = mean_service(state)
        sp = state.spec

        def density(j):
            ev = sum(p * sp.weights[k] * sp.g(k, j.age + s) for k, p in enumerate(j.header.tier_probs))
            return ev / s
        return by_score(state, density)


class SpiFirstEdf(OrderPolicy):
    """Predicted tier ascending (Critical first), EDF (D_pred - age) within tier."""
    name = "spi_first_edf"

    def order(self, state):
        return by_key(state, lambda j: (pred_tier(j), d_pred(j, state) - j.age))


class PriorityAging(OrderPolicy):
    """score = w[pred] + alpha * age / D_pred, descending."""
    name = "priority_aging"

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def order(self, state):
        return by_score(state, lambda j: w_pred(j, state) + self.alpha * j.age / d_pred(j, state))


class Mlwdf(OrderPolicy):
    """M-LWDF: score = (-log(delta) / D_pred) * age * w[pred], descending."""
    name = "mlwdf"

    def __init__(self, delta=0.05):
        self.delta = delta

    def order(self, state):
        c = -math.log(self.delta)
        return by_score(state, lambda j: c / d_pred(j, state) * j.age * w_pred(j, state))


class Llf(OrderPolicy):
    """Least laxity first: ascending D_pred - age - mean_service."""
    name = "llf"

    def order(self, state):
        s = mean_service(state)
        return by_key(state, lambda j: d_pred(j, state) - j.age - s)


class Lelf(OrderPolicy):
    """Lowest effective lifetime first: ascending H_pred - age."""
    name = "lelf"

    def order(self, state):
        return by_key(state, lambda j: state.spec.hard_deadline(pred_tier(j)) - j.age)


class ExpRule(OrderPolicy):
    """EXP rule: w * exp((a_j - abar) / (1 + sqrt(abar))), a_j = age / D_pred, abar = mean a over queue."""
    name = "exp_rule"

    def order(self, state):
        a = {j.id: j.age / d_pred(j, state) for j in state.queue}
        abar = sum(a.values()) / len(a)
        den = 1.0 + math.sqrt(abar)
        return by_score(state, lambda j: w_pred(j, state) * math.exp((a[j.id] - abar) / den))


class Dapf(OrderPolicy):
    """Deadline-aware priority: w[pred] * (1 + age / D_pred), descending."""
    name = "dapf"

    def order(self, state):
        return by_score(state, lambda j: w_pred(j, state) * (1.0 + j.age / d_pred(j, state)))


class PfMission(OrderPolicy):
    """Proportional fair across missions: w[pred] / (avg served value of the mission + eps), descending."""
    name = "pf_mission"

    def __init__(self, eps=1e-3):
        self.eps = eps

    def order(self, state):
        served = dict(state.mission_served)
        return by_score(state, lambda j: w_pred(j, state) / (served.get(j.header.mission, 0.0) + self.eps))


class GenCmu(OrderPolicy):
    """Generalised c-mu: w[pred] * mu * gamma(age), mu = 1 / mean_service, gamma = 1 + age / D_pred."""
    name = "gen_cmu"

    def order(self, state):
        mu = 1.0 / mean_service(state)
        return by_score(state, lambda j: w_pred(j, state) * mu * (1.0 + j.age / d_pred(j, state)))


POLICIES = {
    "fifo": Fifo, "random": RandomPolicy, "rr_mission": RoundRobinMission, "edf": Edf,
    "tuf_dasa": TufDasa, "spi_first_edf": SpiFirstEdf, "priority_aging": PriorityAging,
    "mlwdf": Mlwdf, "llf": Llf, "lelf": Lelf, "exp_rule": ExpRule, "dapf": Dapf,
    "pf_mission": PfMission, "gen_cmu": GenCmu,
}
