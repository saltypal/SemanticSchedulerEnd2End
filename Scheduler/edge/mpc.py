"""Model-predictive / rollout planner (plan sec. 8, "Planning" row): strong but slow baseline.

At each decision it enumerates up to ``K`` candidate first batches (top-b of the EDF and of the
expected-value orderings, several batch sizes), simulates ``H`` further decode decisions in a
lightweight forward model built only from ``State`` (header tier posteriors, latency means,
``dup_prob``, ``u_hat``, spec weights/``g``), rolls out an EDF default policy, and returns the
candidate with the highest expected return. Arrivals enter the model only through the
``arrival_rate`` expectation (mean-field virtual jobs with the queue-average tier posterior).

Complexity per decision: O(K * H * M log M) with M = ``max_queue_considered`` (queue truncated to
the best M/2 by EDF plus best M/2 by expected value). Deterministic: no randomness is used, ties
break to the earlier candidate. Reuses ``contracts`` only.
"""
from Evaluation.edge.contracts import Decision, Policy, State

_TIERS = 4


def _zeta(spec, k, dup):
    return (1.0 - dup * (1.0 - spec.zeta_critical_copy)) if k == 0 else (1.0 - dup)


class _Item:
    """Mutable model job: tier posterior, age at model time 0, dup/u estimates."""
    __slots__ = ("id", "p", "age", "dup", "u", "arr")

    def __init__(self, id_, p, age, dup, u, arr):
        self.id, self.p, self.age, self.dup, self.u, self.arr = id_, p, age, dup, u, arr


def _value(spec, item, delay):
    s = sum(item.p[k] * spec.weights[k] * spec.g(k, delay) * _zeta(spec, k, item.dup)
            for k in range(_TIERS))
    return item.u * s


def _edf_key(spec, item, dt):
    slack = sum(item.p[k] * (spec.soft_deadline[k] - item.age - dt) for k in range(_TIERS))
    return (slack, item.arr, item.id)


class MPCRollout(Policy):
    name = "mpc_rollout"
    uses_truth = False

    def __init__(self, K=6, horizon=4, max_queue_considered=24):
        if K < 1 or horizon < 0 or max_queue_considered < 1:
            raise ValueError("K and max_queue_considered must be >= 1, horizon >= 0")
        self.K, self.horizon, self.max_queue = int(K), int(horizon), int(max_queue_considered)

    # ---------------- candidates ----------------
    def _items(self, state):
        items = [_Item(j.id, j.header.tier_probs, j.age, j.dup_prob, j.u_hat, j.header.arrival_time)
                 for j in state.queue]
        if len(items) > self.max_queue:
            spec, lat = self.spec, state.latency
            m1 = lat.mean(lat.batch_sizes[0])
            half = max(1, self.max_queue // 2)
            by_edf = sorted(items, key=lambda i: _edf_key(spec, i, 0.0))[:half]
            by_val = sorted(items, key=lambda i: (-_value(spec, i, i.age + m1), i.arr, i.id))
            keep = {i.id: i for i in by_edf}
            for i in by_val:
                if len(keep) >= self.max_queue:
                    break
                keep.setdefault(i.id, i)
            items = list(keep.values())
        return items

    @staticmethod
    def _sizes(state, n):
        cap = min(state.max_batch, n)
        return sorted([b for b in state.latency.batch_sizes if b <= cap], reverse=True) or \
            [state.latency.batch_sizes[0]]

    def _candidates(self, state, items):
        spec, lat = self.spec, state.latency
        m1 = lat.mean(lat.batch_sizes[0])
        orders = (sorted(items, key=lambda i: _edf_key(spec, i, 0.0)),
                  sorted(items, key=lambda i: (-_value(spec, i, i.age + m1), i.arr, i.id)))
        cands, seen = [], set()
        for b in self._sizes(state, len(items)):
            for order in orders:
                ids = tuple(i.id for i in order[:b])
                if ids not in seen:
                    seen.add(ids)
                    cands.append(ids)
        return cands[:self.K]

    # ---------------- forward model ----------------
    def _rollout(self, state, items, first_ids):
        spec, lat, c = self.spec, state.latency, max(1, state.workers)
        h_max = max(spec.hard_deadline(k) for k in range(_TIERS))
        m1 = lat.mean(lat.batch_sizes[0])
        free = [0.0] * state.free_workers + [0.5 * m1] * (c - state.free_workers)
        n = max(len(items), 1)
        mean_p = tuple(sum(i.p[k] for i in items) / n for k in range(_TIERS))
        pool = {i.id: i for i in items}
        total, carry, last_t = 0.0, 0.0, 0.0
        vid = -1
        for step in range(self.horizon + 1):
            w = min(range(len(free)), key=lambda x: free[x])
            t = free[w]
            carry += state.arrival_rate * max(0.0, t - last_t)
            last_t = max(last_t, t)
            while carry >= 1.0 and len(pool) < 4 * self.max_queue:
                pool[vid] = _Item(vid, mean_p, -t, 0.0, 1.0, t)
                vid -= 1
                carry -= 1.0
            for jid in [k for k, i in pool.items() if i.age + t > h_max]:
                del pool[jid]
            if step == 0:
                ids = list(first_ids)
            else:
                if not pool:
                    break
                order = sorted(pool.values(), key=lambda i: _edf_key(spec, i, t))
                ids = [i.id for i in order[:self._sizes(state, len(order))[0]]]
            b = len(ids)
            fin = t + lat.mean(b if b in lat.samples else lat.batch_sizes[0])
            for jid in ids:
                item = pool.pop(jid)
                total += _value(spec, item, item.age + fin)
            free[w] = fin
        return total

    def select(self, state: State) -> Decision:
        items = self._items(state)
        if not items:
            return Decision()
        cands = self._candidates(state, items)
        best_ids, best_val = cands[0], None
        for ids in cands:
            v = self._rollout(state, items, ids)
            if best_val is None or v > best_val + 1e-12:
                best_ids, best_val = ids, v
        return Decision(batch=best_ids)


POLICIES = {"mpc_rollout": MPCRollout}
