"""SV-DPP v3.1 (plan sec. 6): soft-value drift-plus-penalty scheduler with renewal-frame batching.

Reuses Evaluation.edge.contracts (Policy, State, Decision, ValueSpec, LatencyProfile) and
config/edge/sim.yaml (V, debt_init, hard_purge_prob, eta). Only ``GenieSVDPP`` reads ground truth.

Simplifications (documented estimates, the plan leaves the exact form open):
* W_hat_i = (sum of amortised per-job service of jobs ranked above i by E[v|now]) / c
            + lam * slack_i * s_bar * share_higher_i / c
  with amortised service s_bar = min_b mean(b)/b, slack_i = p-weighted (H_k - age_i)+, and
  share_higher_i = sum_k p_ik * (queue-average probability mass of tiers more critical than k).
* sigma_W^2 = (n_ahead * var(S_1) + lam * slack_i * share_higher_i * E[S_1^2]) / c^2.
* G_defer uses a normal approximation of age + W_hat + S_B with variance sigma_W^2 + var(S_B).
* The policy keeps its own debt queues Z (updated through observe), not State.debts.
* D18: E[v_i] (now and defer terms) is multiplied by factor_i = 1 + relevance_beta * (r_tilde_i - 0.5),
  r_tilde_i = clip((header relevance - lo[mission]) / (hi[mission] - lo[mission]), 0, 1). Header only.
  relevance_beta = 0 gives factor 1 exactly, i.e. the plan v3 policy.
"""
import math

import numpy as np

from Evaluation.edge.contracts import Decision, Policy, State
from Evaluation.edge.edge_config import load_config

_TIE = 1e-12


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class SVDPP(Policy):
    name = "sv_dpp"
    uses_truth = False

    def __init__(self, V=None, eta=None, debt_init=None, hard_purge_prob=None,
                 relevance_beta=None, relevance_range=None, tie_tol_frac=None):
        cfg = load_config("sim")
        self.V = float(cfg["sv_dpp"]["V"] if V is None else V)
        self._eta_arg = None if eta is None else tuple(float(e) for e in eta)
        d0 = cfg["sv_dpp"]["debt_init"] if debt_init is None else debt_init
        self._debt_init = tuple(d0) if hasattr(d0, "__len__") else (float(d0),) * 4
        self.hard_purge_prob = float(cfg["dup"]["hard_purge_prob"] if hard_purge_prob is None
                                     else hard_purge_prob)
        self.eta = self._eta_arg or tuple(float(e) for e in cfg["eta_timely"])
        self.relevance_beta = float(cfg["sv_dpp"]["relevance_beta"] if relevance_beta is None
                                    else relevance_beta)
        self.tie_tol_frac = float(cfg["sv_dpp"].get("tie_tol_frac", 0.1) if tie_tol_frac is None
                                  else tie_tol_frac)
        rr = cfg["sv_dpp"]["relevance_range"] if relevance_range is None else relevance_range
        self._relevance_range = {int(m): (float(lo), float(hi)) for m, (lo, hi) in dict(rr).items()}
        if 0 not in self._relevance_range or any(hi <= lo for lo, hi in self._relevance_range.values()):
            raise ValueError("sv_dpp relevance_range needs mission 0 and lo < hi for every mission")
        self.Z = list(self._debt_init)
        self._A, self._T = [0.0] * 4, [0.0] * 4

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.eta = self._eta_arg or tuple(spec.eta)
        self.Z = list(self._debt_init)
        self._A, self._T = [0.0] * 4, [0.0] * 4

    @property
    def debts(self):
        return tuple(self.Z)

    # ---------------- debt queues (sec. 6.1) ----------------
    def _probs(self, job_id, probs):
        return probs

    def _after_complete(self, ids):
        pass

    def observe(self, event, payload):
        if event == "arrival":
            p = self._probs(payload["id"], payload["tier_probs"])
            for k in range(4):
                self._A[k] += p[k]
        elif event == "complete":
            for jid, p, d in zip(payload["ids"], payload["tier_probs"], payload["delays"]):
                p = self._probs(jid, p)
                for k in range(4):
                    if d <= self.spec.hard_deadline(k):
                        self._T[k] += p[k]
            self._after_complete(payload["ids"])
            for k in range(4):
                self.Z[k] = max(0.0, self.Z[k] + self.eta[k] * self._A[k] - self._T[k])
            self._A, self._T = [0.0] * 4, [0.0] * 4

    # ---------------- per-job view (overridden by the genie) ----------------
    def _view(self, job):
        return job.header.tier_probs, job.u_hat, job.dup_prob

    def _zeta(self, k, dup):
        return (1.0 - dup * (1.0 - self.spec.zeta_critical_copy)) if k == 0 else (1.0 - dup)

    def _relevance_factor(self, job):
        """D18: multiplier on E[v] from the header relevance only. Exactly 1.0 when beta is 0."""
        if self.relevance_beta == 0.0:
            return 1.0
        lo, hi = self._relevance_range.get(int(job.header.mission), self._relevance_range[0])
        r_tilde = min(1.0, max(0.0, (job.header.relevance - lo) / (hi - lo)))
        return 1.0 + self.relevance_beta * (r_tilde - 0.5)

    def _ev(self, view, delay, factor=1.0):
        p, u, dup = view
        s = self.spec
        return factor * (u * sum(p[k] * s.weights[k] * s.g(k, delay) * self._zeta(k, dup) for k in range(4)))

    # ---------------- selection (sec. 6.2-6.5) ----------------
    def select(self, state: State) -> Decision:
        spec, lat, c = self.spec, state.latency, max(1, state.workers)
        views = {j.id: self._view(j) for j in state.queue}
        b1 = lat.batch_sizes[0]
        purge = set()
        for j in state.queue:
            p, _, dup = views[j.id]
            live = [k for k in range(4) if p[k] > 0.05] or [int(np.argmax(p))]
            dead = all(lat.cdf(b1, spec.hard_deadline(k) - j.age) == 0.0 for k in live)
            if dup > self.hard_purge_prob or dead:
                purge.add(j.id)
        jobs = [j for j in state.queue if j.id not in purge]
        if not jobs:
            return Decision(purge=tuple(sorted(purge)))
        sizes = [b for b in lat.batch_sizes if b <= state.max_batch and b <= len(jobs)] or [b1]

        s1 = lat.samples[b1]
        amort = min(lat.mean(b) / b for b in lat.batch_sizes)
        var1, m2 = float(s1.var()), float(np.mean(s1 ** 2))
        fac = {j.id: self._relevance_factor(j) for j in jobs}
        e1 = {j.id: self._ev(views[j.id], j.age + lat.mean(b1), fac[j.id]) for j in jobs}
        order = sorted(jobs, key=lambda j: (-e1[j.id], j.header.arrival_time))
        q_mass = np.mean([views[j.id][0] for j in jobs], axis=0)
        lam = state.arrival_rate
        what = {}
        for r, j in enumerate(order):
            p = views[j.id][0]
            slack = sum(p[k] * max(0.0, spec.hard_deadline(k) - j.age) for k in range(4))
            share = sum(p[k] * float(q_mass[:k].sum()) for k in range(4))
            what[j.id] = ((r * amort + lam * slack * amort * share) / c,
                          (r * var1 + lam * slack * share * m2) / c ** 2)

        best = None
        for B in sizes:
            mB, vB = lat.mean(B), float(lat.samples[B].var())
            rows = []
            for j in jobs:
                view = views[j.id]
                w, var = what[j.id]
                en = self._ev(view, j.age + mB, fac[j.id])
                wd = w + mB / c    # deferral costs at least one service round of the pool (sec. 6.2 / 6.3)
                ed = self._ev(view, j.age + wd + mB, fac[j.id])
                sd = math.sqrt(var + vB)
                dg = 0.0
                for k in range(4):
                    h = spec.hard_deadline(k) - j.age
                    gn = lat.cdf(B, h)
                    gd = _phi((h - wd - mB) / sd) if sd > 0 else float(h - wd - mB >= 0)
                    dg += self.Z[k] * view[0][k] * (gn - gd)
                rows.append((self.V * (en - ed) + dg, en, j))
            # sec. 6.3 ties: a job whose score is negligible next to the value scale is "not at risk"
            # and is ranked by E[v] (then arrival); per job, not only when the whole queue ties.
            tol = max(_TIE, self.tie_tol_frac * max(r[1] for r in rows))
            rows = [((r[0] if r[0] > tol else 0.0), r[1], r[2]) for r in rows]
            rows.sort(key=lambda r: (-r[0], -r[1], r[2].header.arrival_time))
            top = rows[:B]
            ratio = sum(r[0] for r in top) / mB
            ev_ratio = sum(r[1] for r in top) / mB   # tie-break (plan 6.3): expected value rate
            if (best is None or ratio > best[0] + _TIE
                    or (abs(ratio - best[0]) <= _TIE and ev_ratio > best[1] + 1e-15)):
                best = (ratio, ev_ratio, [r[2].id for r in top])
        return Decision(batch=tuple(best[2]), purge=tuple(sorted(purge)))


class GenieSVDPP(SVDPP):
    """Reference only: reads ground truth (true tier one-hot, delivered-dup flag). Fidelity is dropped (D5).

    This is the ONLY policy in this module that touches ``Truth``; ``SVDPP`` never does.
    """
    name = "genie_sv_dpp"
    uses_truth = True

    def __init__(self, truth, **kw):
        super().__init__(**kw)
        self.truth = dict(truth)
        self._delivered = set()

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self._delivered = set()

    def _probs(self, job_id, probs):
        t = self.truth[job_id].tier
        return tuple(1.0 if k == t else 0.0 for k in range(4))

    def _after_complete(self, ids):
        for i in ids:
            g = self.truth[i].dup_group
            if g is not None:
                self._delivered.add(g)

    def _view(self, job):
        t = self.truth[job.id]
        dup = 1.0 if (t.dup_group is not None and t.dup_group in self._delivered) else 0.0
        return self._probs(job.id, None), 1.0, dup     # D5: fidelity dropped (u = 1)


POLICIES = {"sv_dpp": SVDPP, "genie_sv_dpp": GenieSVDPP}
