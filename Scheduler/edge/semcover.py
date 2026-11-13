"""SemCover: semantic-coverage scheduler (docs/design/SEMANTIC_SCHEDULER_DESIGN.md section B). Header-only.

Selection maths (torch float64):
* Kernel S = w1*cos(pi*d_sem/64) + w2*cos(7-class STS histogram; 0 if sts empty) + w3*[d_perc <= 8], w >= 0.
  Critical floor: for two DIFFERENT jobs both believed Critical (fused p0 > 0.5) S_ij *= (1 - zeta_C), so a
  corroborating Critical copy keeps marginal gain >= 1 - (1-zeta_C)^2 (0.4375 at zeta_C = 0.25), never fully
  suppressed. cos(pi d/64) and the d_perc indicator are not PSD in general, so the combined kernel is
  PSD-PROJECTED (eigenvalue clip at 0), normalised to unit diagonal and gets a ridge eps*I (design said "PSD by construction").
* Memory conditioning (time-decayed): a delivered header m has decay a_m = exp(-age_m / tau_k), k = argmax tier of
  its belief, tau_k = tau_mult * H_k; sliding window of `window` newest, a_m < 0.02 dropped. Memory is observed
  through a noisy channel, i.e. Schur complement S|M = S_QQ - S_QM A (A S_MM A + I - A^2)^-1 A S_MQ
  (a=1: plain DPP conditioning, a=0: no suppression). Stays PSD.
* L = diag(q) S|M diag(q); exact incremental-Cholesky greedy MAP (Chen et al. 2018). Alternative objective
  "facility": sum_i q_i * max(base_i, max_{j in B} Sn_ij), Sn = clipped, diag-normalised S (monotone submodular).
* q_i = (sum_k p_k w_k) * G_i(now) [* exp(net residual)]; G_i = sum_k p_k CDF_B(H_k - age). Jobs that cannot meet
  any live-tier H_k are masked out (not purged).
"""
import math
from collections import deque

import numpy as np
import torch

from Evaluation.edge.contracts import Decision, Policy

DT = torch.float64
NEG = -1e30
PERC_NEAR = 8
MEM_MIN_DECAY = 0.02


# ---------------- kernel ----------------
def _ham(keys):
    a = np.array(keys, dtype=np.uint64)
    x = np.ascontiguousarray(a[:, None] ^ a[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(len(a), len(a), 8), axis=-1).sum(-1)


def pair_components(headers):
    """(3, N, N) tensor [cos(pi d_sem/64), STS cosine, near-copy indicator] and (N,) Critical flags (p0 > 0.5)."""
    n = len(headers)
    c_sem = np.cos(np.pi * _ham([h.sem_key for h in headers]) / 64.0)
    v = np.array([list(h.sts[:7]) if len(h.sts) >= 7 else [0.0] * 7 for h in headers], dtype=float).reshape(n, 7)
    nrm = np.linalg.norm(v, axis=1)
    vn = v / np.where(nrm > 0, nrm, 1.0)[:, None]
    c_perc = (_ham([h.perc_key for h in headers]) <= PERC_NEAR).astype(float)
    comps = torch.as_tensor(np.stack([c_sem, vn @ vn.T, c_perc]), dtype=DT)
    crit = torch.as_tensor([h.tier_probs[0] > 0.5 for h in headers], dtype=torch.bool)
    return comps, crit


def psd_project(S):
    w, v = torch.linalg.eigh((S + S.T) / 2)
    return (v * w.clamp(min=0)) @ v.T


def build_kernel(comps, crit, w, zeta, eps):
    """Full (N, N) kernel: weighted sum, Critical floor, PSD projection, ridge."""
    n = comps.shape[1]
    S = (w.to(DT)[:, None, None] * comps).sum(0)
    both = crit[:, None] & crit[None, :] & ~torch.eye(n, dtype=torch.bool)
    S = S * torch.where(both, 1.0 - zeta, 1.0).to(DT)
    S = psd_project(S)
    d = torch.sqrt(torch.diagonal(S).clamp(min=1e-12))
    return S / d[:, None] / d[None, :] + eps * torch.eye(n, dtype=DT)


def condition(S, decay, eps=0.0):
    """Schur-condition the queue block (rows m:) of the joint [memory; queue] kernel on decayed memory."""
    m = decay.shape[0]
    if m == 0:
        return S
    a = decay.to(DT)
    Smq = a[:, None] * S[:m, m:]
    Amm = a[:, None] * S[:m, :m] * a[None, :] + torch.diag(1.0 - a ** 2) + eps * torch.eye(m, dtype=DT)
    return S[m:, m:] - Smq.T @ torch.linalg.solve(Amm, Smq)


def build_cond_kernel(comps, crit, decay, w, zeta, eps):
    return condition(build_kernel(comps, crit, w, zeta, eps), decay, eps)


# ---------------- selection ----------------
def greedy_map(L, k, avail, rng=None):
    """Incremental-Cholesky greedy MAP of det(L_B) (Chen et al. 2018). With rng: sequential sampling prop. to the
    marginal gains (greedy = its mode). Continues past zero gain so the batch size is exact."""
    n = L.shape[0]
    d2, c, sel, avail = torch.diagonal(L).clone(), torch.zeros(k, n, dtype=DT), [], avail.clone()
    tie = 1e-12 * torch.diagonal(L).clamp(min=0)
    for t in range(min(k, int(avail.sum()))):
        if rng is None:
            j = int(torch.argmax(torch.where(avail, d2.clamp(min=0) + tie, torch.full_like(d2, NEG))))
        else:
            p = torch.where(avail, d2.clamp(min=0) + 1e-15, torch.zeros_like(d2)).numpy()
            j = int(rng.choice(n, p=p / p.sum()))
        sel.append(j)
        avail[j] = False
        dj = torch.sqrt(d2[j].clamp(min=1e-12))
        e = (L[j] - c[:t, j] @ c[:t]) / dj
        c[t] = e
        d2 = d2 - e ** 2
    return sel


def sequential_logp(L, chosen, avail):
    """Differentiable log-prob of an ordered batch under sequential gain sampling (p(j|S) prop. to d2_j)."""
    n = L.shape[0]
    d2, cs, avail, total = torch.diagonal(L), [], avail.clone(), L.new_zeros(())
    for j in chosen:
        g = d2.clamp(min=1e-15)
        total = total + torch.log(g[j]) - torch.log((g * avail.to(DT)).sum())
        avail = avail.clone()
        avail[j] = False
        e = (L[j] - sum((c[j] * c for c in cs), L.new_zeros(n))) / torch.sqrt(g[j])
        cs.append(e)
        d2 = d2 - e ** 2
    return total


def facility_inputs(S, decay):
    """From the full joint kernel: queue similarity Sn in [0,1] (unit diagonal) and memory baseline coverage."""
    m = decay.shape[0]
    d = torch.sqrt(torch.diagonal(S).clamp(min=1e-12))
    Sn = (S / d[:, None] / d[None, :]).clamp(0.0, 1.0)
    base = torch.zeros(S.shape[0] - m, dtype=DT)
    if m:
        base = (Sn[m:, :m] * decay.to(DT)[None, :]).max(1).values
    return Sn[m:, m:], base


def facility_value(Sn, q, base, batch):
    cov = torch.maximum(base, Sn[:, batch].max(1).values) if len(batch) else base
    return float((q * (cov - base)).sum())


def greedy_facility(Sn, q, base, k, avail):
    cov, sel, avail = base.clone(), [], avail.clone()
    for _ in range(min(k, int(avail.sum()))):
        gain = (q[:, None] * (Sn - cov[:, None]).clamp(min=0)).sum(0)
        j = int(torch.argmax(torch.where(avail, gain, torch.full_like(gain, NEG))))
        sel.append(j)
        avail[j] = False
        cov = torch.maximum(cov, Sn[:, j])
    return sel


# ---------------- policy ----------------
def _bsize(state, n):
    sizes = [b for b in state.latency.batch_sizes if b <= min(state.max_batch, n)]
    return max(sizes) if sizes else min(state.latency.batch_sizes)


class SemCover(Policy):
    """Analytic SemCover (net=None, kernel weights 1,1,1) or learned (net: log_q(jf, gf) residual, kernel_w())."""
    name = "semcover"
    uses_truth = False

    def __init__(self, objective="logdet", net=None, w=(1.0, 1.0, 1.0), eps=1e-3, tau_mult=2.0, window=32,
                 mode="greedy", teacher=None, record=False):
        if objective not in ("logdet", "facility"):
            raise ValueError("objective must be logdet|facility")
        if mode not in ("greedy", "sample", "teacher"):
            raise ValueError("mode must be greedy|sample|teacher")
        self.objective, self.net, self.w = objective, net, torch.as_tensor(w, dtype=DT)
        self.eps, self.tau_mult, self.window = eps, tau_mult, int(window)
        self.mode, self.teacher, self.record = mode, teacher, record
        self.memory, self._pending, self.decisions = deque(maxlen=self.window), {}, []
        self.np_rng = np.random.default_rng(0)

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.memory, self._pending, self.decisions = deque(maxlen=self.window), {}, []
        self.np_rng = np.random.default_rng(int(rng.integers(2 ** 31)))
        if self.teacher is not None:
            self.teacher.reset(spec, np.random.default_rng(int(rng.integers(2 ** 31))))

    def observe(self, event, payload):
        if self.teacher is not None:
            self.teacher.observe(event, payload)
        if event == "complete":
            for i in payload["ids"]:
                if i in self._pending:
                    self.memory.append((self._pending.pop(i), payload["time"]))

    def _live_memory(self, now):
        out = []
        for h, t in self.memory:
            k = int(np.argmax(h.tier_probs))
            a = math.exp(-max(now - t, 0.0) / (self.tau_mult * self.spec.hard_deadline(k)))
            if a >= MEM_MIN_DECAY:
                out.append((h, a))
        return out

    def prepare(self, state):
        """Everything a decision needs (also the training record). None when nothing is selectable."""
        spec, lat, qs = state.spec, state.latency, state.queue
        P = np.array([j.header.tier_probs for j in qs], dtype=float)
        ages = np.array([j.age for j in qs])
        H = np.array([spec.hard_deadline(k) for k in range(4)])
        b1 = lat.batch_sizes[0]
        feas = np.array([any(P[i, k] > 0.05 and lat.cdf(b1, H[k] - ages[i]) > 0 for k in range(4))
                         for i in range(len(qs))])
        if not feas.any():
            return None
        B = _bsize(state, int(feas.sum()))
        G = np.array([sum(P[i, k] * lat.cdf(B, H[k] - ages[i]) for k in range(4)) for i in range(len(qs))])
        ev = P @ np.asarray(spec.weights, dtype=float)
        mem = self._live_memory(state.now)
        comps, crit = pair_components([h for h, _ in mem] + [j.header for j in qs])
        prep = dict(B=B, avail=torch.as_tensor(feas), comps=comps, crit=crit, ids=[j.id for j in qs],
                    decay=torch.as_tensor([a for _, a in mem], dtype=DT), time=state.now,
                    log_prior=torch.as_tensor(np.log(ev + 1e-9) + np.log(G + 1e-9), dtype=DT))
        if self.net is not None:
            from Scheduler.training.edge.state_features import global_features, job_features
            prep.update(jf=job_features(state), gf=global_features(state))
        return prep

    def kernel_and_q(self, prep):
        lq, w = prep["log_prior"], self.w
        if self.net is not None:
            with torch.no_grad():
                lq = lq + self.net.log_q(prep["jf"], prep["gf"]).to(DT)
                w = self.net.kernel_w()
        S = build_kernel(prep["comps"], prep["crit"], w, self.spec.zeta_critical_copy, self.eps)
        return lq.clamp(-30, 5).exp(), S

    def select(self, state):
        if not state.queue:
            return Decision()
        prep = self.prepare(state)
        if prep is None:
            return Decision()
        q, S = self.kernel_and_q(prep)
        k, avail, decay = prep["B"], prep["avail"], prep["decay"]
        Sc = condition(S, decay, self.eps)
        L = q[:, None] * Sc * q[None, :]
        if self.objective == "facility":
            Sn, base = facility_inputs(S, decay)
            chosen = greedy_facility(Sn, q, base, k, avail)
        else:
            chosen = greedy_map(L, k, avail, self.np_rng if self.mode == "sample" else None)
        tchosen = None
        if self.teacher is not None:
            pos = {j.id: i for i, j in enumerate(state.queue)}
            tchosen = [pos[i] for i in self.teacher.select(state).batch if i in pos]
            if self.mode == "teacher" and tchosen:
                chosen = tchosen
        if self.record:
            self.decisions.append(dict(prep, chosen=list(chosen), teacher=tchosen,
                                       log_prob=float(sequential_logp(L, chosen, avail))))
        for i in chosen:
            self._pending[state.queue[i].id] = state.queue[i].header
        return Decision(batch=tuple(state.queue[i].id for i in chosen))


POLICIES = {"semcover": SemCover}
