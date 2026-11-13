"""SemCover learning (design B.3, SCOPE recipe). Train split only; labels (true tier) enter only the loss.

``SemCoverNet``: SetPolicy encoder (context embedding) + monotone urgency residual + kernel weights (softplus)
+ tier head. log q = log(analytic prior) + residual, residual = mono(p0, p0+p1, pressure) + ctx head (zero init);
the residual is non-decreasing in the three explicit inputs (non-negative weights, monotone activations) with
the context embedding held fixed.
Stage 1: sequential-DPP NLL of the teacher's batches (weighted by batch value) + tier cross-entropy.
Stage 2: policy gradient (PPO-clip on the sequential-DPP policy) with Lagrangian duals and annealed imitation.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Scheduler.edge.semcover import DT, SemCover, build_kernel, condition, sequential_logp
from Scheduler.training.edge.policy_net import SetPolicy
from Scheduler.training.edge.rollout import collect_episode
from Scheduler.training.edge.scope import ScopeConfig, discounted_returns, dual_update, imitation_weight
from Scheduler.training.edge.state_features import GLOBAL_DIM, JOB_DIM

_INV_SOFTPLUS_1 = 0.5413248546129181


class SemCoverNet(nn.Module):
    def __init__(self, hidden=32, mono_hidden=8):
        super().__init__()
        self.cfg = dict(hidden=hidden, mono_hidden=mono_hidden)
        self.enc = SetPolicy(JOB_DIM, GLOBAL_DIM, hidden)
        self.m1 = nn.Parameter(torch.randn(mono_hidden, 3) * 0.1)
        self.mb = nn.Parameter(torch.zeros(mono_hidden))
        self.m2 = nn.Parameter(torch.full((mono_hidden,), -5.0))
        self.ctx_head = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.ctx_head.weight)
        nn.init.zeros_(self.ctx_head.bias)
        self.tier_head = nn.Linear(2 * hidden, 4)
        self.kw = nn.Parameter(torch.full((3,), _INV_SOFTPLUS_1))

    def kernel_w(self):
        return F.softplus(self.kw).to(DT)

    def _ctx(self, jf, gf):
        job = torch.as_tensor(np.asarray(jf), dtype=torch.float32)[None]
        glob = torch.as_tensor(np.asarray(gf), dtype=torch.float32)[None]
        x = torch.cat([job, glob.unsqueeze(1).expand(-1, job.shape[1], -1)], -1)
        h = self.enc.phi(x)
        return torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], -1)[0], job[0]

    def mono(self, x):
        """x (n,3) = [p0, p0+p1, pressure] -> (n,), non-decreasing in every column."""
        return torch.tanh(x @ F.softplus(self.m1).T + self.mb) @ F.softplus(self.m2)

    def log_q(self, jf, gf):
        ctx, job = self._ctx(jf, gf)
        pressure = 1.0 - (job[:, 0:4] * job[:, 7:11]).sum(1)          # prob-weighted age / H_k, clipped to [0, 1]
        x = torch.stack([job[:, 0], job[:, 0] + job[:, 1], pressure], 1)
        return self.mono(x) + self.ctx_head(ctx).squeeze(-1)

    def tier_logits(self, jf, gf):
        return self.tier_head(self._ctx(jf, gf)[0])


def save_net(net, path):
    torch.save(dict(cfg=net.cfg, state=net.state_dict()), path)


def load_net(path):
    blob = torch.load(path, weights_only=False)
    net = SemCoverNet(**blob["cfg"])
    net.load_state_dict(blob["state"])
    return net.eval()


def decision_L(net, rec, zeta, eps=1e-3):
    q = (rec["log_prior"] + net.log_q(rec["jf"], rec["gf"]).to(DT)).clamp(-30, 5).exp()
    S = condition(build_kernel(rec["comps"], rec["crit"], net.kernel_w(), zeta, eps), rec["decay"], eps)
    return q[:, None] * S * q[None, :]


def _labels(rec, result):
    tier = {r["id"]: r["tier"] for r in result.outcomes}
    return torch.as_tensor([tier[i] for i in rec["ids"]], dtype=torch.long)


def _run(env_factory, net, mode, teacher, seed, lambdas=None):
    pol = SemCover(net=net.eval(), mode=mode, record=True)
    ep = collect_episode(env_factory, pol, teacher, seed, lambdas)
    spec = env_factory(seed).spec
    for rec in ep.decisions:
        rec["labels"] = _labels(rec, ep.result)
    return ep, spec


def _nll(net, rec, zeta):
    ch = rec["teacher"]
    return -sequential_logp(decision_L(net, rec, zeta), ch, rec["avail"]) if ch else torch.zeros((), dtype=DT)


def train_semcover(env_factory, teacher, iters=3, episodes=2, seed=0, hidden=32, epochs=3, lr=1e-2, cfg=None):
    cfg = cfg or ScopeConfig(lr=lr)
    net = SemCoverNet(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    zeta, hist = None, dict(stage1=[], stage2=[])
    # ---- stage 1: DPP likelihood of teacher batches (value-weighted) + tier CE ----
    recs = []
    for e in range(episodes):
        ep, spec = _run(env_factory, net, "teacher", teacher, seed + e)
        zeta = spec.zeta_critical_copy
        vmax = max(float(ep.rewards.max()), 1e-9) if len(ep.rewards) else 1.0
        for rec, r in zip(ep.decisions, ep.rewards):
            rec["wt"] = max(float(r), 0.0) / vmax + 0.05
            recs.append(rec)
    for _ in range(epochs):
        tot = 0.0
        for rec in recs:
            ce = F.cross_entropy(net.tier_logits(rec["jf"], rec["gf"]), rec["labels"])
            loss = rec["wt"] * _nll(net, rec, zeta).float() + 0.5 * ce
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        hist["stage1"].append(tot / max(len(recs), 1))
    # ---- stage 2: constrained policy gradient with annealed distillation ----
    lambdas, n_live = np.zeros(4), 0.0
    for it in range(iters):
        eps_ = [_run(env_factory, net, "sample", teacher, seed + 100 + it * episodes + e, lambdas)
                for e in range(episodes)]
        succ = np.mean([ep.tier_success for ep, _ in eps_], axis=0)
        batch = []
        for ep, spec in eps_:
            ret = discounted_returns(ep.rewards, cfg.gamma)
            batch += [(rec, float(r)) for rec, r in zip(ep.decisions, ret)]
            n_live += len(ep.decisions)
        rets = np.array([r for _, r in batch])
        adv = (rets - rets.mean()) / (rets.std() + 1e-8) if len(rets) > 1 else rets
        beta = imitation_weight(n_live, cfg.beta0, cfg.beta_decay_k)
        tot = 0.0
        for _ in range(cfg.epochs):
            for (rec, _), a in zip(batch, adv):
                logp = sequential_logp(decision_L(net, rec, zeta), rec["chosen"], rec["avail"]).float()
                ratio = torch.exp(logp - rec["log_prob"])
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a)
                ce = F.cross_entropy(net.tier_logits(rec["jf"], rec["gf"]), rec["labels"])
                loss = pg + beta * _nll(net, rec, zeta).float() + 0.2 * ce
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss)
        lambdas = dual_update(lambdas, succ, spec.eta, cfg.lambda_lr)
        hist["stage2"].append(dict(loss=tot / max(len(batch) * cfg.epochs, 1), beta=float(beta),
                                   lambdas=[float(x) for x in lambdas], tier_success=[float(x) for x in succ]))
    return net.eval(), hist
