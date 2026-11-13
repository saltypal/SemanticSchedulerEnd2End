"""Training loops: plain PPO, SCOPE-PPO, behaviour cloning and DAgger for the set policy.

All take ``env_factory(seed) -> EdgeEnv``. Teachers are any ``Policy`` (e.g. SVDPP), queried at the
states the learner visits through ``policy_net.teacher_scores`` (chosen batch -> smoothed target).
Each returns ``(model, history)``; weights persist via ``policy_net.save_model/load_model``.
"""
from dataclasses import replace

import numpy as np
import torch

from Scheduler.training.edge.policy_net import LearnedPolicy, SetPolicy, pad_batch
from Scheduler.training.edge.rollout import collect_episode
from Scheduler.training.edge.scope import (ScopeConfig, build_batch, discounted_returns, dual_update,
                                           scope_ppo_update)


def _ppo_loop(env_factory, model, cfg, iters, episodes, seed, teacher, use_dual):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lambdas, n_live, history = None, 0.0, []
    for it in range(iters):
        decisions, returns, rewards, succ = [], [], [], []
        for e in range(episodes):
            pol = LearnedPolicy(model, "sample", teacher, teacher_eps=cfg.teacher_eps)
            ep = collect_episode(env_factory, pol, seed=seed + it * episodes + e, lambdas=lambdas)
            if lambdas is None:
                lambdas = np.zeros(len(ep.tier_success))
            decisions += ep.decisions
            returns += list(discounted_returns(ep.rewards, cfg.gamma))
            rewards.append(float(ep.rewards.sum()))
            succ.append(ep.tier_success)
        if not decisions:
            continue
        n_live += cfg.alpha_live * len(decisions)
        batch = build_batch(decisions, np.asarray(returns))
        stats = {}
        for _ in range(cfg.epochs):
            stats = scope_ppo_update(model, opt, batch, cfg, lambdas, n_live)
        j = np.mean(succ, axis=0)
        history.append(dict(iter=it, mean_reward=float(np.mean(rewards)), tier_success=j.tolist(),
                            lambdas=[float(x) for x in lambdas], beta=stats["beta"], loss=stats["loss"]))
        if use_dual:
            lambdas = dual_update(lambdas, j, cfg.etas, cfg.lambda_lr)
    return model, history


def train_ppo(env_factory, iters=5, episodes=2, cfg=None, model=None, seed=0):
    """Plain PPO: beta = 0, lambdas fixed at 0, no teacher."""
    cfg = replace(cfg or ScopeConfig(), beta0=0.0)
    return _ppo_loop(env_factory, model or SetPolicy(), cfg, iters, episodes, seed, None, False)


def train_scope_ppo(env_factory, teacher, iters=5, episodes=2, cfg=None, model=None, seed=0):
    cfg = cfg or ScopeConfig()
    if cfg.etas is None:
        cfg = replace(cfg, etas=tuple(env_factory(seed).spec.eta))
    return _ppo_loop(env_factory, model or SetPolicy(), cfg, iters, episodes, seed, teacher, True)


def bc_fit(model, decisions, epochs=50, lr=1e-2):
    """Minimise cross-entropy between the first-pick softmax and the teacher target. Returns losses."""
    batch = pad_batch([d for d in decisions if d.get("teacher") is not None])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        logits, _ = model(batch["job"], batch["glob"], batch["mask"])
        lp = torch.log_softmax(logits, dim=-1).masked_fill(~batch["mask"], 0.0)
        loss = -(batch["teacher"] * lp).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_bc(env_factory, teacher, episodes=4, epochs=50, lr=1e-2, model=None, seed=0):
    """Clone the teacher on the states it visits itself (LearnedPolicy ``teacher`` mode)."""
    torch.manual_seed(seed)
    model = model or SetPolicy()
    data = []
    for e in range(episodes):
        data += collect_episode(env_factory, LearnedPolicy(model, "teacher", teacher), seed=seed + e).decisions
    losses = bc_fit(model, data, epochs, lr)
    return model, [dict(iter=0, bc_loss_first=losses[0], bc_loss_last=losses[-1], n_samples=len(data))]


def train_dagger(env_factory, teacher, iters=3, episodes=2, epochs=30, lr=1e-2, model=None, seed=0):
    """Iter 0 visits teacher states; later iters run the learner greedily; all states are relabelled by
    the teacher (inside the policy) and aggregated before retraining."""
    torch.manual_seed(seed)
    model = model or SetPolicy()
    data, history = [], []
    for it in range(iters):
        mode = "teacher" if it == 0 else "greedy"
        rew = []
        for e in range(episodes):
            ep = collect_episode(env_factory, LearnedPolicy(model, mode, teacher), seed=seed + it * episodes + e)
            data += ep.decisions
            rew.append(float(ep.rewards.sum()))
        losses = bc_fit(model, data, epochs, lr)
        history.append(dict(iter=it, mean_reward=float(np.mean(rew)), n_samples=len(data), bc_loss=losses[-1]))
    return model, history


# ---- v3 learners wired into train_cli: DQN, SCOPE-DQN, LTR (LambdaMART / RankNet), SCOPE-LTR ----

def train_dqn_plain(env_factory, episodes=6, seed=0, hidden=32, **kw):
    """Plain double-DQN, no teacher (margin loss off). Returns (model, history)."""
    from Scheduler.training.edge import dqn
    return dqn.train_dqn(env_factory, episodes, teacher=None, hidden=hidden, seed=seed, **kw)


def train_scope_dqn(env_factory, teacher, iters=3, episodes=2, seed=0, hidden=32, lambda_lr=0.05, **kw):
    """SCOPE-DQN: DQfD margin loss against ``teacher`` (annealed beta) + dual-updated tier penalties.

    Each outer iteration trains ``episodes`` episodes with the current lambdas, then measures per-tier
    timely success of the greedy policy on one fresh episode and applies the dual update."""
    from Scheduler.training.edge import dqn
    from Scheduler.training.edge.rollout import tier_success_rates
    lam, model, history = np.zeros(4), None, []
    etas = tuple(env_factory(seed).spec.eta)
    for it in range(iters):
        model, h = dqn.train_dqn(lambda e, it=it: env_factory(seed + it * episodes + e), episodes,
                                 teacher=teacher, model=model, hidden=hidden, margin_weight=1.0,
                                 beta_k=500.0, lambdas=tuple(lam), eps_start=0.5, seed=seed + it, **kw)
        env = env_factory(seed + 1000 + it)
        res = env.run(dqn.DQNPolicy(model))
        j = tier_success_rates(res.measured() or res.outcomes, env.spec)
        lam = dual_update(lam, j, etas, lambda_lr)
        history.append(dict(iter=it, lambdas=lam.tolist(), tier_success=j.tolist(), loss=h[-1]["loss"]))
    return model, history


def _teacher_episodes(env_factory, teacher, episodes, seed):
    from Scheduler.training.edge.reward_attribution import LoggingPolicy
    out = []
    for e in range(episodes):
        pol = LoggingPolicy(teacher)
        out.append((pol, env_factory(seed + e).run(pol)))
    return [(p.records, r.outcomes) for p, r in out]


def train_ltr(env_factory, teacher, ranker="lambdamart", episodes=4, seed=0, **kw):
    """Learning to rank on the realised values of the teacher's decisions.

    ``ranker`` is 'lambdamart' (lightgbm, else sklearn pointwise fallback) or 'ranknet'."""
    from Scheduler.training.edge import ltr
    data = ltr.build_ranking_dataset(_teacher_episodes(env_factory, teacher, episodes, seed))
    if ranker == "ranknet":
        model = ltr.train_ranknet(data, seed=seed, **kw)
    elif ranker == "lambdamart":
        model = ltr.train_lambdamart(data, seed=seed, **kw)
    else:
        raise ValueError("ranker must be lambdamart or ranknet")
    return model, [dict(iter=0, ranker=ranker, n_rows=int(len(data["y"])), n_groups=int(len(data["group"])))]


def train_scope_ltr(env_factory, teacher, iters=3, episodes=3, epochs=30, lr=1e-2, lambda_lr=0.05, hidden=32,
                    seed=0):
    """SCOPE-LTR: RankNet scorer fitted with the Lagrangian-reweighted ListNet distillation loss towards
    the teacher's batch choices; lambdas follow a dual update from the scorer's own greedy rollout."""
    from Scheduler.training.edge import ltr
    from Scheduler.training.edge.rollout import tier_success_rates
    torch.manual_seed(seed)
    recs = [r for rs, _ in _teacher_episodes(env_factory, teacher, episodes, seed) for r in rs]
    if not recs:
        raise ValueError("teacher produced no decisions")
    model = ltr.RankNet(recs[0].job_x.shape[1] + recs[0].glob.shape[0], hidden)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    etas = tuple(env_factory(seed).spec.eta)
    lam, history = np.zeros(4), []
    for it in range(iters):
        for _ in range(epochs):
            loss = 0.0
            for r in recs:
                x = torch.as_tensor(ltr._x(r.job_x, r.glob), dtype=torch.float32)
                t = torch.zeros(len(r.ids))
                t[list(r.idx)] = 3.0
                loss = loss + ltr.listwise_distill_loss(model.net(x).squeeze(-1), t, lam, r.job_x[:, :4])
            loss = loss / len(recs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        env = env_factory(seed + 1000 + it)
        res = env.run(ltr.LTRPolicy(model))
        j = tier_success_rates(res.measured() or res.outcomes, env.spec)
        lam = dual_update(lam, j, etas, lambda_lr)
        history.append(dict(iter=it, loss=float(loss.detach()), lambdas=lam.tolist(), tier_success=j.tolist()))
    return model, history


def train_semcover(env_factory, teacher, **kw):
    """SemCover (semantic-coverage DPP scheduler): see semcover_train (imported lazily, keeps torch-only deps local)."""
    from Scheduler.training.edge.semcover_train import train_semcover as _train
    return _train(env_factory, teacher, **kw)
