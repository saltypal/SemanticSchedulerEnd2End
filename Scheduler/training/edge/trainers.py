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
