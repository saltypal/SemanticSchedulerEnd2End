"""SCOPE-PPO pieces: config, imitation-weight schedule, PPO-clip update with KL-to-teacher, dual update.

Constraint costs enter ONLY through the returns (the per-decision reward already contains
-sum_k lambda_k c_k); the loss has no separate lambda*(CMR-eta) term. The teacher distribution comes from
``policy_net.teacher_scores`` (batch members share (1-eps) mass, eps smoothing); the KL is taken on the
first-pick softmax over the queue. beta(n) = beta0/(1+n_live/k) with n_live = alpha_live * decisions seen.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from Scheduler.training.edge.policy_net import pad_batch, plackett_luce_logp


@dataclass
class ScopeConfig:
    alpha_live: float = 1.0
    beta0: float = 1.0
    beta_decay_k: float = 500.0
    etas: Optional[Tuple[float, ...]] = None   # defaults to spec.eta in the trainer
    lambda_lr: float = 0.05
    lr: float = 3e-3
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gamma: float = 0.99
    epochs: int = 4
    teacher_eps: float = 0.1


def imitation_weight(n_live, beta0, k):
    return beta0 / (1.0 + n_live / k)


def dual_update(lambdas, tier_success, etas, lr):
    return np.maximum(0.0, np.asarray(lambdas, float) + lr * (np.asarray(etas, float) - np.asarray(tier_success, float)))


def discounted_returns(rewards, gamma):
    out, acc = np.zeros(len(rewards)), 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def build_batch(decisions, returns, ret_scale=1.0):
    """``ret_scale`` divides the returns: the value head learns scaled returns, so its loss stays O(1) and does not
    swamp the policy gradient under the global grad-norm clip (raw returns are O(100): vf ~ 4e4 vs pg ~ 1)."""
    b = pad_batch(decisions)
    ret = torch.as_tensor(returns, dtype=torch.float32) / float(ret_scale)
    val = torch.as_tensor([d["value"] for d in decisions], dtype=torch.float32)
    adv = ret - val
    b.update(ret=ret, adv=(adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv)
    return b


def scope_ppo_update(model, optimizer, batch, cfg, lambdas, n_live=0.0, beta=None):
    """One PPO epoch on ``batch``. ``lambdas`` is reported only (already inside the returns)."""
    logits, value = model(batch["job"], batch["glob"], batch["mask"])
    new_logp = plackett_luce_logp(logits, batch["mask"], batch["chosen"])
    ratio = torch.exp(new_logp - batch["old_logp"])
    adv = batch["adv"]
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
    vf = ((value - batch["ret"]) ** 2).mean()
    logp1 = torch.log_softmax(logits, dim=-1)
    ent = -(logp1.exp() * logp1.masked_fill(~batch["mask"], 0.0)).sum(-1).mean()
    beta = imitation_weight(n_live, cfg.beta0, cfg.beta_decay_k) if beta is None else float(beta)
    kl = torch.zeros(())
    if beta > 0 and bool(batch["has_teacher"].any()):
        t = batch["teacher"].clamp(min=1e-8)
        per = (logp1.exp() * (logp1 - torch.log(t))).masked_fill(~batch["mask"], 0.0).sum(-1)
        kl = per[batch["has_teacher"]].mean()
    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + beta * kl
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return dict(loss=float(loss.detach()), pg=float(pg), vf=float(vf), entropy=float(ent), kl=float(kl),
                beta=float(beta), lambdas=[float(x) for x in np.asarray(lambdas)])
