"""Train the BCEdge batch-size actor: entropy-regularised advantage actor-critic on EdgeEnv episodes.

Reward per decision = realised value of its decoded jobs (reward_attribution); returns are discounted;
loss = -logp(a) * advantage + 0.5 * (R - V)^2 - alpha * H(pi), with illegal batch sizes masked out.
"""
import numpy as np
import torch

from Scheduler.edge.bcedge import BCEdge, make_net
from Scheduler.training.edge.reward_attribution import attribute_rewards
from Scheduler.training.edge.scope import discounted_returns


def train_bcedge(env_factory, iters=5, episodes=2, hidden=32, lr=3e-3, gamma=0.95, alpha=0.05, seed=0, epochs=4):
    """Returns (net, history list of dicts)."""
    torch.manual_seed(seed)
    net = make_net(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(int(iters)):
        xs, masks, acts, rets, total = [], [], [], [], 0.0
        for e in range(int(episodes)):
            s = seed * 1_000_003 + it * episodes + e   # seeds no longer share episodes
            env = env_factory(s)
            pol = BCEdge(net, sample=True, record=True, seed=s)
            res = env.run(pol)
            rew = attribute_rewards([r[3] for r in pol.records], res.outcomes, env.spec)
            rets += list(discounted_returns(rew, gamma))
            xs += [r[0] for r in pol.records]
            masks += [r[1] for r in pol.records]
            acts += [r[2] for r in pol.records]
            total += float(rew.sum())
        if not xs:
            continue
        x, m = torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(masks))
        a, ret = torch.as_tensor(acts), torch.as_tensor(np.asarray(rets), dtype=torch.float32)
        ret = ret / (ret.std() + 1e-8)           # scaled returns: value loss O(1), not O(1e4) (2026-11-13)
        for _ in range(int(epochs)):            # was a single gradient step per iteration
            logits, v = net(x)
            logp = torch.log_softmax(logits.masked_fill(~m, -1e9), dim=-1)
            ent = -(logp.exp() * logp.masked_fill(~m, 0.0)).sum(-1).mean()
            adv = (ret - v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
            loss = -(logp.gather(1, a[:, None]).squeeze(1) * adv).mean() + 0.5 * ((ret - v) ** 2).mean() - alpha * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        hist.append(dict(iter=it, mean_return=total / episodes, loss=float(loss.detach()), entropy=float(ent.detach()),
                         decisions=len(xs)))
    return net, hist
