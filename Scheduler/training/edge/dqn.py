"""Per-job DQN scheduler (plan v3 sec. 7).

Q(job | state) is a shared MLP over [job features, global features]. A batch's value is the mean Q
of its jobs; the policy takes the top-B jobs by Q (epsilon-greedy). Double-DQN target with a target
network; variable-length queues: the next-state value is the mean of the top-B next-queue Q values
(selected by the online net, evaluated by the target net). Optional DQfD large-margin loss against a
teacher batch with an annealed weight beta(n) = beta0 / (1 + n / k).
Reuses: reward_attribution (logging, rewards), state_features (encoders).
"""
import random
from dataclasses import dataclass

import numpy as np

from Evaluation.edge.contracts import Decision, Policy
from Scheduler.training.edge.reward_attribution import attribute_rewards, make_record
from Scheduler.training.edge.state_features import GLOBAL_DIM, JOB_DIM, global_features, job_features


def anneal_beta(n, beta0, k):
    """beta(n) = beta0 / (1 + n / k); constant beta0 when k is None."""
    return beta0 if not k else beta0 / (1.0 + n / k)


def make_qnet(hidden=64, job_dim=JOB_DIM, global_dim=GLOBAL_DIM):
    import torch
    from torch import nn

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(job_dim + global_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, job_x, glob):
            """job_x (N, job_dim), glob (global_dim,) -> (N,) Q values."""
            g = glob.reshape(1, -1).expand(job_x.shape[0], -1)
            return self.net(torch.cat([job_x, g], dim=1)).squeeze(-1)

    return QNet()


def _t(a):
    import torch
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def batch_value(q, idx):
    """Mean Q of the chosen positions."""
    return q[list(idx)].mean()


def large_margin_loss(q, teacher_idx, margin):
    """DQfD: max_j (Q_j + margin * [j not in teacher]) - mean Q(teacher)."""
    import torch
    mask = torch.full_like(q, margin)
    mask[list(teacher_idx)] = 0.0
    return (q + mask).max() - q[list(teacher_idx)].mean()


@dataclass
class Transition:
    job_x: np.ndarray
    glob: np.ndarray
    idx: tuple
    batch: tuple
    reward: float
    next_job_x: np.ndarray
    next_glob: np.ndarray
    done: bool
    teacher_idx: tuple = ()


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.capacity, self.items, self.pos = int(capacity), [], 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def push(self, tr):
        if len(self.items) < self.capacity:
            self.items.append(tr)
        else:
            self.items[self.pos] = tr
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n):
        return self.rng.sample(self.items, min(n, len(self.items)))


def records_to_transitions(records, rewards):
    """Chain consecutive decisions of one episode into (s, a, r, s', done)."""
    if len(records) != len(rewards):
        raise ValueError("records and rewards differ in length")
    out = []
    for t, (rec, r) in enumerate(zip(records, rewards)):
        last = t == len(records) - 1
        nxt = rec if last else records[t + 1]
        out.append(Transition(rec.job_x, rec.glob, rec.idx, rec.batch, float(r),
                              rec.job_x[:0] if last else nxt.job_x, nxt.glob, last, rec.teacher_idx))
    return out


def sync_target(online, target):
    target.load_state_dict(online.state_dict())


def dqn_loss(online, target, trs, gamma, margin_weight=0.0, margin=0.8):
    """Mean Huber TD loss (double DQN) plus optional margin term over a list of transitions."""
    import torch
    td, mg = [], []
    for tr in trs:
        q = online(_t(tr.job_x), _t(tr.glob))
        qa = batch_value(q, tr.idx)
        y = tr.reward
        if not tr.done and len(tr.next_job_x):
            with torch.no_grad():
                nj, ng = _t(tr.next_job_x), _t(tr.next_glob)
                top = torch.topk(online(nj, ng), min(len(tr.idx), nj.shape[0])).indices
                y = tr.reward + gamma * float(target(nj, ng)[top].mean())
        td.append(torch.nn.functional.smooth_l1_loss(qa, torch.tensor(float(y))))
        if margin_weight and tr.teacher_idx:
            mg.append(large_margin_loss(q, tr.teacher_idx, margin))
    loss = torch.stack(td).mean()
    if mg:
        loss = loss + margin_weight * torch.stack(mg).mean()
    return loss


class DQNPolicy(Policy):
    """Epsilon-greedy over the queue; batch = top-B by Q. Records decisions when ``record``."""
    name = "dqn"

    def __init__(self, model, epsilon=0.0, teacher=None, record=False, seed=0):
        self.model, self.epsilon, self.teacher, self.record = model, epsilon, teacher, record
        self.records, self._rng = [], random.Random(seed)

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.records = []
        if self.teacher is not None:
            self.teacher.reset(spec, rng)

    def observe(self, event, payload):
        if self.teacher is not None:
            self.teacher.observe(event, payload)

    @staticmethod
    def _size(state):
        cap = min(state.max_batch, len(state.queue))
        ok = [b for b in state.latency.batch_sizes if b <= cap]
        return max(ok) if ok else cap

    def select(self, state):
        import torch
        if not state.queue:
            return Decision()
        ids = [j.id for j in state.queue]
        if self._rng.random() < self.epsilon:
            order = list(range(len(ids)))
            self._rng.shuffle(order)
        else:
            with torch.no_grad():
                q = self.model(_t(job_features(state)), _t(global_features(state))).numpy()
            order = list(np.argsort(-q, kind="stable"))
        dec = Decision(tuple(ids[i] for i in order[:self._size(state)]))
        if self.record:
            tb = self.teacher.select(state).batch if self.teacher is not None else ()
            self.records.append(make_record(state, dec, tb))
        return dec


def train_dqn(env_factory, episodes, teacher=None, model=None, hidden=64, gamma=0.95, lr=1e-3,
              batch_size=32, updates_per_episode=20, target_sync=10, eps_start=1.0, eps_end=0.05,
              margin_weight=0.0, margin=0.8, beta0=1.0, beta_k=None, lambdas=(0, 0, 0, 0),
              reward_scale=1.0, buffer_size=50000, seed=0, value_fn=None):
    """Train a DQN. ``env_factory(episode) -> EdgeEnv``. Returns (model, history list of dicts)."""
    import torch
    torch.manual_seed(seed)
    model = model or make_qnet(hidden)
    target = make_qnet(hidden)
    sync_target(model, target)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buf, hist, step = ReplayBuffer(buffer_size, seed), [], 0
    for ep in range(int(episodes)):
        eps = eps_start + (eps_end - eps_start) * ep / max(episodes - 1, 1)
        env = env_factory(ep)
        pol = DQNPolicy(model, eps, teacher, record=True, seed=seed + ep)
        res = env.run(pol)
        rew = attribute_rewards([r.batch for r in pol.records], res.outcomes, env.spec, lambdas, value_fn)
        for tr in records_to_transitions(pol.records, rew * reward_scale):
            buf.push(tr)
        losses = []
        for _ in range(updates_per_episode if len(buf) else 0):
            coef = margin_weight * anneal_beta(step, beta0, beta_k) if teacher is not None else 0.0
            loss = dqn_loss(model, target, buf.sample(batch_size), gamma, coef, margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % target_sync == 0:
                sync_target(model, target)
        hist.append({"episode": ep, "epsilon": eps, "return": float(rew.sum()),
                     "loss": float(np.mean(losses)) if losses else None, "decisions": len(rew)})
    return model, hist
