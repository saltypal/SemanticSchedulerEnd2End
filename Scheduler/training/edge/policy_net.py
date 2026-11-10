"""Set-based queue policy network and the ``LearnedPolicy`` wrapper used inside ``EdgeEnv.run``.

``SetPolicy``: per-job MLP phi over [job features, global features], masked mean pooling, per-job logit
head on [phi_i, pooled] and a value head on the pooled vector. Variable queue length is handled by
padding + mask. A batch of size B is a Plackett-Luce draw (top-B by logit when greedy), so its
log-probability is the sum of sequential softmax terms with chosen items removed.
Reuses ``state_features`` (never modified) and the ``contracts`` Policy/Decision types.
"""
import numpy as np
import torch
from torch import nn

from Evaluation.edge.contracts import Decision, Policy
from Scheduler.training.edge.state_features import GLOBAL_DIM, JOB_DIM, global_features, job_features

NEG = -1e9


class SetPolicy(nn.Module):
    def __init__(self, job_dim=JOB_DIM, global_dim=GLOBAL_DIM, hidden=32):
        super().__init__()
        self.cfg = dict(job_dim=job_dim, global_dim=global_dim, hidden=hidden)
        self.phi = nn.Sequential(nn.Linear(job_dim + global_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.logit_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, job, glob, mask=None):
        """job (B,N,D), glob (B,G), mask (B,N) bool -> logits (B,N) (padding = -1e9), value (B,)."""
        if mask is None:
            mask = torch.ones(job.shape[:2], dtype=torch.bool)
        x = torch.cat([job, glob.unsqueeze(1).expand(-1, job.shape[1], -1)], dim=-1)
        h = self.phi(x)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        logits = self.logit_head(torch.cat([h, pooled.unsqueeze(1).expand_as(h)], -1)).squeeze(-1)
        return logits.masked_fill(~mask, NEG), self.value_head(pooled).squeeze(-1)


def plackett_luce_logp(logits, mask, chosen):
    """Log-prob of ordered picks. chosen (B,K) long, -1 = no pick. Returns (B,)."""
    mask, total = mask.clone(), torch.zeros(logits.shape[0])
    for j in range(chosen.shape[1]):
        c = chosen[:, j]
        valid = c >= 0
        lp = torch.log_softmax(logits.masked_fill(~mask, NEG), dim=-1)
        idx = c.clamp(min=0)
        total = total + torch.where(valid, lp.gather(1, idx[:, None]).squeeze(1), torch.zeros(()))
        mask = mask & ~(torch.nn.functional.one_hot(idx, mask.shape[1]).bool() & valid[:, None])
    return total


def pad_batch(decisions):
    """Stack recorded decisions into padded tensors (job, glob, mask, chosen, old_logp, teacher)."""
    bs, n = len(decisions), max(len(d["job_feats"]) for d in decisions)
    k = max(len(d["chosen"]) for d in decisions)
    job = torch.zeros(bs, n, JOB_DIM)
    mask = torch.zeros(bs, n, dtype=torch.bool)
    chosen = torch.full((bs, k), -1, dtype=torch.long)
    teacher = torch.zeros(bs, n)
    has_t = torch.zeros(bs, dtype=torch.bool)
    for i, d in enumerate(decisions):
        m = len(d["job_feats"])
        job[i, :m] = torch.as_tensor(d["job_feats"], dtype=torch.float32)
        mask[i, :m] = True
        chosen[i, :len(d["chosen"])] = torch.as_tensor(d["chosen"], dtype=torch.long)
        if d.get("teacher") is not None:
            teacher[i, :m] = torch.as_tensor(d["teacher"], dtype=torch.float32)
            has_t[i] = True
    return dict(job=job, mask=mask, chosen=chosen, has_teacher=has_t, teacher=teacher,
                glob=torch.as_tensor(np.stack([d["global_feats"] for d in decisions]), dtype=torch.float32),
                old_logp=torch.as_tensor([d["log_prob"] for d in decisions], dtype=torch.float32))


def teacher_scores(teacher, state, eps=0.1):
    """Target distribution over the queue from the teacher's chosen batch.

    Simplification: ``teacher.select(state)`` yields a batch, not scores; we put (1-eps) mass uniformly
    on the batch members and eps uniformly on all jobs (smoothing), giving a valid distribution.
    """
    dec = teacher.select(state)
    pos = {j.id: i for i, j in enumerate(state.queue)}
    n = len(state.queue)
    ids = [i for i in dec.batch if i in pos]
    if not ids:
        return np.full(n, 1.0 / n)
    t = np.full(n, eps / n)
    for i in ids:
        t[pos[i]] += (1.0 - eps) / len(ids)
    return t / t.sum()


def batch_size_for(state):
    n = len(state.queue)
    sizes = [b for b in state.latency.batch_sizes if b <= min(state.max_batch, n)]
    return max(sizes) if sizes else min(state.latency.batch_sizes)


class LearnedPolicy(Policy):
    """Orders the queue with a SetPolicy; never purges. Modes: greedy, sample, teacher (imitate)."""
    uses_truth = False

    def __init__(self, model, mode="greedy", teacher=None, name="learned", teacher_eps=0.1):
        if mode not in ("greedy", "sample", "teacher"):
            raise ValueError("mode must be greedy|sample|teacher")
        if mode == "teacher" and teacher is None:
            raise ValueError("teacher mode needs a teacher")
        self.model, self.mode, self.teacher, self.name = model, mode, teacher, name
        self.teacher_eps, self.decisions = teacher_eps, []
        self.rng = np.random.default_rng(0)

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.decisions = []
        if self.teacher is not None:
            self.teacher.reset(spec, np.random.default_rng(int(rng.integers(2**31))))

    def observe(self, event, payload):
        if self.teacher is not None:
            self.teacher.observe(event, payload)

    def select(self, state):
        feats, g = job_features(state), global_features(state)
        n = len(state.queue)
        job = torch.as_tensor(feats, dtype=torch.float32)[None]
        glob = torch.as_tensor(g, dtype=torch.float32)[None]
        with torch.no_grad():
            logits, value = self.model(job, glob)
        target = teacher_scores(self.teacher, state, self.teacher_eps) if self.teacher is not None else None
        b = batch_size_for(state)
        lg = logits[0].numpy().astype(float)
        if self.mode == "teacher":
            order = list(np.argsort(-target, kind="stable")[:b])
        elif self.mode == "sample":
            order = list(np.argsort(-(lg + self.rng.gumbel(size=n)), kind="stable")[:b])
        else:
            order = list(np.argsort(-lg, kind="stable")[:b])
        ch = torch.as_tensor(order, dtype=torch.long)[None]
        with torch.no_grad():
            lp = float(plackett_luce_logp(logits, torch.ones(1, n, dtype=torch.bool), ch)[0])
        self.decisions.append(dict(job_feats=feats, global_feats=g, chosen=[int(i) for i in order],
                                   ids=[state.queue[i].id for i in order], log_prob=lp,
                                   value=float(value[0]), teacher=target, time=state.now))
        return Decision(batch=tuple(state.queue[i].id for i in order))


def save_model(model, path):
    torch.save(dict(cfg=model.cfg, state=model.state_dict()), path)


def load_model(path):
    blob = torch.load(path, weights_only=False)
    model = SetPolicy(**blob["cfg"])
    model.load_state_dict(blob["state"])
    return model
