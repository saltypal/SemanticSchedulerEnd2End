"""Learning to rank for queue ordering (plan v3 S8) and the SCOPE-LTR listwise distillation loss.

Dataset: one query group per decision epoch; features per queued job = job features + global
features; relevance = realised value of the job when it was decoded (0 if purged/expired/never),
discretised to gains 0..4 as round(4 * value / vmax).
Models: LambdaMART via lightgbm.LGBMRanker when installed, otherwise a pointwise sklearn
GradientBoostingRegressor on the gain (documented fallback); RankNet (pairwise logistic, torch).
Every model exposes ``score(X) -> (N,)`` and ``save(path)`` / ``load(path)``.
Reuses: reward_attribution.DecisionRecord (episodes come from LoggingPolicy / DQNPolicy records).
"""
import pickle

import numpy as np

from Evaluation.edge.contracts import Decision, Policy
from Scheduler.training.edge.state_features import GLOBAL_DIM, JOB_DIM, global_features, job_features

N_GAINS = 4


def discretise(values, vmax):
    """Gain 0..4 from realised value."""
    if vmax <= 0:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.rint(N_GAINS * np.asarray(values, dtype=float) / vmax), 0, N_GAINS).astype(int)


def _x(job_x, glob):
    return np.hstack([job_x, np.tile(glob, (job_x.shape[0], 1))])


def build_ranking_dataset(episodes, vmax=None):
    """``episodes``: list of (records, rows). Returns dict X (M,F), y (M,), group (G,) sizes, vmax."""
    values = [r["value"] for _, rows in episodes for r in rows if r.get("status") == "decoded"]
    vmax = float(vmax if vmax is not None else (max(values) if values else 1.0))
    xs, ys, groups = [], [], []
    for records, rows in episodes:
        val = {r["id"]: float(r["value"]) for r in rows if r.get("status") == "decoded"}
        for rec in records:
            if not rec.ids:
                continue
            xs.append(_x(rec.job_x, rec.glob))
            ys.append(discretise([val.get(i, 0.0) for i in rec.ids], vmax))
            groups.append(len(rec.ids))
    dim = JOB_DIM + GLOBAL_DIM
    X = np.vstack(xs) if xs else np.zeros((0, dim))
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    return {"X": X, "y": y, "group": np.asarray(groups, dtype=int), "vmax": vmax}


class TreeRanker:
    """LambdaMART (lightgbm) or pointwise GradientBoosting fallback."""

    def __init__(self, model, backend):
        self.model, self.backend = model, backend

    def score(self, X):
        return np.asarray(self.model.predict(X), dtype=float)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "backend": self.backend}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:   # only load files written by save()
            d = pickle.load(f)
        return cls(d["model"], d["backend"])


def train_lambdamart(data, n_estimators=100, learning_rate=0.1, seed=0, prefer_lightgbm=True):
    """Fit a tree ranker on ``build_ranking_dataset`` output."""
    if len(data["y"]) == 0:
        raise ValueError("empty ranking dataset")
    if prefer_lightgbm:
        try:
            import lightgbm
            m = lightgbm.LGBMRanker(n_estimators=n_estimators, learning_rate=learning_rate,
                                    random_state=seed, verbose=-1, min_child_samples=5)
            m.fit(data["X"], data["y"], group=data["group"])
            return TreeRanker(m, "lightgbm")
        except ImportError:
            pass
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed)
    m.fit(data["X"], data["y"].astype(float))
    return TreeRanker(m, "sklearn_pointwise")


class RankNet:
    """Scalar-score MLP trained with pairwise logistic loss."""

    def __init__(self, in_dim, hidden=32):
        from torch import nn
        self.in_dim, self.hidden = in_dim, hidden
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def score(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).squeeze(-1).numpy()

    def save(self, path):
        import torch
        torch.save({"in_dim": self.in_dim, "hidden": self.hidden, "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, map_location="cpu")
        m = cls(d["in_dim"], d["hidden"])
        m.net.load_state_dict(d["state"])
        return m


def _pairs(y, group):
    hi, lo, start = [], [], 0
    for g in group:
        yy = y[start:start + g]
        for a in range(g):
            for b in range(g):
                if yy[a] > yy[b]:
                    hi.append(start + a)
                    lo.append(start + b)
        start += g
    return np.asarray(hi, dtype=int), np.asarray(lo, dtype=int)


def train_ranknet(data, hidden=32, epochs=200, lr=1e-2, seed=0):
    """Pairwise logistic loss -log sigmoid(s_hi - s_lo) over within-group pairs of different gain."""
    import torch
    torch.manual_seed(seed)
    hi, lo = _pairs(data["y"], data["group"])
    if len(hi) == 0:
        raise ValueError("no ordered pairs in dataset")
    X = torch.as_tensor(data["X"], dtype=torch.float32)
    model = RankNet(X.shape[1], hidden)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    for _ in range(epochs):
        s = model.net(X).squeeze(-1)
        loss = torch.nn.functional.softplus(-(s[hi] - s[lo])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


class LTRPolicy(Policy):
    """Order the queue by ranker score and decode the top batch."""
    name = "ltr"

    def __init__(self, ranker):
        self.ranker = ranker

    def select(self, state):
        if not state.queue:
            return Decision()
        s = self.ranker.score(_x(job_features(state), global_features(state)))
        ids = [j.id for j in state.queue]
        from Scheduler.edge.batching import rank_alive_first, ratio_batch_size
        order = rank_alive_first(s, state)
        return Decision(tuple(ids[i] for i in order[:ratio_batch_size(state, order)]))   # F2: was max B


def listwise_distill_loss(scores, teacher_scores, lambdas_by_tier, tier_probs):
    """ListNet cross-entropy with Lagrangian tier reweighting (torch scalar).

    Teacher target p = softmax(teacher_scores) is reweighted per job by
    w_i = 1 + sum_k tier_probs[i,k] * lambda_k and renormalised; loss = -sum_i q_i log softmax(scores)_i.
    """
    import torch
    scores = torch.as_tensor(scores, dtype=torch.float32)
    teacher = torch.as_tensor(teacher_scores, dtype=torch.float32)
    w = 1.0 + torch.as_tensor(tier_probs, dtype=torch.float32) @ torch.as_tensor(lambdas_by_tier, dtype=torch.float32)
    q = torch.softmax(teacher, dim=0) * w
    q = q / q.sum()
    return -(q * torch.log_softmax(scores, dim=0)).sum()
