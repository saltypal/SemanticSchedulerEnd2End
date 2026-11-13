"""BCEdge-style learned adaptive batching (arXiv 2305.01519), header-only.

BCEdge picks the batch size each scheduling epoch with a maximum-entropy RL agent and orders requests by SLO.
Adapted here: an actor-critic (entropy-regularised, trained in ``Scheduler.training.edge.bcedge_train``) chooses
b in {1, 2, 4, 8} from a compact queue summary; the batch is the first b jobs by predicted tier (critical
first) then earliest soft deadline (D_pred - age), i.e. EDF within the tier. Uses headers only (tier_probs,
age, spec); ``uses_truth`` is False. Untrained (``net=None``) it takes the largest feasible size.
"""
import numpy as np

from Evaluation.edge.contracts import Decision, Policy, top_batch
from Scheduler.edge.classic import d_pred, fifo_key, pred_tier
from Scheduler.training.edge.state_features import global_features

ACTIONS = (1, 2, 4, 8)
FEAT_DIM = 4 + 4 + 2


def state_vector(state):
    """(FEAT_DIM,) summary: global feats, predicted-tier fractions, min slack ratio, mean log age."""
    q = state.queue
    frac, slack = np.zeros(4), 1.0
    for j in q:
        k = pred_tier(j)
        frac[k] += 1.0 / len(q)
        slack = min(slack, max(0.0, (state.spec.hard_deadline(k) - j.age) / state.spec.hard_deadline(k)))
    mean_age = float(np.mean([np.log1p(j.age) for j in q])) / 3.0 if q else 0.0
    return np.concatenate([global_features(state), frac, [slack, mean_age]]).astype(np.float32)


def action_mask(state):
    """Boolean mask over ACTIONS: sizes up to min(max_batch, queue length); index 0 is always allowed."""
    cap = max(1, min(state.max_batch, len(state.queue)))
    return np.array([a <= cap for a in ACTIONS])


def decode_size(state, action):
    """Chosen action -> usable batch size (largest profiled size <= action, <= max_batch, <= queue)."""
    cap = min(ACTIONS[action], state.max_batch, len(state.queue))
    ok = [b for b in state.latency.batch_sizes if b <= cap]
    return max(ok) if ok else cap


def make_net(hidden=32):
    from torch import nn

    class BCEdgeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = dict(hidden=hidden)
            self.body = nn.Sequential(nn.Linear(FEAT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.actor, self.critic = nn.Linear(hidden, len(ACTIONS)), nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            return self.actor(h), self.critic(h).squeeze(-1)

    return BCEdgeNet()


def save_net(net, path):
    import torch
    torch.save(dict(cfg=net.cfg, state=net.state_dict()), path)


def load_net(path):
    import torch
    blob = torch.load(path, weights_only=False)
    net = make_net(**blob["cfg"])
    net.load_state_dict(blob["state"])
    return net.eval()


class BCEdge(Policy):
    """Greedy (or sampled, when ``sample``) batch-size actor + tier-then-EDF ordering. Records when ``record``."""
    name = "bcedge"

    def __init__(self, net=None, sample=False, record=False, seed=0):
        self.net, self.sample, self.record = net, sample, record
        self._rng, self.records = np.random.default_rng(seed), []

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.records = []

    def pick(self, state):
        mask = action_mask(state)
        x = state_vector(state)
        if self.net is None:
            return int(np.flatnonzero(mask)[-1]), x, mask
        import torch
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(x)[None])
        logits = logits[0].numpy().astype(float)
        logits[~mask] = -1e9
        if self.sample:
            p = np.exp(logits - logits.max())
            return int(self._rng.choice(len(ACTIONS), p=p / p.sum())), x, mask
        return int(np.argmax(logits)), x, mask

    def select(self, state):
        if not state.queue:
            return Decision()
        action, x, mask = self.pick(state)
        order = [j.id for j in sorted(sorted(state.queue, key=fifo_key),
                                      key=lambda j: (pred_tier(j), d_pred(j, state) - j.age))]
        batch = top_batch(order, state, decode_size(state, action))
        if self.record:
            self.records.append((x, mask, action, batch))
        return Decision(batch)


POLICIES = {"bcedge": BCEdge}
