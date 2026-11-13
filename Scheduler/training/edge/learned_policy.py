"""Evaluate trained learners in sweeps: policy specs ``learned:<algo>:<checkpoint>``.

``<algo>`` is one of the ``train_cli --algo`` names and ``<checkpoint>`` the file that CLI wrote (``--out``).
Each spec builds a header-only ``Policy`` in greedy (inference) mode: no sampling, no teacher, no purges.
``uses_truth`` stays False and ``build_learned`` takes no truth argument, so the sweep runner never hands
a learner the Truth table. The ``State`` seen in ``select`` holds ``QueuedJob`` headers only.

Checkpoint formats (written by train_cli):
  ppo, scope_ppo, bc, dagger   policy_net.save_model -> SetPolicy (dict cfg/state)
  dqn, scope_dqn               torch dict (hidden, state) -> dqn.make_qnet, DQNPolicy
  semcover                     semcover_train.save_net -> SemCoverNet (cfg/state) -> semcover.SemCover(net=...)
  bcedge                       bcedge.save_net -> BCEdge(net=...): greedy batch size, tier-then-EDF order
  ltr, scope_ltr               LambdaMART pickle (TreeRanker) or RankNet torch file -> ltr.LTRPolicy
"""
from pathlib import Path

import numpy as np
import torch

from Evaluation.edge.contracts import Decision, Policy
from Scheduler.training.edge.dqn import DQNPolicy, make_qnet
from Scheduler.training.edge.ltr import LTRPolicy, RankNet, TreeRanker
from Scheduler.edge.batching import rank_alive_first, ratio_batch_size
from Scheduler.training.edge.policy_net import load_model
from Scheduler.training.edge.state_features import global_features, job_features

PREFIX = "learned:"
SET_ALGOS = ("ppo", "scope_ppo", "bc", "dagger")
DQN_ALGOS = ("dqn", "scope_dqn")
LTR_ALGOS = ("ltr", "scope_ltr")
SEMCOVER_ALGOS = ("semcover",)
BCEDGE_ALGOS = ("bcedge",)
ALGOS = SET_ALGOS + DQN_ALGOS + LTR_ALGOS + SEMCOVER_ALGOS + BCEDGE_ALGOS


def is_learned(name):
    return isinstance(name, str) and name.startswith(PREFIX)


def parse_spec(spec):
    """``learned:<algo>:<checkpoint>`` -> (algo, checkpoint path). The checkpoint must exist.

    Only the first colon after the algo separates it from the path, so Windows drive letters are fine.
    """
    if not is_learned(spec):
        raise ValueError("not a learned policy spec: %r" % (spec,))
    algo, sep, path = spec[len(PREFIX):].partition(":")
    if algo not in ALGOS:
        raise ValueError("unknown learned algo %r in %r; known: %s" % (algo, spec, list(ALGOS)))
    if not sep or not path:
        raise ValueError("expected learned:<algo>:<checkpoint>, got %r" % (spec,))
    if not Path(path).is_file():
        raise ValueError("checkpoint for %r not found: %s" % (spec, path))
    return algo, path


class SetGreedyPolicy(Policy):
    """Top-B by SetPolicy logit (greedy), with no teacher, no purges and no recording."""
    uses_truth = False

    def __init__(self, model, name="learned"):
        self.model, self.name = model.eval(), name

    def select(self, state):
        if not state.queue:
            return Decision()
        job = torch.as_tensor(job_features(state), dtype=torch.float32)[None]
        glob = torch.as_tensor(global_features(state), dtype=torch.float32)[None]
        with torch.no_grad():
            logits, _ = self.model(job, glob)
        lg = logits[0].numpy().astype(float)
        self.last_scores = {j.id: float(v) for j, v in zip(state.queue, lg)}   # trace only
        full = rank_alive_first(lg, state)
        order = full[:ratio_batch_size(state, full)]          # F2: shared batch rule, was max B
        return Decision(tuple(state.queue[int(i)].id for i in order))


def _load_ranker(path):
    """LambdaMART checkpoints are pickles; RankNet checkpoints are torch zip files (start with 'PK')."""
    with open(path, "rb") as handle:
        head = handle.read(2)
    return RankNet.load(path) if head == b"PK" else TreeRanker.load(path)


def build_learned(spec):
    """Build the greedy header-only policy for a ``learned:`` spec (no Truth argument by design)."""
    algo, path = parse_spec(spec)
    if algo in DQN_ALGOS:
        blob = torch.load(path, weights_only=False)
        model = make_qnet(int(blob["hidden"]))
        model.load_state_dict(blob["state"])
        model.eval()
        return DQNPolicy(model, epsilon=0.0)
    if algo in SEMCOVER_ALGOS:
        from Scheduler.edge.semcover import SemCover
        from Scheduler.training.edge.semcover_train import load_net
        return SemCover(net=load_net(path))
    if algo in BCEDGE_ALGOS:
        from Scheduler.edge.bcedge import BCEdge, load_net
        return BCEdge(net=load_net(path))
    if algo in LTR_ALGOS:
        return LTRPolicy(_load_ranker(path))
    return SetGreedyPolicy(load_model(path), name=algo)
