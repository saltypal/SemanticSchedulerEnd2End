"""Per-decision reward attribution and decision logging for learned policies.

``LoggingPolicy`` wraps any policy and records, for every ``select`` call, the encoded state
(job features, global features, queue ids) and the chosen batch. After the run,
``attribute_rewards`` maps the ordered decisions and the ``RunResult`` rows to one reward per
decision: sum of ``value_fn(row)`` over the decoded jobs of that decision minus sum_k lambda_k for
true-tier-k decoded jobs that finished at or after their hard deadline H_k.
Reuses: Evaluation.edge.contracts, Scheduler.training.edge.state_features.
"""
from dataclasses import dataclass

import numpy as np

from Evaluation.edge.contracts import Policy
from Scheduler.training.edge.state_features import global_features, job_features


@dataclass
class DecisionRecord:
    ids: tuple          # queue ids in state order
    job_x: np.ndarray   # (N, JOB_DIM)
    glob: np.ndarray    # (GLOBAL_DIM,)
    batch: tuple        # chosen job ids
    idx: tuple          # positions of ``batch`` inside ``ids``
    teacher_idx: tuple = ()


def make_record(state, decision, teacher_batch=()):
    ids = tuple(j.id for j in state.queue)
    pos = {i: p for p, i in enumerate(ids)}
    return DecisionRecord(ids, job_features(state), global_features(state), tuple(decision.batch),
                          tuple(pos[i] for i in decision.batch), tuple(pos[i] for i in teacher_batch if i in pos))


class LoggingPolicy(Policy):
    """Delegates to ``inner`` and keeps ``records`` (one DecisionRecord per non-idle decision)."""

    def __init__(self, inner):
        self.inner, self.records = inner, []
        self.name, self.uses_truth = "log_" + inner.name, inner.uses_truth

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.records = []
        self.inner.reset(spec, rng)

    def select(self, state):
        dec = self.inner.select(state)
        if dec.batch:
            self.records.append(make_record(state, dec))
        return dec

    def observe(self, event, payload):
        self.inner.observe(event, payload)


def attribute_rewards(batches, rows, spec, lambdas=(0.0, 0.0, 0.0, 0.0), value_fn=None):
    """Reward per decision. ``batches``: list of id tuples in decision order; ``rows``: outcome dicts.

    ``value_fn(row) -> float`` replaces the true realised value (e.g. a learned reward model).
    """
    if len(lambdas) != 4:
        raise ValueError("need one lambda per tier")
    by_id = {r["id"]: r for r in rows}
    out = []
    for batch in batches:
        total = 0.0
        for i in batch:
            r = by_id.get(i)
            if r is None or r.get("status") != "decoded":
                continue
            total += float(value_fn(r) if value_fn else r["value"])
            k = int(r["tier"])
            delay = r.get("delay")
            if delay is not None and delay >= spec.hard_deadline(k):
                total -= float(lambdas[k])
        out.append(total)
    return np.asarray(out, dtype=float)
