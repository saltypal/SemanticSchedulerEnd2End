"""Framework-free conversion of semantic snapshots to fixed DQN inputs."""

from __future__ import annotations

import numpy as np

from Evaluation.scheduling.semantic_contracts import ResourceBudgetSemantic, SemanticCandidateSnapshot


MAX_CANDIDATES = 25
CANDIDATE_DIM = 22
GLOBAL_DIM = 9
STATE_DIM = MAX_CANDIDATES * CANDIDATE_DIM + GLOBAL_DIM


def model_state(snapshot: SemanticCandidateSnapshot, budget: ResourceBudgetSemantic,
                *, max_candidates: int = MAX_CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """Return a 559-float observation and 25-element valid-action mask.

    Dataset labels, evidence groups, outcomes and future arrivals are absent
    by construction. Candidate order is the simulator's arrival/ID order.
    """
    if max_candidates != MAX_CANDIDATES:
        raise ValueError("The exported DQN schema requires exactly 25 slots")
    if len(snapshot.jobs) > MAX_CANDIDATES:
        raise ValueError("Queue exceeds the DQN candidate cap of 25")
    ue_ids = tuple(ue for ue, _ in snapshot.per_ue_served)
    if len(ue_ids) > 5:
        raise ValueError("The DQN schema supports at most five UEs")
    ue_slot = {ue: index for index, ue in enumerate(ue_ids)}
    candidates = np.zeros((MAX_CANDIDATES, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
    for index, (job, eligible) in enumerate(zip(snapshot.jobs, snapshot.eligible_mask)):
        features = job.features
        normalized = features.normalized_values(cold_start_novelty=1.0)
        availability = tuple(float(value) for value in features.availability_mask())
        age = min(1.0, max(0.0, snapshot.now_ms - job.arrival_ms) / 500.0)
        slack = 1.0 if job.deadline_ms is None else float(np.clip(
            (job.deadline_ms - snapshot.now_ms) / 500.0, -1.0, 1.0))
        cost = min(1.0, job.resource_units /
                   max(1.0, budget.service_rate_units_per_ms * 200.0))
        one_hot = np.zeros(5, dtype=np.float32)
        if job.ue_id not in ue_slot:
            raise ValueError(f"Job references unregistered UE {job.ue_id}")
        one_hot[ue_slot[job.ue_id]] = 1.0
        candidates[index] = np.asarray((
            *normalized, *availability, age, slack, cost, *one_hot), dtype=np.float32)
        mask[index] = eligible
    served = [count for _, count in snapshot.per_ue_served]
    total_served = max(1, sum(served))
    served_fraction = [count / total_served for count in served]
    served_fraction.extend([0.0] * (5 - len(served_fraction)))
    global_values = np.asarray((
        min(1.0, snapshot.now_ms / 2000.0),
        len(snapshot.jobs) / MAX_CANDIDATES,
        min(1.0, snapshot.delivered_count / MAX_CANDIDATES),
        max(0.0, snapshot.queue_capacity - len(snapshot.jobs)) / snapshot.queue_capacity,
        *served_fraction,
    ), dtype=np.float32)
    state = np.concatenate((candidates.reshape(-1), global_values))
    if state.shape != (STATE_DIM,) or mask.shape != (MAX_CANDIDATES,):
        raise AssertionError("DQN state layout changed unexpectedly")
    return state, mask


def static_feature_row(job) -> np.ndarray:
    """Seven values and seven availability bits for supervised policies."""
    return np.asarray((*job.features.normalized_values(cold_start_novelty=1.0),
                       *job.features.availability_mask()), dtype=np.float32)
