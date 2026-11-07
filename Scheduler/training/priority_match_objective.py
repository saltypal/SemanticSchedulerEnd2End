"""Optional DQN objective for prescribed class-order scheduling.

Only the trainer/evaluator reads the class labels. The policy observation is
unchanged and does not contain them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from Evaluation.semantic_contracts import SemanticCandidateSnapshot, ServiceOutcome


@dataclass(frozen=True, slots=True)
class PriorityPairCounts:
    matches: int
    mismatches: int

    @property
    def total(self) -> int:
        return self.matches + self.mismatches

    @property
    def match_rate(self) -> float | None:
        return self.matches / self.total if self.total else None


def count_selected_priority_pairs(
    snapshot: SemanticCandidateSnapshot,
    action_index: int,
    priority_by_job: Mapping[str, int],
) -> PriorityPairCounts:
    """Compare a selected job with other eligible, different-class jobs."""
    if action_index not in snapshot.eligible_indices:
        raise ValueError("Action is not eligible")
    chosen = snapshot.jobs[action_index]
    chosen_priority = priority_by_job[chosen.job_id]
    matches = mismatches = 0
    for index in snapshot.eligible_indices:
        if index == action_index:
            continue
        other_priority = priority_by_job[snapshot.jobs[index].job_id]
        if chosen_priority < other_priority:
            matches += 1
        elif chosen_priority > other_priority:
            mismatches += 1
    return PriorityPairCounts(matches, mismatches)


def class_order_reward(
    counts: PriorityPairCounts,
    outcome: ServiceOutcome,
    selected_priority: int,
    expired_priorities: tuple[int, ...] = (),
) -> float:
    """Give clear feedback on ordering while retaining delivery consequences.

    One comparable decision contributes in [-1, 1] regardless of queue size.
    A successful on-time delivery adds up to 0.25. Expiry/failure penalties
    stop an agent from optimizing order while neglecting delivery.
    """
    if selected_priority not in (0, 1, 2, 3):
        raise ValueError("Priority must be an integer class from 0 to 3")
    pair_term = ((counts.matches - counts.mismatches) / counts.total
                 if counts.total else 0.0)
    selected_weight = (4 - selected_priority) / 4.0
    delivery_term = (0.25 * selected_weight
                     if outcome.status == "completed_on_time" else 0.0)
    failure_term = (0.25 * selected_weight
                    if outcome.status in {"failed_transport", "horizon_censored"} else 0.0)
    expiry_term = 0.25 * sum((4 - priority) / 4.0 for priority in expired_priorities)
    return pair_term + delivery_term - failure_term - expiry_term


def episode_priority_match_rate(result: Mapping[str, object]) -> float:
    """Evaluate a full rollout using decision traces and hidden class facts."""
    priority = {str(row["job_id"]): int(row["base_priority"])
                for row in result["job_facts"]}
    matches = mismatches = 0
    for decision in result["decision_trace"]:
        chosen_id = str(decision["job_id"])
        chosen_priority = priority[chosen_id]
        for candidate in decision["candidate_snapshot"]:
            other_id = str(candidate["job_id"])
            if not candidate["eligible"] or other_id == chosen_id:
                continue
            other_priority = priority[other_id]
            matches += int(chosen_priority < other_priority)
            mismatches += int(chosen_priority > other_priority)
    total = matches + mismatches
    return matches / total if total else 0.0
