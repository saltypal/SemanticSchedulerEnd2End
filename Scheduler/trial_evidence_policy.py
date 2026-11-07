"""Development policy for causal, deadline-aware receiver trials."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrialCandidate:
    token: str
    arrival_ms: float
    deadline_ms: float
    task_score: float
    signature: int
    expected_service_ms: float


def select_trial_candidate(
    ready: Sequence[TrialCandidate],
    positive_signatures: Sequence[int],
    now_ms: float,
    *,
    max_wait_ms: float,
    near_bits: int,
    receiver_conditioning: bool,
) -> TrialCandidate | None:
    """Choose from ready headers using signatures from completed positive reports.

    Both controls discount exact repeats. The conditional policy also discounts
    near repeats in proportion to similarity. The distance rule is a hypothesis
    for development, with no claim that image similarity establishes road identity.
    """
    if not ready:
        return None
    oldest = min(ready, key=lambda job: (job.arrival_ms, job.token))
    if now_ms - oldest.arrival_ms >= max_wait_ms:
        return oldest

    def priority(job):
        if now_ms + job.expected_service_ms > job.deadline_ms:
            return 0.0
        if job.signature in positive_signatures:
            return 0.0
        novelty = 1.0
        if receiver_conditioning and positive_signatures:
            nearest = min((job.signature ^ known).bit_count() for known in positive_signatures)
            novelty = min(1.0, nearest / (near_bits + 1))
        return job.task_score * novelty / job.expected_service_ms

    return min(ready, key=lambda job: (-priority(job), job.arrival_ms, job.token))
