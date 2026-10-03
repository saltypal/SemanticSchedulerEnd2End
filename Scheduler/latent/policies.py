"""Policies for whole latent jobs. No device allocation or queue mutation."""

from __future__ import annotations

from dataclasses import dataclass

from INFRA.latent.contracts import LatentCandidate, LatentSnapshot


POLICIES = ("fifo", "sjf", "edf", "priority_fifo", "priority_edf", "semantic")


@dataclass(frozen=True)
class LatentPolicy:
    name: str = "semantic"

    def __post_init__(self):
        if self.name not in POLICIES:
            raise ValueError(f"Unknown latent policy: {self.name}")

    def select(self, snapshot: LatentSnapshot) -> tuple[str, float | None, str]:
        eligible = [job for job in snapshot.jobs if job.eligible]
        if not eligible:
            raise ValueError("No eligible latent jobs")
        if self.name.startswith("priority_") or self.name == "semantic":
            if any(job.priority_class is None for job in eligible):
                raise ValueError("Priority policies require explicit or predicted priority for every job")
            highest = min(job.priority_class for job in eligible)
            eligible = [job for job in eligible if job.priority_class == highest]

        def key(job: LatentCandidate):
            fifo = (job.input_index, job.job_id)
            deadline = float("inf") if job.deadline_ms is None else job.deadline_ms
            if self.name in {"fifo", "priority_fifo"}:
                return fifo
            if self.name == "sjf":
                return job.service_ms, *fifo
            if self.name in {"edf", "priority_edf"}:
                return deadline, *fifo
            # Separate observable scalars remain in the trace. This simple
            # heuristic is an explicitly declared proxy, not measured VoI.
            score = semantic_score(job)
            return -score, deadline, *fifo

        chosen = min(eligible, key=key)
        score = semantic_score(chosen) if self.name == "semantic" else None
        return chosen.job_id, score, (
            "strict supplied/predicted priority, then semantic proxy per service ms"
            if self.name == "semantic" else self.name)


def semantic_score(job: LatentCandidate) -> float:
    relevance = 0.5 if job.context_relevance is None else (job.context_relevance + 1) / 2
    novelty = 1.0 if job.novelty is None else job.novelty
    importance = 1.0 if job.task_importance is None else job.task_importance / 3
    # Missing is neutral for selection but remains None in every trace.
    # Nonzero floors prevent an uncertain proxy from suppressing a priority class.
    relevance_factor = 0.25 + 0.75 * relevance
    novelty_factor = 0.25 + 0.75 * novelty
    importance_factor = 0.25 + 0.75 * importance
    urgency = 1.0
    if job.slack_ms is not None:
        urgency += job.service_ms / max(job.service_ms, job.slack_ms + job.service_ms)
    return relevance_factor * novelty_factor * importance_factor * urgency / job.service_ms
