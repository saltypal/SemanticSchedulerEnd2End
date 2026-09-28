"""Task-scoped receiver knowledge used only after successful delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class DeliveredKnowledgeSnapshot:
    task_profile_id: str
    history_version: int
    image_ids: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...]
    best_sufficiency_by_group: Mapping[str, int]
    delivery_times_ms: tuple[float, ...]

    @property
    def successful_delivery_count(self) -> int:
        return len(self.image_ids)


class DeliveredKnowledgeState:
    """Mutable owner of delivered history; policies receive snapshots only."""

    def __init__(self, task_profile_id: str) -> None:
        if not task_profile_id:
            raise ValueError("Delivered history requires a task_profile_id.")
        self.task_profile_id = task_profile_id
        self._history_version = 0
        self._image_ids: list[str] = []
        self._embeddings: list[tuple[float, ...]] = []
        self._delivery_times_ms: list[float] = []
        self._best_sufficiency_by_group: dict[str, int] = {}
        self._recorded_ids: set[str] = set()

    def snapshot(self) -> DeliveredKnowledgeSnapshot:
        return DeliveredKnowledgeSnapshot(
            task_profile_id=self.task_profile_id,
            history_version=self._history_version,
            image_ids=tuple(self._image_ids),
            embeddings=tuple(self._embeddings),
            best_sufficiency_by_group=dict(self._best_sufficiency_by_group),
            delivery_times_ms=tuple(self._delivery_times_ms),
        )

    def record_success(
        self,
        *,
        image_id: str,
        image_embedding: tuple[float, ...],
        delivered_at_ms: float,
        evidence_group_id: str | None = None,
        evidence_sufficiency: int | None = None,
    ) -> bool:
        """Record exactly one successful completion and return whether state changed.

        Callers must only invoke this after the service/channel outcome is known
        to be successful.  A duplicate image id is deliberately idempotent so a
        retry in the event loop cannot make a job artificially less novel.
        """

        if image_id in self._recorded_ids:
            return False
        values = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Delivered image embedding must be a finite non-empty vector.")
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            raise ValueError("Delivered image embedding must have a non-zero norm.")
        if evidence_sufficiency is not None and evidence_sufficiency not in (0, 1):
            raise ValueError("evidence_sufficiency must be 0, 1, or None.")
        self._recorded_ids.add(image_id)
        self._image_ids.append(image_id)
        self._embeddings.append(tuple(float(value) for value in values / norm))
        self._delivery_times_ms.append(float(delivered_at_ms))
        if evidence_group_id is not None and evidence_sufficiency is not None:
            previous = self._best_sufficiency_by_group.get(evidence_group_id, 0)
            self._best_sufficiency_by_group[evidence_group_id] = max(previous, evidence_sufficiency)
        self._history_version += 1
        return True


def novelty_from_history(
    *,
    image_embedding: tuple[float, ...],
    history: DeliveredKnowledgeSnapshot,
    feature_name: str = "delivered_history_novelty",
):
    """Return P3 as ``(1 - max cosine) / 2`` or an explicit unavailable record."""

    from SemParam.scheduler_features import FeatureValue

    values = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if values.size == 0 or norm == 0.0:
        raise ValueError("Novelty requires a non-empty, non-zero image embedding.")
    normalized = values / norm
    if not history.embeddings:
        return FeatureValue.unavailable(
            feature_name,
            method="one_minus_max_history_cosine",
            source="receiver_delivered_history",
            reason="empty_delivered_history",
            history_version=history.history_version,
            history_count=history.successful_delivery_count,
        )
    prior = np.asarray(history.embeddings, dtype=np.float32)
    if prior.ndim != 2 or prior.shape[1] != normalized.size:
        raise ValueError("Delivered history embedding dimensions do not match the candidate image embedding.")
    maximum = float(np.clip(np.max(prior @ normalized), -1.0, 1.0))
    novelty = float(np.clip((1.0 - maximum) / 2.0, 0.0, 1.0))
    return FeatureValue.measured(
        feature_name,
        novelty,
        method="one_minus_max_history_cosine",
        source="receiver_delivered_history",
        max_cosine_similarity=maximum,
        history_version=history.history_version,
        history_count=history.successful_delivery_count,
    )
