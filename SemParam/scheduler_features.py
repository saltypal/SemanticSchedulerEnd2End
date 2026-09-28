"""Track-A scheduling feature records and lazy image extraction.

This module deliberately depends on an image embedding backend only.  It does
not import a communication codec or infer a dataset label from an image.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from SemParam.clip_backbone import CLIPBackbone, SemanticEmbeddingBackend, _normalize
from SemParam.delivered_history import DeliveredKnowledgeSnapshot, novelty_from_history


SCHEDULER_FEATURE_SCHEMA_ID = 0x0201
SCHEDULER_FEATURE_VERSION = "track-a/v2"
TASK_PROFILE_ID = "post_disaster_access_and_damage_triage_v1"
TASK_TEXT = (
    "Prioritize imagery that most improves assessment of immediate threats "
    "to people, access routes, and affected structures."
)


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One scheduling feature with enough evidence to audit its origin."""

    name: str
    value: float | None
    available: bool
    method: str
    source: str
    version: str
    latency_ms: float | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.available != (self.value is not None):
            raise ValueError("Available feature values must have a numeric value, and unavailable values must not.")

    @classmethod
    def unavailable(
        cls,
        name: str,
        *,
        method: str,
        source: str,
        version: str = SCHEDULER_FEATURE_VERSION,
        latency_ms: float | None = None,
        **provenance: Any,
    ) -> "FeatureValue":
        return cls(name, None, False, method, source, version, latency_ms, dict(provenance))

    @classmethod
    def measured(
        cls,
        name: str,
        value: float,
        *,
        method: str,
        source: str,
        version: str = SCHEDULER_FEATURE_VERSION,
        latency_ms: float | None = None,
        **provenance: Any,
    ) -> "FeatureValue":
        return cls(name, float(value), True, method, source, version, latency_ms, dict(provenance))


@dataclass(frozen=True, slots=True)
class SchedulerAnnotation:
    """Human-reviewed fields used by the oracle-assisted Track-A prototype."""

    task_impact: int
    urgency: int
    evidence_sufficiency: int
    annotation_source: str = "reviewed_task_rubric"
    reviewer: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if self.task_impact not in range(4):
            raise ValueError("task_impact must be an integer in [0, 3].")
        if self.urgency not in range(4):
            raise ValueError("urgency must be an integer in [0, 3].")
        if self.evidence_sufficiency not in (0, 1):
            raise ValueError("evidence_sufficiency must be 0 or 1.")


@dataclass(frozen=True, slots=True)
class SchedulingFeatureVectorV2:
    """Immutable static image record with decision-time novelty refresh support."""

    image_id: str
    task_profile_id: str
    image_embedding: tuple[float, ...]
    context_relevance: FeatureValue
    delivered_history_novelty: FeatureValue
    task_impact: FeatureValue
    urgency: FeatureValue
    evidence_sufficiency: FeatureValue
    extraction_latency_ms: float
    extractor_version: str = SCHEDULER_FEATURE_VERSION
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("Scheduling features require a stable image_id.")
        if not self.task_profile_id:
            raise ValueError("Scheduling features require a task_profile_id.")
        if not self.image_embedding:
            raise ValueError("Scheduling features require a non-empty image embedding.")
        expected = {
            "context_relevance": self.context_relevance,
            "delivered_history_novelty": self.delivered_history_novelty,
            "task_impact": self.task_impact,
            "urgency": self.urgency,
            "evidence_sufficiency": self.evidence_sufficiency,
        }
        for name, value in expected.items():
            if value.name != name:
                raise ValueError(f"Feature {name!r} has mismatched record name {value.name!r}.")

    @property
    def values(self) -> tuple[FeatureValue, FeatureValue, FeatureValue, FeatureValue, FeatureValue]:
        return (
            self.context_relevance,
            self.delivered_history_novelty,
            self.task_impact,
            self.urgency,
            self.evidence_sufficiency,
        )

    def normalized_values(self, *, cold_start_novelty: float = 1.0) -> tuple[float, float, float, float, float]:
        """Return model inputs only; availability remains available through ``availability_mask``."""

        relevance = 0.0 if self.context_relevance.value is None else np.clip((self.context_relevance.value + 1.0) / 2.0, 0.0, 1.0)
        novelty = cold_start_novelty if self.delivered_history_novelty.value is None else self.delivered_history_novelty.value
        impact = 0.0 if self.task_impact.value is None else self.task_impact.value / 3.0
        urgency = 0.0 if self.urgency.value is None else self.urgency.value / 3.0
        sufficiency = 0.0 if self.evidence_sufficiency.value is None else self.evidence_sufficiency.value
        return tuple(float(np.clip(value, 0.0, 1.0)) for value in (relevance, novelty, impact, urgency, sufficiency))

    def availability_mask(self) -> tuple[int, int, int, int, int]:
        return tuple(int(value.available) for value in self.values)

    def with_delivered_history(self, history: DeliveredKnowledgeSnapshot) -> "SchedulingFeatureVectorV2":
        """Recalculate only P3 against an immutable receiver history snapshot."""

        novelty = novelty_from_history(
            image_embedding=self.image_embedding,
            history=history,
            feature_name="delivered_history_novelty",
        )
        return replace(self, delivered_history_novelty=novelty)


class SchedulerFeatureExtractor:
    """Extract cached Track-A image features without loading a codec checkpoint."""

    def __init__(
        self,
        backend: SemanticEmbeddingBackend | None = None,
        *,
        task_profile_id: str = TASK_PROFILE_ID,
        task_text: str = TASK_TEXT,
        version: str = SCHEDULER_FEATURE_VERSION,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.backend = backend or CLIPBackbone(
            model_name=model_name, device=device, local_files_only=local_files_only
        )
        self.task_profile_id = task_profile_id
        self.task_text = task_text
        self.version = version

    def extract(
        self,
        image_path: str | Path,
        *,
        image_id: str,
        annotation: SchedulerAnnotation,
        metadata: Mapping[str, Any] | None = None,
    ) -> SchedulingFeatureVectorV2:
        """Create static P2/P5/P6/P7 values and leave P3 unavailable until scheduling."""

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Scheduler feature image does not exist: {path}")
        started = perf_counter_ns()
        image_embedding = _normalize(self.backend.encode_image(path))
        task_embedding = _normalize(self.backend.encode_texts([self.task_text])[self.task_text])
        relevance = float(np.clip(np.dot(image_embedding, task_embedding), -1.0, 1.0))
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
        annotation_provenance = {
            "annotation_source": annotation.annotation_source,
            "reviewer": annotation.reviewer,
            "rationale": annotation.rationale,
        }
        return SchedulingFeatureVectorV2(
            image_id=image_id,
            task_profile_id=self.task_profile_id,
            image_embedding=tuple(float(value) for value in image_embedding),
            context_relevance=FeatureValue.measured(
                "context_relevance", relevance, method="clip_cosine_similarity",
                source="openai/clip-vit-base-patch32", version=self.version,
                latency_ms=elapsed_ms, task_text=self.task_text,
            ),
            delivered_history_novelty=FeatureValue.unavailable(
                "delivered_history_novelty", method="one_minus_max_history_cosine",
                source="receiver_delivered_history", version=self.version,
                reason="history_not_materialized",
            ),
            task_impact=FeatureValue.measured(
                "task_impact", annotation.task_impact, method="reviewed_task_rubric",
                source=annotation.annotation_source, version=self.version, **annotation_provenance,
            ),
            urgency=FeatureValue.measured(
                "urgency", annotation.urgency, method="reviewed_task_rubric",
                source=annotation.annotation_source, version=self.version, **annotation_provenance,
            ),
            evidence_sufficiency=FeatureValue.measured(
                "evidence_sufficiency", annotation.evidence_sufficiency,
                method="reviewed_evidence_sufficiency", source=annotation.annotation_source,
                version=self.version, **annotation_provenance,
            ),
            extraction_latency_ms=elapsed_ms,
            extractor_version=self.version,
            metadata=dict(metadata or {}),
        )
