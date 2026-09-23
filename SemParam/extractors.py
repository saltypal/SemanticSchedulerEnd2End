"""Independent deterministic extractors for the seven Stage 1B parameters."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from INFRA.Artifacts import (
    KnowledgeArtifact,
    ObservationArtifact,
    ParameterDefinition,
    ParameterValue,
    TaskContext,
)
from INFRA.Artifacts.observation_artifact import ensure_utc
from INFRA.Interfaces import ParameterExtractorInterface
from SemParam.catalog import SemanticPipelineSettings


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class _BaseExtractor(ParameterExtractorInterface):
    source = "stage1b"

    def __init__(self, definition: ParameterDefinition, settings: SemanticPipelineSettings) -> None:
        self.definition = definition
        self.settings = settings

    def _value(self, value: float, raw_value: Any, method: str, source: str | None = None) -> ParameterValue:
        return ParameterValue(
            value=float(value),
            raw_value=raw_value,
            available=True,
            method=method,
            source=source or self.source,
            version=self.settings.version,
            provenance={"parameter": self.definition.name},
            units=self.definition.units,
            producer_level=self.definition.producer_level,
        )

    def _unavailable(self, reason: str, method: str, source: str | None = None, **provenance: Any) -> ParameterValue:
        return ParameterValue.unavailable(
            method=method,
            source=source or self.source,
            version=self.settings.version,
            reason=reason,
            units=self.definition.units,
            producer_level=self.definition.producer_level,
            provenance={"parameter": self.definition.name, **provenance},
        )


class AoIExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        if observation.captured_at is None:
            return self._unavailable("capture_timestamp_missing", "capture_timestamp_difference", "image_metadata")
        seconds = (ensure_utc(now) - observation.captured_at).total_seconds()
        if seconds < 0:
            return self._unavailable(
                "capture_timestamp_is_in_future",
                "capture_timestamp_difference",
                "image_metadata",
                captured_at=observation.captured_at,
                now=ensure_utc(now),
            )
        return self._value(seconds, {"captured_at": observation.captured_at, "now": ensure_utc(now)}, "capture_timestamp_difference", "image_metadata")


class FreshnessExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        if observation.captured_at is None:
            return self._unavailable("capture_timestamp_missing", "exponential_aoi_decay", "image_metadata")
        age_seconds = (ensure_utc(now) - observation.captured_at).total_seconds()
        if age_seconds < 0:
            return self._unavailable("capture_timestamp_is_in_future", "exponential_aoi_decay", "image_metadata")
        value = math.exp(-age_seconds / self.settings.freshness_tau_seconds)
        return self._value(
            value,
            {"AoI_seconds": age_seconds, "tau_seconds": self.settings.freshness_tau_seconds},
            "exponential_aoi_decay",
            "derived_from_AoI",
        )


class NoveltyExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        max_similarity = knowledge.evidence.get("max_cosine_similarity")
        if max_similarity is None:
            return self._value(1.0, {"history_size": 0, "max_cosine_similarity": None}, "one_minus_max_history_cosine", knowledge.knowledge_source)
        similarity = float(max_similarity)
        return self._value(
            _bounded(1.0 - similarity),
            {
                "history_size": knowledge.evidence.get("history_size"),
                "max_cosine_similarity": similarity,
                "nearest_observation_id": knowledge.evidence.get("nearest_observation_id"),
            },
            "one_minus_max_history_cosine",
            knowledge.knowledge_source,
        )


class RelevanceExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        task_embedding = observation.task_embeddings.get(f"task:{task.task_id}")
        if task_embedding is None:
            return self._unavailable("task_embedding_missing", "normalized_cosine_similarity", "clip_text_cache")
        similarity = float(np.dot(np.asarray(observation.image_embedding), np.asarray(task_embedding)))
        return self._value(
            _bounded((similarity + 1.0) / 2.0),
            {"cosine_similarity": similarity, "task_id": task.task_id},
            "normalized_cosine_similarity",
            "clip_image_task_embedding",
        )


class ReliabilityExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        if not observation.concept_scores:
            return self._unavailable("configured_concept_scores_missing", "maximum_zero_shot_probability", "clip_zero_shot")
        concept, probability = max(observation.concept_scores.items(), key=lambda item: item[1])
        return self._value(
            _bounded(float(probability)),
            {"top_concept": concept, "top_concept_probability": float(probability)},
            "maximum_zero_shot_probability",
            "clip_zero_shot_semantic_confidence",
        )


class UrgencyExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        if task.deadline_at is None:
            return self._unavailable("deadline_or_urgency_context_missing", "deadline_pressure", "task_context")
        seconds_remaining = (task.deadline_at - ensure_utc(now)).total_seconds()
        value = _bounded(1.0 - seconds_remaining / task.urgency_horizon_seconds)
        return self._value(
            value,
            {
                "deadline_at": task.deadline_at,
                "now": ensure_utc(now),
                "seconds_remaining": seconds_remaining,
                "horizon_seconds": task.urgency_horizon_seconds,
            },
            "deadline_pressure",
            "task_context.deadline",
        )


class TaskImportanceExtractor(_BaseExtractor):
    def extract(self, observation: ObservationArtifact, task: TaskContext, knowledge: KnowledgeArtifact, now: datetime) -> ParameterValue:
        if not task.concept_importance:
            return self._unavailable("task_policy_missing", "probability_weighted_concept_policy", "task_policy")
        matched = {
            concept: (float(observation.concept_scores[concept]), float(weight))
            for concept, weight in task.concept_importance.items()
            if concept in observation.concept_scores
        }
        if not matched:
            return self._unavailable("task_policy_has_no_configured_concept_overlap", "probability_weighted_concept_policy", "task_policy")
        raw_value = sum(probability * weight for probability, weight in matched.values())
        return self._value(
            _bounded(raw_value),
            {"matched_concepts": matched, "weighted_sum": raw_value},
            "probability_weighted_concept_policy",
            "clip_zero_shot_plus_task_policy",
        )


EXTRACTOR_TYPES = {
    "AoI_seconds": AoIExtractor,
    "freshness": FreshnessExtractor,
    "novelty": NoveltyExtractor,
    "relevance": RelevanceExtractor,
    "reliability": ReliabilityExtractor,
    "urgency": UrgencyExtractor,
    "task_importance": TaskImportanceExtractor,
}


def build_extractors(settings: SemanticPipelineSettings) -> dict[str, ParameterExtractorInterface]:
    """Instantiate independent extractors only for enabled catalog entries."""

    return {
        definition.name: EXTRACTOR_TYPES[definition.name](definition, settings)
        for definition in settings.catalog
    }
