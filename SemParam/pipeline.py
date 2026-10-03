"""Composable CLIP-backed Stage 1B semantic parameter pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from INFRA.Artifacts import ObservationArtifact, SemanticParameterBundle, TaskContext
from INFRA.Artifacts.observation_artifact import ensure_utc
from INFRA.Interfaces import KnowledgeInterface
from SemParam.catalog import SemanticPipelineSettings
from SemParam.features.clip import SemanticEmbeddingBackend
from SemParam.features.extractors import build_extractors
from SemParam.features.observation import ObservationBuilder


class TransformerParameterPipeline:
    """Build one observation, invoke independent extractors, then update knowledge."""

    name = "transformer_clip"

    def __init__(
        self,
        settings: SemanticPipelineSettings,
        embedding_backend: SemanticEmbeddingBackend,
        knowledge_base: KnowledgeInterface,
    ) -> None:
        self.settings = settings
        self.knowledge_base = knowledge_base
        self.observation_builder = ObservationBuilder(embedding_backend, settings)
        self.extractors = build_extractors(settings)

    def prepare_observation(
        self,
        image_path: str | Path,
        task: TaskContext,
        *,
        captured_at: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObservationArtifact:
        return self.observation_builder.build(image_path, task, captured_at=captured_at, metadata=metadata)

    def extract_from_path(
        self,
        image_path: str | Path,
        task: TaskContext,
        now: datetime,
        *,
        captured_at: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticParameterBundle:
        observation = self.prepare_observation(
            image_path,
            task,
            captured_at=captured_at,
            metadata=metadata,
        )
        return self.extract_observation(observation, task, now)

    def extract_observation(
        self,
        observation: ObservationArtifact,
        task: TaskContext,
        now: datetime,
    ) -> SemanticParameterBundle:
        reference_now = ensure_utc(now)
        knowledge = self.knowledge_base.query(observation)
        values = {
            name: extractor.extract(observation, task, knowledge, reference_now)
            for name, extractor in self.extractors.items()
        }
        unknown_provided = sorted(set(task.provided_parameters).difference(values))
        if unknown_provided:
            raise ValueError(f"TaskContext supplied values outside the enabled semantic catalog: {unknown_provided}")
        # O-RAN/external producers can supply a fully-provenanced value without
        # making this UE image path invent or recompute it.
        values.update(task.provided_parameters)
        bundle = SemanticParameterBundle(
            values=values,
            extractor_name=self.name,
            metadata={
                "observation_id": observation.observation_id,
                "image_hash": observation.image_hash,
                "source_path": observation.source_path,
                "task_id": task.task_id,
                "knowledge_source": knowledge.knowledge_source,
                "knowledge_evidence": knowledge.evidence,
                "settings_version": self.settings.version,
            },
        )
        # Query happens before record so an observation never makes itself non-novel.
        self.knowledge_base.record(observation, bundle)
        return bundle
