"""Runtime codebook semantic extraction with no CLIP dependency or fallback."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from SemParam.Codebooks.semantic_codebook import CONTENT_PARAMETER_NAMES, SemanticParameterCodebook, pool_swin_representation
from INFRA.features import FeatureProvider
from INFRA.hashing import file_sha256
from INFRA.Artifacts import (
    ImageArtifact,
    ObservationArtifact,
    ParameterValue,
    SemanticParameterBundle,
    TaskContext,
)
from INFRA.Artifacts.observation_artifact import ensure_utc
from INFRA.Interfaces import KnowledgeInterface
from SemParam.catalog import SemanticPipelineSettings
from SemParam.features.extractors import build_extractors


@dataclass(slots=True)
class CodebookExtractionResult:
    bundle: SemanticParameterBundle
    latent: Any
    semantic_representation: Any


class CodebookParameterPipeline:
    """Swin representation -> semantic codeword -> same bundle API as Transformer."""

    name = "codebook_swin_semantic"

    def __init__(
        self,
        settings: SemanticPipelineSettings,
        codebook: SemanticParameterCodebook,
        knowledge_base: KnowledgeInterface,
        feature_provider: FeatureProvider | None = None,
    ) -> None:
        self.settings = settings
        self.feature_provider = feature_provider
        self.codebook = codebook
        self.knowledge_base = knowledge_base
        self.extractors = build_extractors(settings)
        self._sequence = 0

    def _observation_from_representation(
        self,
        representation: Any,
        source_path: str | Path,
        *,
        captured_at: datetime | None,
        metadata: dict[str, Any] | None,
    ) -> ObservationArtifact:
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Codebook semantic source image does not exist: {path}")
        descriptor = pool_swin_representation(representation)
        norm = float(np.linalg.norm(descriptor))
        if norm == 0.0:
            raise ValueError("Pooled Swin semantic descriptor has zero norm.")
        self._sequence += 1
        image_hash = file_sha256(path)
        observation_id = hashlib.sha256(
            f"codebook:{path.resolve()}:{image_hash}:{self._sequence}".encode("utf-8")
        ).hexdigest()[:24]
        return ObservationArtifact(
            observation_id=observation_id,
            image_hash=image_hash,
            source_path=str(path),
            image_embedding=descriptor / norm,
            concept_scores={},
            captured_at=captured_at,
            metadata=dict(metadata or {}),
        )

    def extract_from_representation(
        self,
        representation: Any,
        source_path: str | Path,
        task: TaskContext,
        now: datetime,
        *,
        captured_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticParameterBundle:
        """Build a complete seven-field bundle without loading or calling CLIP."""

        self.codebook.require_task_profile(task.task_id)
        reference_now = ensure_utc(now)
        observation = self._observation_from_representation(
            representation,
            source_path,
            captured_at=captured_at,
            metadata=metadata,
        )
        knowledge = self.knowledge_base.query(observation)
        values: dict[str, ParameterValue] = {}
        for name in ("AoI_seconds", "freshness", "novelty", "urgency"):
            extractor = self.extractors.get(name)
            if extractor is not None:
                values[name] = extractor.extract(observation, task, knowledge, reference_now)
        prediction = self.codebook.predict(representation, task_profile_id=task.task_id)
        definitions = {definition.name: definition for definition in self.settings.catalog}
        for name in CONTENT_PARAMETER_NAMES:
            if name not in definitions:
                continue
            definition = definitions[name]
            value = prediction.values[name]
            values[name] = ParameterValue(
                value=value,
                raw_value={"codeword_id": prediction.codeword_id, "squared_distance": prediction.squared_distance},
                available=True,
                method="nearest_semantic_codeword",
                source="swin_semantic_parameter_codebook",
                version=self.codebook.schema_version,
                provenance={
                    "task_profile_id": self.codebook.task_profile_id,
                    "codebook_content_hash": self.codebook.content_hash(),
                    "demo_only": self.codebook.demo_only,
                },
                units=definition.units,
                producer_level=definition.producer_level,
            )
        unknown_provided = sorted(set(task.provided_parameters).difference(definitions))
        if unknown_provided:
            raise ValueError(f"TaskContext supplied values outside the enabled semantic catalog: {unknown_provided}")
        values.update(task.provided_parameters)
        ordered_values = {definition.name: values[definition.name] for definition in self.settings.catalog if definition.name in values}
        bundle = SemanticParameterBundle(
            values=ordered_values,
            extractor_name=self.name,
            metadata={
                "observation_id": observation.observation_id,
                "image_hash": observation.image_hash,
                "task_id": task.task_id,
                "codebook_demo_only": self.codebook.demo_only,
                "codeword_id": prediction.codeword_id,
                "knowledge_source": knowledge.knowledge_source,
            },
        )
        self.knowledge_base.record(observation, bundle)
        return bundle

    def extract_with_latent(
        self,
        adapter: Any,
        image: ImageArtifact,
        source_path: str | Path,
        task: TaskContext,
        now: datetime,
        *,
        snr_db: float,
        rate: int,
        captured_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CodebookExtractionResult:
        """Capture Swin semantics and Stage 1A latent in the same encoder forward pass."""

        provider = self.feature_provider
        if provider is None:
            # Only the compatibility method supplies an implicit Swin provider.
            from EncDec.Models.SwinJSCC.features import SwinFeatureProvider
            provider = SwinFeatureProvider(adapter)
        capture = provider.capture(image, snr_db=snr_db, rate=rate)
        bundle = self.extract_from_representation(
            capture.representation,
            source_path,
            task,
            now,
            captured_at=captured_at,
            metadata=metadata,
        )
        return CodebookExtractionResult(
            bundle=bundle,
            latent=capture.latent,
            semantic_representation=capture.representation,
        )


    def extract_with_features(self, image, source_path, task, now, *, feature_options=None, captured_at=None, metadata=None):
        """Use an injected provider; no codec implementation is imported here."""
        if self.feature_provider is None:
            raise ValueError('extract_with_features requires an explicit FeatureProvider')
        capture = self.feature_provider.capture(image, **(feature_options or {}))
        bundle = self.extract_from_representation(capture.representation, source_path, task, now,
                                                  captured_at=captured_at, metadata=metadata)
        return CodebookExtractionResult(bundle, capture.latent, capture.representation)
