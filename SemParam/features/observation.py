"""Build reusable semantic observations from a source image and cached embeddings."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from INFRA.hashing import file_sha256
from INFRA.Artifacts import ObservationArtifact, TaskContext
from INFRA.Artifacts.observation_artifact import ensure_utc
from SemParam.catalog import SemanticPipelineSettings
from SemParam.features.clip import SemanticEmbeddingBackend, _normalize


def parse_semantic_timestamp(value: str | datetime | None) -> datetime | None:
    """Parse only explicit, timezone-aware timestamps from config or metadata."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    normalized = value.replace("Z", "+00:00")
    return ensure_utc(datetime.fromisoformat(normalized))


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


class ObservationBuilder:
    """Owns the one CLIP image call and all cached text-embedding lookups."""

    def __init__(self, backend: SemanticEmbeddingBackend, settings: SemanticPipelineSettings) -> None:
        self.backend = backend
        self.settings = settings
        self._sequence = 0

    def build(
        self,
        image_path: str | Path,
        task: TaskContext,
        *,
        captured_at: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObservationArtifact:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Semantic observation image does not exist: {path}")
        image_embedding = _normalize(self.backend.encode_image(path))
        concept_names = tuple(self.settings.concept_prompts)
        prompts = [self.settings.concept_prompts[name] for name in concept_names]
        requested_texts = [*prompts, task.task_text]
        if task.context_text:
            requested_texts.append(task.context_text)
        text_embeddings = self.backend.encode_texts(requested_texts)
        concept_matrix = np.stack([_normalize(text_embeddings[prompt]) for prompt in prompts])
        similarities = concept_matrix @ image_embedding
        probabilities = _softmax(similarities, self.settings.clip_temperature)
        concept_scores = {name: float(score) for name, score in zip(concept_names, probabilities, strict=True)}
        task_embeddings = {f"task:{task.task_id}": _normalize(text_embeddings[task.task_text])}
        if task.context_text:
            task_embeddings[f"context:{task.task_id}"] = _normalize(text_embeddings[task.context_text])
        image_hash = file_sha256(path)
        self._sequence += 1
        observation_id = hashlib.sha256(
            f"{path.resolve()}:{image_hash}:{self._sequence}".encode("utf-8")
        ).hexdigest()[:24]
        return ObservationArtifact(
            observation_id=observation_id,
            image_hash=image_hash,
            source_path=str(path),
            image_embedding=image_embedding,
            concept_scores=concept_scores,
            task_embeddings=task_embeddings,
            captured_at=parse_semantic_timestamp(captured_at),
            metadata=dict(metadata or {}),
        )
