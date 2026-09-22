"""Offline-trained semantic parameter codebook; independent of communication VQ."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from INFRA.Registries import PARAMETER_CODEBOOK_REGISTRY


CONTENT_PARAMETER_NAMES = ("relevance", "reliability", "task_importance")


def pool_swin_representation(representation: Any) -> np.ndarray:
    """Deterministically reduce a Swin semantic representation to one descriptor."""

    detached = getattr(representation, "detach", None)
    if callable(detached):
        representation = detached()
    to_cpu = getattr(representation, "cpu", None)
    if callable(to_cpu):
        representation = to_cpu()
    to_numpy = getattr(representation, "numpy", None)
    values = np.asarray(to_numpy() if callable(to_numpy) else representation, dtype=np.float32)
    if values.ndim == 3:
        pooled = values.mean(axis=(0, 1))
    elif values.ndim == 2:
        pooled = values.mean(axis=0)
    elif values.ndim == 1:
        pooled = values
    else:
        raise ValueError(f"Expected [batch, tokens, channels], [tokens, channels], or [channels], got {values.shape}")
    if not np.all(np.isfinite(pooled)):
        raise ValueError("Swin semantic representation contains non-finite values.")
    return np.asarray(pooled, dtype=np.float32).reshape(-1)


@dataclass(frozen=True, slots=True)
class SemanticCodebookPrediction:
    codeword_id: int
    values: dict[str, float]
    squared_distance: float


@dataclass(slots=True)
class SemanticParameterCodebook:
    """A small nearest-centroid map from pooled Swin features to parameter targets."""

    centroids: np.ndarray
    target_vectors: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    task_profile_id: str
    training_image_hashes: tuple[str, ...]
    random_state: int
    demo_only: bool = True
    schema_version: str = "semantic-parameter-codebook/v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.centroids.ndim != 2 or self.target_vectors.ndim != 2:
            raise ValueError("Semantic codebook centroids and targets must be rank-two arrays.")
        if self.centroids.shape[0] != self.target_vectors.shape[0]:
            raise ValueError("Semantic codebook centroid and target counts differ.")
        if self.target_vectors.shape[1] != len(CONTENT_PARAMETER_NAMES):
            raise ValueError("Semantic codebook target vectors must contain exactly the content-derived parameters.")
        if self.centroids.shape[1] != self.pca_components.shape[0]:
            raise ValueError("Semantic codebook centroid dimension does not match PCA output dimension.")
        if not self.task_profile_id:
            raise ValueError("Semantic codebook requires a task_profile_id.")

    @classmethod
    def train(
        cls,
        representations: Sequence[Any],
        target_rows: Sequence[Mapping[str, float]],
        *,
        task_profile_id: str,
        training_image_hashes: Sequence[str],
        random_state: int = 20260906,
        codebook_size: int | None = None,
        demo_only: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SemanticParameterCodebook":
        """Fit the deterministic demonstration codebook from offline teacher pairs."""

        if len(representations) != len(target_rows) or len(representations) != len(training_image_hashes):
            raise ValueError("Representations, target rows, and image hashes must have equal length.")
        if len(representations) < 2:
            raise ValueError("Semantic codebook training requires at least two distinct images.")
        if len(set(training_image_hashes)) != len(training_image_hashes):
            raise ValueError("Semantic codebook training images must be unique by SHA-256 hash.")
        descriptors = np.stack([pool_swin_representation(value) for value in representations])
        targets = np.asarray(
            [[float(row[name]) for name in CONTENT_PARAMETER_NAMES] for row in target_rows],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(targets)) or np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("Semantic codebook teacher targets must be finite normalized scores.")
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:
            raise RuntimeError("Semantic codebook training requires scikit-learn.") from error
        scaler = StandardScaler()
        normalized = scaler.fit_transform(descriptors)
        pca_dimensions = min(8, len(representations) - 1, normalized.shape[1])
        if pca_dimensions < 1:
            raise ValueError("Semantic codebook training needs at least one PCA dimension.")
        # `full` is deterministic; randomized SVD is intentionally not used.
        pca = PCA(n_components=pca_dimensions, svd_solver="full")
        projected = pca.fit_transform(normalized)
        cluster_count = min(codebook_size or 4, len(representations))
        kmeans = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10, algorithm="lloyd")
        labels = kmeans.fit_predict(projected)
        target_vectors = np.vstack([targets[labels == identifier].mean(axis=0) for identifier in range(cluster_count)])
        provenance = {
            "teacher": "offline_clip_transformer_parameter_bundle",
            "pooling": "global_average_pooling",
            "projection": "StandardScaler_then_PCA_full",
            "clustering": "KMeans_lloyd_fixed_seed",
            "content_parameter_names": list(CONTENT_PARAMETER_NAMES),
            "training_image_count": len(representations),
            **dict(metadata or {}),
        }
        return cls(
            centroids=np.asarray(kmeans.cluster_centers_, dtype=np.float32),
            target_vectors=np.asarray(target_vectors, dtype=np.float32),
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
            pca_mean=np.asarray(pca.mean_, dtype=np.float32),
            pca_components=np.asarray(pca.components_, dtype=np.float32),
            task_profile_id=task_profile_id,
            training_image_hashes=tuple(training_image_hashes),
            random_state=random_state,
            demo_only=demo_only,
            metadata=provenance,
        )

    def require_task_profile(self, task_profile_id: str) -> None:
        if task_profile_id != self.task_profile_id:
            raise ValueError(
                "Semantic codebook task-profile mismatch: "
                f"artifact={self.task_profile_id!r}, runtime={task_profile_id!r}."
            )

    def _project(self, representation: Any) -> np.ndarray:
        descriptor = pool_swin_representation(representation)
        if descriptor.shape[0] != self.scaler_mean.shape[0]:
            raise ValueError(
                "Swin representation width does not match the codebook artifact: "
                f"runtime={descriptor.shape[0]}, artifact={self.scaler_mean.shape[0]}."
            )
        standardized = (descriptor - self.scaler_mean) / self.scaler_scale
        centered = standardized - self.pca_mean
        return centered @ self.pca_components.T

    def predict(self, representation: Any, *, task_profile_id: str) -> SemanticCodebookPrediction:
        """Predict content-derived values without importing or calling CLIP at runtime."""

        self.require_task_profile(task_profile_id)
        projected = self._project(representation)
        squared_distances = np.sum((self.centroids - projected[None, :]) ** 2, axis=1)
        codeword_id = int(np.argmin(squared_distances))
        values = {
            name: float(value)
            for name, value in zip(CONTENT_PARAMETER_NAMES, self.target_vectors[codeword_id], strict=True)
        }
        return SemanticCodebookPrediction(
            codeword_id=codeword_id,
            values=values,
            squared_distance=float(squared_distances[codeword_id]),
        )

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (self.centroids, self.target_vectors, self.scaler_mean, self.scaler_scale, self.pca_mean, self.pca_components):
            digest.update(np.asarray(array, dtype=np.float32).tobytes())
        digest.update(self.task_profile_id.encode("utf-8"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:
        """Persist arrays and auditable metadata without serializing Python model objects."""

        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Semantic parameter codebooks must use the .npz artifact format.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "task_profile_id": self.task_profile_id,
            "training_image_hashes": list(self.training_image_hashes),
            "random_state": self.random_state,
            "demo_only": self.demo_only,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
            "content_hash": self.content_hash(),
        }
        np.savez_compressed(
            destination,
            centroids=self.centroids,
            target_vectors=self.target_vectors,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SemanticParameterCodebook":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Semantic parameter codebook does not exist: {source}")
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            instance = cls(
                centroids=np.asarray(archive["centroids"], dtype=np.float32),
                target_vectors=np.asarray(archive["target_vectors"], dtype=np.float32),
                scaler_mean=np.asarray(archive["scaler_mean"], dtype=np.float32),
                scaler_scale=np.asarray(archive["scaler_scale"], dtype=np.float32),
                pca_mean=np.asarray(archive["pca_mean"], dtype=np.float32),
                pca_components=np.asarray(archive["pca_components"], dtype=np.float32),
                task_profile_id=str(metadata["task_profile_id"]),
                training_image_hashes=tuple(str(value) for value in metadata["training_image_hashes"]),
                random_state=int(metadata["random_state"]),
                demo_only=bool(metadata["demo_only"]),
                schema_version=str(metadata["schema_version"]),
                metadata=dict(metadata.get("metadata", {})),
            )
        expected_hash = metadata.get("content_hash")
        if expected_hash and instance.content_hash() != expected_hash:
            raise ValueError("Semantic parameter codebook content hash mismatch.")
        return instance


def register_semantic_parameter_codebook() -> None:
    """Register in the semantic-codebook registry, never the communication VQ registry."""

    if not PARAMETER_CODEBOOK_REGISTRY.contains("swin_semantic_parameter_codebook"):
        PARAMETER_CODEBOOK_REGISTRY.register("swin_semantic_parameter_codebook", SemanticParameterCodebook)
