"""Image/task features and novelty for the device-independent latent scheduler."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from INFRA.scheduling.semantic_contracts import cosine_similarity


def confirmed_novelty(embedding, delivered_embeddings) -> float | None:
    """Cosine distance proxy, unavailable without compatible receiver evidence."""
    similarities = [cosine_similarity(embedding, prior) for prior in delivered_embeddings]
    valid = [value for value in similarities if value is not None]
    return None if not valid else (1.0 - max(valid)) / 2.0


def extract_latent_features(image_path, *, task_id, task_text, backend=None):
    """Run the image backbone once. No priority or urgency is guessed from pixels."""
    if not task_id or not task_text:
        raise ValueError("An explicit task ID and task text are required")
    if backend is None:
        from SemParam.features.clip import CLIPBackbone
        backend = CLIPBackbone()
    path = Path(image_path)
    image_embedding = tuple(float(value) for value in backend.encode_image(path))
    task_embedding = tuple(float(value) for value in backend.encode_texts([task_text])[task_text])
    relevance = cosine_similarity(image_embedding, task_embedding)
    if relevance is None:
        raise ValueError("Feature backend returned incompatible or zero embeddings")
    return {
        "source_sha256": sha256(path.read_bytes()).hexdigest(),
        "task_id": task_id,
        "task_text_sha256": sha256(task_text.encode()).hexdigest(),
        "embedding": image_embedding,
        "task_embedding": task_embedding,
        "context_relevance": relevance,
        "extractor": getattr(backend, "model_name", type(backend).__name__),
        "task_importance": None,
        "priority_probabilities": None,
        "priority_source": "unavailable",
    }
