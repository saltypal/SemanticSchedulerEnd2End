"""Convert extracted image evidence into immutable scheduler job features.

SemParam owns extraction and provenance. INFRA owns the compact policy-facing
snapshot. Keeping the conversion explicit prevents model embeddings and manual
labels from accidentally entering a policy observation.
"""

from __future__ import annotations

from INFRA.semantic_contracts import FeatureValue, SchedulingFeatureVectorV2
from SemParam.scheduler_features import SchedulingFeatureVectorV2 as ExtractedFeatures


def to_policy_features(extracted: ExtractedFeatures) -> SchedulingFeatureVectorV2:
    """Copy the five values and cached embedding across the module boundary."""

    def convert(name: str) -> FeatureValue:
        source = getattr(extracted, name)
        return FeatureValue(
            value=source.value,
            available=source.available,
            provenance=f"{source.source}:{source.method}",
            extractor_version=source.version,
            latency_ms=0.0 if source.latency_ms is None else float(source.latency_ms),
        )

    return SchedulingFeatureVectorV2(
        context_relevance=convert("context_relevance"),
        delivered_history_novelty=convert("delivered_history_novelty"),
        task_impact=convert("task_impact"),
        urgency=convert("urgency"),
        evidence_sufficiency=convert("evidence_sufficiency"),
        embedding=tuple(extracted.image_embedding),
    )
