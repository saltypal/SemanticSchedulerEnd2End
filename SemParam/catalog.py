"""Stage 1B parameter catalog and deterministic extractor settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from INFRA.Artifacts import ParameterDefinition, ParameterProducerLevel


SEMANTIC_PARAMETER_ORDER = (
    "AoI_seconds",
    "freshness",
    "novelty",
    "relevance",
    "reliability",
    "urgency",
    "task_importance",
)


DEFAULT_PARAMETER_CATALOG: dict[str, ParameterDefinition] = {
    "AoI_seconds": ParameterDefinition(
        name="AoI_seconds",
        producer_level=ParameterProducerLevel.UE,
        units="seconds",
        header_encoding="aoi_milliseconds",
        description="Age of the image relative to an explicit capture timestamp.",
    ),
    "freshness": ParameterDefinition(
        name="freshness",
        producer_level=ParameterProducerLevel.UE,
        units="normalized_score",
        header_encoding="normalized_score",
        description="Exponential freshness decay derived from AoI.",
    ),
    "novelty": ParameterDefinition(
        name="novelty",
        producer_level=ParameterProducerLevel.UE,
        units="normalized_score",
        header_encoding="normalized_score",
        description="Dissimilarity to the local sequential knowledge-base history.",
    ),
    "relevance": ParameterDefinition(
        name="relevance",
        producer_level=ParameterProducerLevel.UE,
        units="normalized_score",
        header_encoding="normalized_score",
        description="Semantic similarity between image evidence and the declared task.",
    ),
    "reliability": ParameterDefinition(
        name="reliability",
        producer_level=ParameterProducerLevel.UE,
        units="normalized_score",
        header_encoding="normalized_score",
        description="CLIP semantic-confidence proxy, not an independent source-trust score.",
    ),
    "urgency": ParameterDefinition(
        name="urgency",
        producer_level=ParameterProducerLevel.UE,
        units="normalized_score",
        header_encoding="normalized_score",
        description="Deadline pressure from explicit context; unavailable without it.",
    ),
    "task_importance": ParameterDefinition(
        name="task_importance",
        producer_level=ParameterProducerLevel.UE,
        units="normalized_score",
        header_encoding="normalized_score",
        description="Image evidence weighted by an explicit task-policy concept map.",
    ),
}


def resolve_catalog(enabled_parameters: Sequence[str] | None = None) -> tuple[ParameterDefinition, ...]:
    """Return ordered catalog entries and reject unsupported fields early."""

    selected = tuple(enabled_parameters or SEMANTIC_PARAMETER_ORDER)
    unknown = [name for name in selected if name not in DEFAULT_PARAMETER_CATALOG]
    if unknown:
        raise ValueError(f"Unknown Stage 1B semantic parameter(s): {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("Semantic parameter configuration must not repeat field names.")
    return tuple(DEFAULT_PARAMETER_CATALOG[name] for name in selected)


@dataclass(frozen=True, slots=True)
class SemanticPipelineSettings:
    """Settings that affect values and therefore belong in provenance."""

    concept_prompts: Mapping[str, str]
    enabled_parameters: tuple[str, ...] = SEMANTIC_PARAMETER_ORDER
    freshness_tau_seconds: float = 10.0
    clip_temperature: float = 0.07
    version: str = "stage1b-transformer/v1"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.concept_prompts:
            raise ValueError("Stage 1B requires at least one configured concept prompt.")
        if self.freshness_tau_seconds <= 0:
            raise ValueError("freshness_tau_seconds must be positive.")
        if self.clip_temperature <= 0:
            raise ValueError("clip_temperature must be positive.")
        resolve_catalog(self.enabled_parameters)

    @property
    def catalog(self) -> tuple[ParameterDefinition, ...]:
        return resolve_catalog(self.enabled_parameters)
