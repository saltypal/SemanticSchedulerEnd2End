"""Immutable contracts for semantic whole-latent scheduling.

The legacy :mod:`INFRA.scheduling.contracts` types power the existing slotted simulator.
This module is additive.  It keeps evaluator-only labels out of policy
snapshots while retaining enough provenance to make a simulation replayable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping


FEATURE_NAMES = (
    "scene_importance", "context_relevance", "delivered_history_novelty",
    "interpretation_confidence", "task_impact", "urgency",
    "evidence_sufficiency",
)
BASE_PRIORITY_WEIGHTS = (4.0, 3.0, 2.0, 1.0)


def _finite(value: float, name: str, *, lower: float | None = None, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}")


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One feature with availability and measurement provenance."""

    value: float | None
    available: bool
    provenance: str
    extractor_version: str
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        _name(self.provenance, "provenance")
        _name(self.extractor_version, "extractor_version")
        _finite(self.latency_ms, "latency_ms", lower=0.0)
        if self.available:
            if self.value is None:
                raise ValueError("available feature values require a numeric value")
            _finite(self.value, "feature value")
        elif self.value is not None:
            raise ValueError("unavailable feature values must use None")

    @classmethod
    def unavailable(cls, provenance: str, extractor_version: str, latency_ms: float = 0.0) -> "FeatureValue":
        return cls(None, False, provenance, extractor_version, latency_ms)


@dataclass(frozen=True, slots=True)
class SchedulingFeatureVectorV2:
    """Seven scheduler features plus the cached image embedding.

    ``embedding`` is used by INFRA to refresh novelty at decision time.  It is
    deliberately not placed in a policy snapshot.
    """

    context_relevance: FeatureValue
    delivered_history_novelty: FeatureValue
    task_impact: FeatureValue
    urgency: FeatureValue
    evidence_sufficiency: FeatureValue
    embedding: tuple[float, ...] = ()
    schema_id: int = 2
    scene_importance: FeatureValue = field(default_factory=lambda:
        FeatureValue.unavailable("classifier_not_attached", "track-a/v2"))
    interpretation_confidence: FeatureValue = field(default_factory=lambda:
        FeatureValue.unavailable("classifier_not_attached", "track-a/v2"))

    def __post_init__(self) -> None:
        if self.schema_id != 2:
            raise ValueError("SchedulingFeatureVectorV2 requires schema_id=2")
        for field_name in FEATURE_NAMES:
            if not isinstance(getattr(self, field_name), FeatureValue):
                raise ValueError(f"{field_name} must be a FeatureValue")
        if not isinstance(self.embedding, tuple):
            raise ValueError("embedding must be an immutable tuple")
        for value in self.embedding:
            _finite(value, "embedding value")

    @property
    def values(self) -> tuple[FeatureValue, ...]:
        return tuple(getattr(self, name) for name in FEATURE_NAMES)

    def normalized_values(self, *, cold_start_novelty: float = 1.0, unavailable_fill: float = 0.0) -> tuple[float, ...]:
        """Return model values in the published order without availability bits."""
        p1, p2, p3, p4, p5, p6, p7 = self.values
        def present(item: FeatureValue, transform, fallback: float) -> float:
            return transform(item.value) if item.available and item.value is not None else fallback
        return (
            min(1.0, max(0.0, present(p1, lambda x: x, unavailable_fill))),
            min(1.0, max(0.0, present(p2, lambda x: (x + 1.0) / 2.0, unavailable_fill))),
            min(1.0, max(0.0, present(p3, lambda x: x, cold_start_novelty))),
            min(1.0, max(0.0, present(p4, lambda x: x, unavailable_fill))),
            min(1.0, max(0.0, present(p5, lambda x: x / 3.0, unavailable_fill))),
            min(1.0, max(0.0, present(p6, lambda x: x / 3.0, unavailable_fill))),
            min(1.0, max(0.0, present(p7, lambda x: x, unavailable_fill))),
        )

    def availability_mask(self) -> tuple[bool, ...]:
        return tuple(item.available for item in self.values)

    def with_novelty(self, value: FeatureValue) -> "SchedulingFeatureVectorV2":
        return replace(self, delivered_history_novelty=value)


@dataclass(frozen=True, slots=True)
class LatentJobSpec:
    """Policy-visible immutable whole-latent job descriptor."""

    job_id: str
    image_id: str
    ue_id: str
    task_id: str
    payload_id: str
    arrival_ms: float
    deadline_ms: float | None
    resource_units: int
    resource_unit: str
    features: SchedulingFeatureVectorV2
    payload_provenance: str = "unavailable"

    def __post_init__(self) -> None:
        for name in ("job_id", "image_id", "ue_id", "task_id", "payload_id", "payload_provenance"):
            _name(getattr(self, name), name)
        _finite(self.arrival_ms, "arrival_ms", lower=0.0)
        if self.deadline_ms is not None:
            _finite(self.deadline_ms, "deadline_ms", lower=self.arrival_ms)
        if type(self.resource_units) is not int or self.resource_units <= 0:
            raise ValueError("resource_units must be a positive integer")
        if self.resource_unit not in ("complex_channel_uses", "digital_bits", "simulated_units"):
            raise ValueError("resource_unit is unsupported")


@dataclass(frozen=True, slots=True)
class EpisodeJobSpec:
    """INFRA-only facts used for utility evaluation and transport outcomes."""

    candidate: LatentJobSpec
    base_priority: int
    evidence_group_id: str
    transport_success: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, LatentJobSpec):
            raise ValueError("candidate must be a LatentJobSpec")
        if type(self.base_priority) is not int or not 0 <= self.base_priority < len(BASE_PRIORITY_WEIGHTS):
            raise ValueError("base_priority must be an integer in [0, 3]")
        _name(self.evidence_group_id, "evidence_group_id")
        if type(self.transport_success) is not bool:
            raise ValueError("transport_success must be boolean")


@dataclass(frozen=True, slots=True)
class DeliveredKnowledgeSnapshot:
    task_id: str
    history_version: int
    delivered_count: int
    embeddings: tuple[tuple[float, ...], ...] = ()
    delivered_image_ids: tuple[str, ...] = ()
    best_sufficiency_by_group: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _name(self.task_id, "task_id")
        if type(self.history_version) is not int or self.history_version < 0:
            raise ValueError("history_version must be nonnegative")
        if type(self.delivered_count) is not int or self.delivered_count < 0:
            raise ValueError("delivered_count must be nonnegative")
        if self.delivered_count != len(self.delivered_image_ids):
            raise ValueError("delivered_count must equal delivered_image_ids length")
        if len(self.embeddings) != self.delivered_count:
            raise ValueError("embeddings and delivered image IDs must align")


@dataclass(frozen=True, slots=True)
class SemanticCandidateSnapshot:
    """Policy input with evaluator-only labels intentionally excluded."""

    now_ms: float
    jobs: tuple[LatentJobSpec, ...]
    eligible_mask: tuple[bool, ...]
    history_version: int
    delivered_count: int
    queue_capacity: int
    per_ue_served: tuple[tuple[str, int], ...]
    scheduling_headers: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        _finite(self.now_ms, "now_ms", lower=0.0)
        if not isinstance(self.jobs, tuple) or not all(isinstance(item, LatentJobSpec) for item in self.jobs):
            raise ValueError("jobs must be a tuple of LatentJobSpec")
        if len({item.job_id for item in self.jobs}) != len(self.jobs):
            raise ValueError("snapshot has duplicate job IDs")
        if not isinstance(self.eligible_mask, tuple) or len(self.eligible_mask) != len(self.jobs) or not all(type(value) is bool for value in self.eligible_mask):
            raise ValueError("eligible_mask must align with jobs")
        if type(self.history_version) is not int or self.history_version < 0:
            raise ValueError("history_version must be nonnegative")
        if type(self.delivered_count) is not int or self.delivered_count < 0:
            raise ValueError("delivered_count must be nonnegative")
        if type(self.queue_capacity) is not int or self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if not all(isinstance(ue, str) and type(count) is int and count >= 0 for ue, count in self.per_ue_served):
            raise ValueError("per_ue_served must contain (UE ID, nonnegative count) pairs")
        if self.scheduling_headers and (len(self.scheduling_headers) != len(self.jobs)
                or any(len(item) != 24 for item in self.scheduling_headers)):
            raise ValueError("scheduling_headers must contain one 24-byte header per job")

    @property
    def eligible_indices(self) -> tuple[int, ...]:
        return tuple(index for index, eligible in enumerate(self.eligible_mask) if eligible)


@dataclass(frozen=True, slots=True)
class ResourceBudgetSemantic:
    service_rate_units_per_ms: float
    max_service_ms: float | None = None
    resource_unit: str = "simulated_units"

    def __post_init__(self) -> None:
        _finite(self.service_rate_units_per_ms, "service_rate_units_per_ms", lower=1e-12)
        if self.max_service_ms is not None:
            _finite(self.max_service_ms, "max_service_ms", lower=0.0)
        if self.resource_unit not in ("complex_channel_uses", "digital_bits", "simulated_units"):
            raise ValueError("resource_unit is unsupported")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy: str
    now_ms: float
    action_index: int
    score: float | None
    reason: str
    explanation: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _name(self.policy, "policy")
        _finite(self.now_ms, "now_ms", lower=0.0)
        if type(self.action_index) is not int or self.action_index < -1:
            raise ValueError("action_index must be -1 or a candidate index")
        if self.score is not None:
            _finite(self.score, "score")
        _name(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class ServiceOutcome:
    job_id: str
    status: str
    started_ms: float | None
    completed_ms: float | None
    wait_ms: float | None
    deadline_met: bool | None
    transport_success: bool | None
    utility_delta: float = 0.0

    def __post_init__(self) -> None:
        _name(self.job_id, "job_id")
        if self.status not in {"completed_on_time", "completed_late", "failed_transport", "expired_waiting", "overflow_drop", "horizon_censored"}:
            raise ValueError("unsupported service outcome status")
        for name in ("started_ms", "completed_ms", "wait_ms", "utility_delta"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name, lower=0.0)
        if self.started_ms is not None and self.completed_ms is not None and self.completed_ms < self.started_ms:
            raise ValueError("completion cannot precede service start")


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    episode_id: str
    task_id: str
    jobs: tuple[EpisodeJobSpec, ...]
    seed: int
    queue_capacity: int = 25
    horizon_ms: float = 2000.0

    def __post_init__(self) -> None:
        _name(self.episode_id, "episode_id")
        _name(self.task_id, "task_id")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(self.jobs, tuple) or not self.jobs:
            raise ValueError("jobs must be a nonempty tuple")
        if len({job.candidate.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("episode has duplicate job IDs")
        if any(job.candidate.task_id != self.task_id for job in self.jobs):
            raise ValueError("all jobs must share the episode task")
        if type(self.queue_capacity) is not int or self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        _finite(self.horizon_ms, "horizon_ms", lower=1e-9)


@dataclass(frozen=True, slots=True)
class SemanticUtilityBreakdown:
    coverage_gain: float
    sufficiency_factor: float
    delay_factor: float
    utility_delta: float
    cost_penalty: float = 0.0
    expiry_penalty: float = 0.0
    failure_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name in ("coverage_gain", "sufficiency_factor", "delay_factor", "utility_delta", "cost_penalty", "expiry_penalty", "failure_penalty"):
            _finite(getattr(self, name), name, lower=0.0)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    """Numerically safe cosine without an ML framework dependency."""
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
