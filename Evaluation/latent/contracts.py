"""Device-independent contracts for scheduling complete cached image latents."""

from __future__ import annotations

from dataclasses import dataclass
import math


def finite(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


@dataclass(frozen=True, slots=True)
class LatentJob:
    job_id: str
    image_id: str
    task_id: str
    payload_id: str
    resource_units: int
    resource_unit: str
    payload_provenance: str
    priority_probabilities: tuple[float, ...] | None = None
    priority_class: int | None = None
    priority_source: str = "unavailable"
    context_relevance: float | None = None
    task_importance: float | None = None
    embedding: tuple[float, ...] = ()
    arrival_ms: float = 0.0
    deadline_ms: float | None = None

    def __post_init__(self):
        for name in ("job_id", "image_id", "task_id", "payload_id", "payload_provenance", "priority_source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if type(self.resource_units) is not int or self.resource_units <= 0:
            raise ValueError("resource_units must be a positive integer")
        if self.resource_unit not in {"complex_channel_uses", "digital_bits", "simulated_units"}:
            raise ValueError("Unsupported resource unit")
        finite(self.arrival_ms, "arrival_ms", minimum=0)
        if self.deadline_ms is not None:
            finite(self.deadline_ms, "deadline_ms", minimum=self.arrival_ms)
        if self.context_relevance is not None:
            finite(self.context_relevance, "context_relevance", minimum=-1, maximum=1)
        if self.task_importance is not None:
            finite(self.task_importance, "task_importance", minimum=0, maximum=3)
        if self.priority_class is not None and (type(self.priority_class) is not int or self.priority_class not in range(4)):
            raise ValueError("priority_class must be 0..3")
        if self.priority_probabilities is not None:
            if not isinstance(self.priority_probabilities, tuple) or len(self.priority_probabilities) != 4:
                raise ValueError("Four immutable priority probabilities are required")
            for value in self.priority_probabilities:
                finite(value, "priority probability", minimum=0, maximum=1)
            if not math.isclose(sum(self.priority_probabilities), 1.0, abs_tol=1e-6):
                raise ValueError("Priority probabilities must sum to one")
            if self.priority_class is not None:
                raise ValueError("Supply probabilities or class, not both")
        if (self.priority_class is not None or self.priority_probabilities is not None) and self.priority_source == "unavailable":
            raise ValueError("Priority input must declare its source")
        if not isinstance(self.embedding, tuple):
            raise ValueError("embedding must be immutable")
        for value in self.embedding:
            finite(value, "embedding")
        if self.embedding and not any(value != 0 for value in self.embedding):
            raise ValueError("Embedding must have a nonzero norm")

    @property
    def predicted_priority(self) -> int | None:
        if self.priority_probabilities is None:
            return self.priority_class
        cumulative = 0.0
        for label, probability in enumerate(self.priority_probabilities):
            cumulative += probability
            if cumulative >= 0.5:
                return label
        return 3


@dataclass(frozen=True, slots=True)
class LatentCandidate:
    job_id: str
    image_id: str
    payload_id: str
    input_index: int
    priority_class: int | None
    priority_source: str
    resource_units: int
    resource_unit: str
    service_ms: float
    deadline_ms: float | None
    slack_ms: float | None
    context_relevance: float | None
    task_importance: float | None
    novelty: float | None
    eligible: bool


@dataclass(frozen=True, slots=True)
class LatentSnapshot:
    task_id: str
    now_ms: float
    history_version: int
    jobs: tuple[LatentCandidate, ...]


@dataclass(frozen=True, slots=True)
class DeliveryEvidence:
    """Evaluator-only outcome. Never enters a policy snapshot.

    Missing task quality is unavailable, even if transport succeeded.
    receiver_embedding, when available, describes the reconstructed image.
    """
    job_id: str
    transport_success: bool
    provenance: str
    task_quality: float | None = None
    receiver_embedding: tuple[float, ...] = ()
    evidence_group_id: str | None = None

    def __post_init__(self):
        if type(self.transport_success) is not bool:
            raise ValueError("transport_success must be boolean")
        if not self.job_id or not self.provenance:
            raise ValueError("Delivery evidence requires identity and provenance")
        if self.task_quality is not None:
            finite(self.task_quality, "task_quality", minimum=0, maximum=1)
        if not self.transport_success and (self.task_quality is not None or self.receiver_embedding):
            raise ValueError("Failed delivery cannot have task quality or receiver embedding")
        if not isinstance(self.receiver_embedding, tuple):
            raise ValueError("receiver_embedding must be immutable")
        for value in self.receiver_embedding:
            finite(value, "receiver_embedding")
        if self.receiver_embedding and not any(self.receiver_embedding):
            raise ValueError("Receiver embedding must have nonzero norm")


@dataclass(frozen=True, slots=True)
class LatentScheduleConfig:
    service_rate_units_per_ms: float
    resource_unit: str
    horizon_ms: float
    mode: str = "static"
    history_mode: str = "confirmed"
    service_quantum_ms: float = 0.0

    def __post_init__(self):
        finite(self.service_rate_units_per_ms, "service rate", minimum=1e-12)
        finite(self.horizon_ms, "horizon", minimum=1e-12)
        finite(self.service_quantum_ms, "service quantum", minimum=0)
        if self.mode not in {"static", "dynamic"}:
            raise ValueError("mode must be static or dynamic")
        if self.history_mode not in {"confirmed", "frozen"}:
            raise ValueError("history_mode must be confirmed or frozen")
        if self.resource_unit not in {"complex_channel_uses", "digital_bits", "simulated_units"}:
            raise ValueError("Unsupported resource unit")

    def service_ms(self, job: LatentJob) -> float:
        value = job.resource_units / self.service_rate_units_per_ms
        if self.service_quantum_ms:
            value = math.ceil(value / self.service_quantum_ms) * self.service_quantum_ms
        return value
