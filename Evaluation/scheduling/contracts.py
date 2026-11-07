"""Small immutable scheduling contracts. No tensor or model dependencies."""
from dataclasses import dataclass
import math
from typing import Protocol

def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")

@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    image_id: str
    ue_id: str
    arrival_ms: float
    resource_units: int
    priority_class: int = 2
    deadline_ms: float | None = None

    def __post_init__(self):
        for name in ('job_id', 'image_id', 'ue_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        finite_nonnegative(self.arrival_ms, 'arrival_ms')
        if type(self.resource_units) is not int or self.resource_units <= 0:
            raise ValueError('resource_units must be a positive integer')
        if type(self.priority_class) is not int or self.priority_class < 0:
            raise ValueError('priority_class must be a nonnegative integer; lower means higher priority')
        if self.deadline_ms is not None:
            finite_nonnegative(self.deadline_ms, 'deadline_ms')
            if self.deadline_ms < self.arrival_ms:
                raise ValueError('deadline_ms precedes arrival_ms')

@dataclass(frozen=True, slots=True)
class ResourceBudget:
    units: int
    unit: str = 'complex_channel_uses'
    max_jobs: int = 1

    def __post_init__(self):
        if type(self.units) is not int or self.units < 0 or type(self.max_jobs) is not int or self.max_jobs < 0:
            raise ValueError('budget units and max_jobs must be nonnegative integers')
        if self.unit not in ('complex_channel_uses', 'digital_bits'):
            raise ValueError('resource unit must be complex_channel_uses or digital_bits')

@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    now_ms: float
    jobs: tuple[Job, ...]

    def __post_init__(self):
        finite_nonnegative(self.now_ms, 'now_ms')
        if not isinstance(self.jobs, tuple) or not all(isinstance(job, Job) for job in self.jobs):
            raise ValueError('snapshot jobs must be a tuple of Job records')
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError('duplicate job IDs')

    def eligible(self):
        return tuple(job for job in self.jobs if job.arrival_ms <= self.now_ms
                     and (job.deadline_ms is None or job.deadline_ms > self.now_ms))

@dataclass(frozen=True, slots=True)
class Decision:
    policy: str
    now_ms: float
    job_ids: tuple[str, ...]
    used_units: int
    resource_unit: str
    reason: str


class SchedulingPolicy(Protocol):
    name: str

    def reset(self) -> None: ...
    def decide(self, snapshot: QueueSnapshot, budget: ResourceBudget) -> Decision: ...
