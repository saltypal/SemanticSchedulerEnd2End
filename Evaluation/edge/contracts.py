"""Shared contract between the UE-side tables, the simulator and every scheduling policy.

Only fields in ``Header`` are visible to a policy. ``Truth`` is evaluator-only and must never be
passed to a non-genie policy. All times are seconds.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

TIERS = ("Critical", "High", "Medium", "Low")


@dataclass(frozen=True)
class Header:
    """What travels in the 64-byte header (decoded form)."""
    capture_time: float
    arrival_time: float
    payload_bytes: int
    mission: int
    codec_id: int
    m_index: int
    tier_probs: Tuple[float, float, float, float]   # calibrated SPI posterior (C, H, M, L)
    relevance: float
    sem_key: int                                     # 64-bit SimHash
    perc_key: int                                    # 64-bit pHash
    sts: Tuple[float, ...] = ()                      # Semantic Token Summary (13 values)


@dataclass(frozen=True)
class Truth:
    """Evaluator-only ground truth for one image."""
    tier: int                  # index into TIERS (human-label derived)
    fidelity: int              # u: judge(reconstruction) == judge(original)
    dup_group: Optional[int]   # images sharing a group are near-duplicates; None if unique
    event: str = ""
    source: str = ""


@dataclass(frozen=True)
class Job:
    id: int
    header: Header
    truth: Truth


@dataclass(frozen=True)
class QueuedJob:
    """A job as a policy sees it: header, current age and receiver-side estimates."""
    id: int
    header: Header
    age: float                 # now - capture_time
    dup_prob: float = 0.0      # receiver-side P(duplicate of an already delivered image)
    u_hat: float = 1.0         # receiver-side predicted fidelity


class LatencyProfile:
    """Empirical decode-time distribution per batch size (seconds, measured on a T4)."""

    def __init__(self, samples):
        if not samples:
            raise ValueError("latency samples required")
        self.samples = {int(b): np.sort(np.asarray(v, dtype=float)) for b, v in samples.items()}
        if any(len(v) == 0 or np.any(v <= 0) for v in self.samples.values()):
            raise ValueError("latency samples must be positive and non-empty")
        self.batch_sizes = tuple(sorted(self.samples))

    def _s(self, b):
        if b not in self.samples:
            raise KeyError("no latency samples for batch size %d" % b)
        return self.samples[b]

    def mean(self, b):
        return float(self._s(b).mean())

    def quantile(self, b, q):
        return float(np.quantile(self._s(b), q))

    def p95(self, b):
        return self.quantile(b, 0.95)

    def cdf(self, b, t):
        """P(S_b <= t)."""
        s = self._s(b)
        return float(np.searchsorted(s, t, side="right")) / len(s)

    def sample(self, b, rng):
        s = self._s(b)
        return float(s[rng.integers(len(s))])


@dataclass(frozen=True)
class ValueSpec:
    """Tier deadlines, weights and targets from config/edge/sim.yaml."""
    soft_deadline: Tuple[float, ...]
    hard_factor: float
    weights: Tuple[float, ...]
    eta: Tuple[float, ...]
    zeta_critical_copy: float = 0.25

    @classmethod
    def from_config(cls, cfg):
        return cls(tuple(cfg["soft_deadline_s"]), float(cfg["hard_deadline_factor"]),
                   tuple(cfg["weights"]), tuple(cfg["eta_timely"]), float(cfg["zeta_critical_copy"]))

    def hard_deadline(self, k):
        return self.hard_factor * self.soft_deadline[k]

    def g(self, k, delay):
        """Time-utility: 1 up to D_k, linear to 0 at H_k, 0 after."""
        d, h = self.soft_deadline[k], self.hard_deadline(k)
        if delay <= d:
            return 1.0
        if delay >= h:
            return 0.0
        return (h - delay) / (h - d)


@dataclass(frozen=True)
class State:
    """Everything a policy may look at when a worker frees up."""
    now: float
    queue: Tuple[QueuedJob, ...]
    free_workers: int
    workers: int
    latency: LatencyProfile
    spec: ValueSpec
    max_batch: int
    debts: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)   # per-tier debt queues owned by the env
    arrival_rate: float = 0.0                          # smoothed recent arrivals per second
    mission_served: Tuple[Tuple[int, float], ...] = () # (mission id, recent served value)
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    batch: Tuple[int, ...] = ()     # job ids to decode together (empty = stay idle)
    purge: Tuple[int, ...] = ()     # job ids to drop without decoding


class Policy(ABC):
    """Scheduling policy. ``select`` is called each time a worker is free and the queue is non-empty."""
    name = "policy"
    uses_truth = False   # only genie policies may set this True

    def reset(self, spec: ValueSpec, rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng

    @abstractmethod
    def select(self, state: State) -> Decision:
        ...

    def observe(self, event: str, payload: dict) -> None:
        """Optional hook for arrivals, completions and expiries (used by debt-queue policies)."""


def top_batch(ids_in_order: Sequence[int], state: State, b: Optional[int] = None) -> Tuple[int, ...]:
    """Take up to ``b`` (default: largest profiled batch <= max_batch) ids in the given order."""
    if b is None:
        b = max([x for x in state.latency.batch_sizes if x <= state.max_batch] or [1])
    return tuple(ids_in_order[:min(b, len(ids_in_order))])
