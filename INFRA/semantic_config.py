"""Strict defaults for the locked LADI-60 whole-latent experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from INFRA.semantic_streams import UEProfile


@dataclass(frozen=True, slots=True)
class SemanticSchedulerConfig:
    task_id: str
    image_count: int
    ue_count: int
    jobs_per_episode: int
    queue_capacity: int
    horizon_ms: float
    train_episodes_per_seed: int
    validation_episodes: int
    test_episodes_per_track: int
    primary_transport_success_probability: float
    robustness_transport_success_probability: float
    deadline_mode: str = "independent_network_sla"

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        for name in ("image_count", "ue_count", "jobs_per_episode",
                     "queue_capacity", "train_episodes_per_seed",
                     "validation_episodes", "test_episodes_per_track"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_count != 60 or self.ue_count != 5 or self.jobs_per_episode != 60:
            raise ValueError("This run requires the locked 60-image, five-UE, 60-job protocol")
        if self.queue_capacity > self.jobs_per_episode:
            raise ValueError("Queue capacity cannot exceed jobs per episode")
        if self.horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        if self.deadline_mode not in {"independent_network_sla", "task_coupled"}:
            raise ValueError("deadline_mode must be independent_network_sla or task_coupled")
        for name in ("primary_transport_success_probability",
                     "robustness_transport_success_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


def load_scheduler_config(path: str | Path) -> SemanticSchedulerConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("semantic scheduler config must be a YAML mapping")
    return SemanticSchedulerConfig(**raw)


def load_ue_profiles(path: str | Path) -> tuple[UEProfile, ...]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"profiles"} or not isinstance(raw["profiles"], list):
        raise ValueError("UE config must contain only a profiles list")
    profiles = tuple(UEProfile(**row) for row in raw["profiles"])
    if len(profiles) != 5 or len({item.ue_id for item in profiles}) != 5:
        raise ValueError("The demonstration needs five distinct UEs")
    return profiles
