"""Deterministic UE stream construction for semantic scheduling experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import random
from collections import defaultdict
from typing import Iterable, Sequence

from Evaluation.scheduling.semantic_contracts import EpisodeJobSpec, EpisodeSpec, LatentJobSpec


@dataclass(frozen=True, slots=True)
class UEProfile:
    ue_id: str
    kind: str
    interval_ms: float
    jitter_ms: float = 0.0
    burst_size: int = 1

    def __post_init__(self) -> None:
        if not self.ue_id or not self.kind:
            raise ValueError("UE profile needs an ID and kind")
        if self.kind not in {"periodic", "burst", "exponential"}:
            raise ValueError("UE profile kind is unsupported")
        if self.interval_ms <= 0 or self.jitter_ms < 0 or self.burst_size <= 0:
            raise ValueError("UE timing values are invalid")


DEFAULT_UE_PROFILES = (
    UEProfile("ue_drone", "periodic", 20.0, 3.0),
    UEProfile("ue_burst_camera", "burst", 60.0, 0.0, 3),
    UEProfile("ue_responder", "exponential", 25.0),
    UEProfile("ue_survey", "periodic", 40.0, 10.0),
    UEProfile("ue_relay", "burst", 45.0, 0.0, 2),
)


def deadline_for_urgency(urgency: int, rng: random.Random) -> float:
    ranges = {3: (40, 80), 2: (80, 140), 1: (140, 220), 0: (300, 500)}
    if urgency not in ranges:
        raise ValueError("urgency must be in [0, 3]")
    lower, upper = ranges[urgency]
    return float(rng.randint(lower, upper))


def deadline_for_network_sla(rng: random.Random) -> float:
    """Generate a packet deadline independently of semantic task urgency."""
    return float(rng.randint(70, 170))


def _profile_arrivals(profile: UEProfile, count: int, rng: random.Random) -> list[float]:
    values: list[float] = []
    current = 0.0
    for ordinal in range(count):
        if profile.kind == "exponential":
            current += rng.expovariate(1.0 / profile.interval_ms)
        elif profile.kind == "burst":
            current = (ordinal // profile.burst_size) * profile.interval_ms
        else:
            current += profile.interval_ms + rng.uniform(-profile.jitter_ms, profile.jitter_ms)
        values.append(max(0.0, current))
    return values


def _urgency(job: EpisodeJobSpec) -> int:
    field = job.candidate.features.urgency
    if not field.available or field.value is None:
        return 0
    return int(round(field.value))


def build_episode(
    templates: Sequence[EpisodeJobSpec],
    *,
    episode_id: str,
    task_id: str,
    seed: int,
    ue_profiles: Sequence[UEProfile] = DEFAULT_UE_PROFILES,
    queue_capacity: int = 25,
    horizon_ms: float = 2000.0,
    require_distinct_images: bool = True,
    deadline_mode: str = "independent_network_sla",
) -> EpisodeSpec:
    """Assign selected templates to balanced UE traffic without label leakage.

    The caller chooses the templates.  This makes split logic auditable while
    this function owns only seeded arrival, UE and deadline generation.
    """
    if not templates:
        raise ValueError("at least one template is required")
    if not ue_profiles:
        raise ValueError("at least one UE profile is required")
    if require_distinct_images and len({item.candidate.image_id for item in templates}) != len(templates):
        raise ValueError("episode requires distinct image IDs")
    if deadline_mode not in {"independent_network_sla", "task_coupled"}:
        raise ValueError("Unsupported deadline_mode")
    rng = random.Random(seed)
    assignments: dict[str, list[EpisodeJobSpec]] = {profile.ue_id: [] for profile in ue_profiles}
    by_stratum: dict[str, list[EpisodeJobSpec]] = defaultdict(list)
    for template in templates:
        # Evidence-group IDs are task-specific evaluator facts. They are used
        # only to balance the experimental stream, never passed to a policy.
        stratum = template.evidence_group_id.rsplit("-group", 1)[0]
        by_stratum[stratum].append(template)
    target_capacity = (len(templates) + len(ue_profiles) - 1) // len(ue_profiles)
    stratum_counts: dict[tuple[str, str], int] = defaultdict(int)
    for stratum in sorted(by_stratum):
        items = by_stratum[stratum]
        rng.shuffle(items)
        for template in items:
            available = [profile.ue_id for profile in ue_profiles
                         if len(assignments[profile.ue_id]) < target_capacity]
            rng.shuffle(available)
            chosen = min(available, key=lambda ue: (
                stratum_counts[(stratum, ue)], len(assignments[ue])))
            assignments[chosen].append(template)
            stratum_counts[(stratum, chosen)] += 1
    generated: list[EpisodeJobSpec] = []
    for profile in ue_profiles:
        local_jobs = assignments[profile.ue_id]
        # Balancing visits strata in sorted order. Randomize transmission
        # order separately so class names do not determine arrival order.
        rng.shuffle(local_jobs)
        arrivals = _profile_arrivals(profile, len(local_jobs), rng)
        for sequence, (template, arrival) in enumerate(zip(local_jobs, arrivals)):
            candidate = template.candidate
            deadline_offset = (deadline_for_network_sla(rng)
                               if deadline_mode == "independent_network_sla"
                               else deadline_for_urgency(_urgency(template), rng))
            deadline = arrival + deadline_offset
            generated_candidate = replace(
                candidate,
                job_id=f"{episode_id}:{profile.ue_id}:{sequence}:{candidate.image_id}",
                ue_id=profile.ue_id,
                task_id=task_id,
                arrival_ms=round(arrival, 6),
                deadline_ms=round(deadline, 6),
            )
            generated.append(replace(template, candidate=generated_candidate))
    generated.sort(key=lambda item: (item.candidate.arrival_ms, item.candidate.job_id))
    return EpisodeSpec(episode_id, task_id, tuple(generated), seed, queue_capacity, horizon_ms)


def generate_episode_suite(
    catalogue: Sequence[EpisodeJobSpec],
    *,
    task_id: str,
    prefix: str,
    episode_count: int,
    jobs_per_episode: int = 25,
    seed: int = 42,
    ue_profiles: Sequence[UEProfile] = DEFAULT_UE_PROFILES,
    queue_capacity: int = 25,
    horizon_ms: float = 2000.0,
    allow_reuse: bool = False,
    deadline_mode: str = "independent_network_sla",
) -> tuple[EpisodeSpec, ...]:
    """Create reproducible streams from a caller-provided split catalogue."""
    if episode_count <= 0 or jobs_per_episode <= 0:
        raise ValueError("episode_count and jobs_per_episode must be positive")
    if not catalogue:
        raise ValueError("catalogue is empty")
    if not allow_reuse and len(catalogue) < jobs_per_episode:
        raise ValueError("catalogue has fewer images than jobs_per_episode")
    root_rng = random.Random(seed)
    episodes = []
    for index in range(episode_count):
        episode_seed = root_rng.randrange(0, 2**31)
        selector = random.Random(episode_seed)
        if allow_reuse:
            selected = [selector.choice(catalogue) for _ in range(jobs_per_episode)]
            # Keep the source image ID stable so receiver history recognizes a
            # repeated image. build_episode gives every arrival a unique job ID.
        else:
            selected = selector.sample(list(catalogue), jobs_per_episode)
        episodes.append(build_episode(
            selected,
            episode_id=f"{prefix}-{index:04d}",
            task_id=task_id,
            seed=episode_seed,
            ue_profiles=ue_profiles,
            queue_capacity=queue_capacity,
            horizon_ms=horizon_ms,
            require_distinct_images=not allow_reuse,
            deadline_mode=deadline_mode,
        ))
    return tuple(episodes)


def with_transport_probability(episode: EpisodeSpec, success_probability: float) -> EpisodeSpec:
    """Freeze policy-independent per-job outcomes from a stable hash tape."""
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must be in [0, 1]")
    jobs = []
    for item in episode.jobs:
        token = f"{episode.seed}:{item.candidate.job_id}:transport-v1".encode()
        draw = int.from_bytes(sha256(token).digest()[:8], "big") / 2**64
        jobs.append(replace(item, transport_success=draw < success_probability))
    return replace(episode, jobs=tuple(jobs))


def with_task_coupled_deadlines(episode: EpisodeSpec) -> EpisodeSpec:
    """Sensitivity case: preserve arrivals but derive deadlines from urgency."""
    jobs = []
    for item in episode.jobs:
        token = f"{episode.seed}:{item.candidate.job_id}:task-deadline-v1".encode()
        seed = int.from_bytes(sha256(token).digest()[:8], "big")
        offset = deadline_for_urgency(_urgency(item), random.Random(seed))
        candidate = replace(
            item.candidate,
            deadline_ms=round(item.candidate.arrival_ms + offset, 6),
        )
        jobs.append(replace(item, candidate=candidate))
    return replace(episode, episode_id=f"task-coupled-{episode.episode_id}",
                   jobs=tuple(jobs))
