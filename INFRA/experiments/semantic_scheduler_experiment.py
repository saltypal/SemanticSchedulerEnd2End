"""Reusable preparation, training and evaluation entry points for notebooks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from time import process_time_ns
from typing import Mapping, Sequence

from INFRA.scheduling.event_simulation import run_episode
from INFRA.data.feature_cache import to_policy_features
from INFRA.data.ladi_manifest import validate_ladi_manifest
from INFRA.scheduling.semantic_contracts import (
    FEATURE_NAMES, EpisodeJobSpec, FeatureValue, LatentJobSpec,
    ResourceBudgetSemantic, SchedulingFeatureVectorV2,
)
from INFRA.semantic_config import SemanticSchedulerConfig
from INFRA.evaluation.semantic import aggregate_episode_metrics
from INFRA.scheduling.streams import (
    DEFAULT_UE_PROFILES, UEProfile, build_episode, generate_episode_suite,
)
from Scheduler.semantic.registry import create_semantic_policy


TASK_ID = "post_disaster_access_and_damage_triage_v1"


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance(repository_root: str | Path) -> dict[str, object]:
    root = Path(repository_root)
    # Kaggle kernels may receive a source archive through a private Kaggle
    # dataset because a private GitHub repository cannot be cloned
    # anonymously.  The bootstrap cell records the exact originating commit
    # in this environment variable and the archive is immutable for the run.
    # Normal local/CI checkouts continue to use Git directly.
    source_commit_override = os.environ.get("SEMANTIC_SOURCE_COMMIT")
    if source_commit_override:
        commit = source_commit_override
        status = ""
    else:
        def git(*args: str) -> str:
            return subprocess.check_output(("git", *args), cwd=root,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    source = sha256()
    for directory in ("SemParam", "Scheduler", "INFRA", "EncDec/integration"):
        for path in sorted((root / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            source.update(path.relative_to(root).as_posix().encode())
            source.update(path.read_bytes())
    return {"git_commit": commit, "git_dirty": bool(status),
            "git_status_sha256": sha256(status.encode()).hexdigest(),
            "source_digest_sha256": source.hexdigest()}


def prepare_feature_cache(
    reviewed_manifest: str | Path, output_dir: str | Path,
    *, include_ladi_predictions: bool = True, device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Path]:
    """Extract CLIP once per reviewed image and optionally predict LADI labels."""
    from SemParam.features.scheduler import SchedulerAnnotation, SchedulerFeatureExtractor
    from SemParam.features.ladi_prediction import (
        LadiPredictionProvider, semantic_scene_features)

    rows = validate_ladi_manifest(reviewed_manifest, require_reviewed=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    extractor = SchedulerFeatureExtractor(device=device, local_files_only=local_files_only)
    classifier = (LadiPredictionProvider(device=device, local_files_only=local_files_only)
                  if include_ladi_predictions else None)
    feature_path = destination / "feature_cache.jsonl"
    prediction_path = destination / "ladi_predictions.json"
    predictions = {}
    import torch
    cuda_measurement = torch.cuda.is_available() and device in {"auto", "cuda"}
    with feature_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            annotation = SchedulerAnnotation(
                task_impact=int(row["task_impact"]), urgency=int(row["urgency"]),
                evidence_sufficiency=int(row["evidence_sufficiency"]),
                annotation_source=row["annotation_source"],
                reviewer=row["reviewer"], rationale=row["rationale"],
            )
            if cuda_measurement:
                torch.cuda.reset_peak_memory_stats()
            cpu_started = process_time_ns()
            extracted = extractor.extract(
                row["image_path"], image_id=row["image_id"],
                annotation=annotation,
                metadata={"source_sha256": row["source_sha256"],
                          "dataset_revision": row["dataset_revision"]},
            )
            process_cpu_ms = (process_time_ns() - cpu_started) / 1_000_000.0
            policy_features = to_policy_features(extracted)
            if classifier is not None:
                classifier_prediction = classifier.predict(row["image_path"])
                probabilities = dict(classifier_prediction.probabilities)
                predictions[row["image_id"]] = probabilities
                scene_importance, interpretation_confidence = semantic_scene_features(
                    probabilities)
                model_version = classifier_prediction.model_repository
                policy_features = replace(
                    policy_features,
                    scene_importance=FeatureValue(
                        scene_importance, True,
                        "ladi_classifier:task_weighted_probability",
                        model_version),
                    interpretation_confidence=FeatureValue(
                        interpretation_confidence, True,
                        "ladi_classifier:one_minus_binary_entropy",
                        model_version),
                )
            cache_row = {
                "image_id": row["image_id"],
                "features": asdict(policy_features),
                "extraction_latency_ms": extracted.extraction_latency_ms,
                "process_cpu_ms": process_cpu_ms,
                "gpu_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if cuda_measurement else None),
                "extractor_version": extracted.extractor_version,
                "source_sha256": row["source_sha256"],
            }
            handle.write(json.dumps(cache_row, sort_keys=True) + "\n")
    prediction_path.write_text(json.dumps(predictions, indent=2, sort_keys=True),
                               encoding="utf-8")
    return {"features": feature_path, "predictions": prediction_path}


def simulated_fixed_payload_costs(reviewed_manifest: str | Path,
                                  output_csv: str | Path, *, units: int = 10) -> Path:
    """Create an explicitly simulated resource fixture for CPU smoke tests."""
    rows = validate_ladi_manifest(reviewed_manifest, require_reviewed=True)
    if units <= 0:
        raise ValueError("Simulated resource units must be positive")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "image_id", "resource_units", "resource_unit", "payload_provenance"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"image_id": row["image_id"],
                             "resource_units": units, "resource_unit": "simulated_units",
                             "payload_provenance": "simulated_fixed"})
    return output


def _load_feature_rows(path: str | Path) -> dict[str, tuple[SchedulingFeatureVectorV2, str]]:
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        raw = record["features"]
        parsed = {
            name: (FeatureValue(**raw[name]) if name in raw else
                   FeatureValue.unavailable("legacy_five_field_cache", "track-a/v2"))
            for name in FEATURE_NAMES
        }
        image_id = record["image_id"]
        if image_id in values:
            raise ValueError(f"Duplicate feature-cache image ID {image_id}")
        values[image_id] = (
            SchedulingFeatureVectorV2(
                **parsed, embedding=tuple(raw["embedding"]), schema_id=raw["schema_id"]),
            str(record.get("source_sha256", "")),
        )
    return values


def load_catalog(reviewed_manifest: str | Path, feature_cache: str | Path,
                 payload_costs: str | Path) -> tuple[EpisodeJobSpec, ...]:
    """Join reviewed image facts with cached extraction and real/simulated cost."""
    rows = validate_ladi_manifest(reviewed_manifest, require_reviewed=True)
    features = _load_feature_rows(feature_cache)
    with Path(payload_costs).open(newline="", encoding="utf-8") as handle:
        cost_rows = {}
        for row in csv.DictReader(handle):
            if row["image_id"] in cost_rows:
                raise ValueError(f"Duplicate payload-cost image ID {row['image_id']}")
            cost_rows[row["image_id"]] = row
    if set(features) != {row["image_id"] for row in rows} or set(cost_rows) != set(features):
        raise ValueError("Feature, payload and reviewed-manifest image IDs differ")
    resource_units = {row["resource_unit"] for row in cost_rows.values()}
    if len(resource_units) != 1:
        raise ValueError("Cannot mix payload resource units within one experiment")
    if resource_units == {"complex_channel_uses"}:
        checkpoint_hashes = {row.get("checkpoint_sha256") for row in cost_rows.values()}
        if len(checkpoint_hashes) != 1 or None in checkpoint_hashes or "" in checkpoint_hashes:
            raise ValueError("Measured payloads require one checkpoint SHA-256")
    catalogue = []
    for row in rows:
        image_id = row["image_id"]
        if image_id not in features or image_id not in cost_rows:
            raise ValueError(f"Missing feature or payload cache for {image_id}")
        cost = cost_rows[image_id]
        if cost.get("resource_unit") == "complex_channel_uses":
            units = int(cost["channel_uses"])
            provenance = f"swinjscc:{cost['checkpoint_sha256']}"
        elif cost.get("resource_unit") == "simulated_units":
            units = int(cost["resource_units"])
            provenance = "simulated_fixed"
        else:
            raise ValueError(f"Unsupported or mixed payload unit for {image_id}")
        policy_features, source_sha = features[image_id]
        if source_sha != row["source_sha256"]:
            raise ValueError(f"Feature-cache source hash differs for {image_id}")
        candidate = LatentJobSpec(
            job_id=image_id, image_id=image_id, ue_id="unassigned",
            task_id=TASK_ID, payload_id=image_id, arrival_ms=0.0,
            deadline_ms=None, resource_units=units,
            resource_unit=cost["resource_unit"], features=policy_features,
            payload_provenance=provenance,
        )
        catalogue.append(EpisodeJobSpec(candidate, int(row["base_priority"]),
                                        row["evidence_group_id"]))
    return tuple(catalogue)


def build_suites_from_manifest(catalogue: Sequence[EpisodeJobSpec],
                               reviewed_manifest: str | Path,
                               *, config: SemanticSchedulerConfig | None = None,
                               ue_profiles: Sequence[UEProfile] = DEFAULT_UE_PROFILES) -> dict[str, tuple]:
    """Freeze train, validation, two test tracks, and a paired faculty stream."""
    rows = validate_ladi_manifest(reviewed_manifest, require_reviewed=True)
    if config is not None and (config.task_id != TASK_ID or len(ue_profiles) != config.ue_count):
        raise ValueError("Task or UE profiles differ from the locked scheduler config")
    jobs_per_episode = len(catalogue) if config is None else config.jobs_per_episode
    queue_capacity = 25 if config is None else config.queue_capacity
    horizon_ms = 2000.0 if config is None else config.horizon_ms
    deadline_mode = "independent_network_sla" if config is None else config.deadline_mode
    train_count = 300 if config is None else config.train_episodes_per_seed
    validation_count = 30 if config is None else config.validation_episodes
    test_count = 100 if config is None else config.test_episodes_per_track
    split_by_image = {row["image_id"]: row["split"] for row in rows}
    by_split: dict[str, list[EpisodeJobSpec]] = defaultdict(list)
    for item in catalogue:
        by_split[split_by_image[item.candidate.image_id]].append(item)
    expected_split = ({"train": 156, "validation": 52, "test": 52}
                      if len(rows) == 260 else
                      {"train": 36, "validation": 12, "test": 12})
    if {name: len(by_split[name]) for name in ("train", "validation", "test")} != expected_split:
        raise ValueError("Catalogue split differs from reviewed manifest")
    train = tuple(by_split["train"])
    validation = tuple(by_split["validation"])
    test = tuple(by_split["test"])
    train_episodes = {
        seed: generate_episode_suite(train, task_id=TASK_ID,
                                     prefix=f"train-s{seed}", episode_count=train_count,
                                     jobs_per_episode=jobs_per_episode, seed=seed,
                                     ue_profiles=ue_profiles,
                                     queue_capacity=queue_capacity, horizon_ms=horizon_ms,
                                     deadline_mode=deadline_mode, allow_reuse=True)
        for seed in (42, 43, 44)
    }
    validation_episodes = generate_episode_suite(
        validation, task_id=TASK_ID, prefix="validation", episode_count=validation_count,
        # Keep model-selection episodes structurally identical to deployment
        # episodes. The validation catalogue has 12 images, so controlled
        # reuse is required to reach the normal 25-job episode size. Without
        # this, the DQN is selected on a different queue/action distribution
        # from the one used for training and test evaluation.
        jobs_per_episode=jobs_per_episode, seed=1001, allow_reuse=True,
        ue_profiles=ue_profiles,
        queue_capacity=queue_capacity, horizon_ms=horizon_ms,
        deadline_mode=deadline_mode)
    image_disjoint = generate_episode_suite(
        test, task_id=TASK_ID, prefix="image-disjoint", episode_count=test_count,
        jobs_per_episode=jobs_per_episode, seed=2001, allow_reuse=True,
        ue_profiles=ue_profiles, queue_capacity=queue_capacity, horizon_ms=horizon_ms,
        deadline_mode=deadline_mode)
    closed_catalogue = generate_episode_suite(
        train, task_id=TASK_ID, prefix="closed-catalogue", episode_count=test_count,
        jobs_per_episode=jobs_per_episode, seed=3001,
        ue_profiles=ue_profiles, queue_capacity=queue_capacity, horizon_ms=horizon_ms,
        allow_reuse=True,
        deadline_mode=deadline_mode)
    distinct_test = generate_episode_suite(
        test, task_id=TASK_ID, prefix="image-disjoint-distinct",
        episode_count=test_count, jobs_per_episode=len(test), seed=4001,
        ue_profiles=ue_profiles, queue_capacity=queue_capacity,
        horizon_ms=horizon_ms, deadline_mode=deadline_mode)

    def all_arrive_together(episode):
        return replace(
            episode, episode_id=f"static-{episode.episode_id}",
            jobs=tuple(replace(item, candidate=replace(
                item.candidate, arrival_ms=0.0,
                deadline_ms=horizon_ms - 100.0)) for item in episode.jobs))

    static_test = tuple(all_arrive_together(item) for item in distinct_test)
    static_closed = tuple(all_arrive_together(item) for item in closed_catalogue)
    demo_templates = list(catalogue)
    demo = build_episode(demo_templates, episode_id="faculty-demo", task_id=TASK_ID,
                         seed=5501, ue_profiles=ue_profiles,
                         queue_capacity=queue_capacity, horizon_ms=horizon_ms,
                         deadline_mode=deadline_mode)
    return {
        "train_by_seed": train_episodes,
        "validation": validation_episodes,
        "image_disjoint": image_disjoint,
        "image_disjoint_distinct": distinct_test,
        "static_image_disjoint": static_test,
        "static_closed_catalogue": static_closed,
        "closed_catalogue": closed_catalogue,
        "faculty_demo": (demo,),
    }


def evaluate_policy_suite(
    episodes: Sequence, budget: ResourceBudgetSemantic,
    *, priority_by_image: Mapping[str, int],
    predictions_by_image: Mapping[str, Mapping[str, float]],
    image_priority_predictions_by_image: Mapping[str, Mapping[str, float]] | None = None,
    model_artifacts: Mapping[str, str | Path],
    policy_names: Sequence[str] | None = None,
    visible_features: tuple[str, ...] | None = None,
    cost_coefficient: float = 0.1,
    retain_results: bool = False,
    enable_delivered_history: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run all available policies on identical episode objects and outcomes."""
    default_names = (
        "fifo", "ue_round_robin", "fixed_priority", "edf", "priority_edf",
        "sjf", "aging", "semantic_greedy", "ladi_classifier_priority",
        "clip_image_priority", "clip_image_priority_edf",
        "semantic_weighted", "semantic_lexicographic",
        "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn",
        "privileged_marginal_utility",
    )
    names = tuple(policy_names) if policy_names is not None else default_names
    results, metrics = [], []
    for episode in episodes:
        oracle_facts = {
            item.candidate.image_id: {
                "base_priority": item.base_priority,
                "evidence_group_id": item.evidence_group_id,
                "evidence_sufficiency": item.candidate.features.evidence_sufficiency.value,
                "urgency": item.candidate.features.urgency.value,
            } for item in episode.jobs
        }
        for name in names:
            artifact = model_artifacts.get(name)
            policy = create_semantic_policy(
                name, priority_by_image=priority_by_image,
                predictions_by_image=predictions_by_image,
                image_priority_predictions_by_image=image_priority_predictions_by_image,
                oracle_facts_by_image=oracle_facts,
                model_artifact=artifact,
            )
            if visible_features is None:
                result = run_episode(episode, policy, budget,
                                     cost_coefficient=cost_coefficient,
                                     enable_delivered_history=enable_delivered_history)
            else:
                result = run_episode(episode, policy, budget,
                                     visible_features=visible_features,
                                     cost_coefficient=cost_coefficient,
                                     enable_delivered_history=enable_delivered_history)
            if retain_results:
                results.append(result)
            metrics.append({"episode_id": episode.episode_id, "policy": name,
                            **aggregate_episode_metrics(result)})
    return results, metrics


def run_provenance(repository_root: str | Path, reviewed_manifest: str | Path,
                   *, config: Mapping[str, object], checkpoint_sha256: str | None = None) -> dict[str, object]:
    import importlib.metadata as metadata
    import torch

    packages = ("numpy", "torch", "transformers", "datasets", "scikit-learn",
                "Pillow", "pandas")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {
        **git_provenance(repository_root),
        "manifest_sha256": _sha256_file(reviewed_manifest),
        "task_id": TASK_ID,
        "task_text_sha256": sha256((
            "Prioritize imagery that most improves assessment of immediate threats "
            "to people, access routes, and affected structures.").encode()).hexdigest(),
        "feature_schema_version": 2,
        "header_schema_id": 0x0201,
        "header_version": 2, "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "packages": versions,
    }
