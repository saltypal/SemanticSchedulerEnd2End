"""Train a separate DQN for dynamic prescribed-class order.

This does not replace the existing semantic-utility checkpoint. The class
rubric supplies evaluator-only reward and validation; policy input stays the
same seven observable parameters plus queue state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Evaluation.class_priority_profiles import PRESENTATION_UE_PROFILES, validate_manifest_rubric
from Evaluation.experiments.semantic_scheduler_experiment import TASK_ID, load_catalog
from Evaluation.data.ladi_manifest import validate_ladi_manifest
from Scheduler.training.dqn_training import DQNConfig, train_dqn
from Evaluation.semantic_config import load_scheduler_config, load_ue_profiles
from Evaluation.scheduling.semantic_contracts import ResourceBudgetSemantic
from Evaluation.scheduling.streams import generate_episode_suite
from evaluate_downloaded_scheduler import DEFAULT_MANIFEST, DEFAULT_RUN, check_manifest_match, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=2400.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    run_root = args.run_root.resolve()
    manifest_path = args.manifest.resolve()
    source_run = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    check_manifest_match(manifest_path, run_root / "dataset_manifest.csv")
    for relative, expected in source_run["artifacts_sha256"].items():
        artifact = run_root / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise ValueError(f"Training artifact missing or changed: {relative}")

    reviewed = validate_ladi_manifest(manifest_path, require_reviewed=True)
    validate_manifest_rubric(reviewed)
    configured_profiles = load_ue_profiles(ROOT / "Evaluation/config/ue_profiles.yaml")
    if tuple(configured_profiles) != PRESENTATION_UE_PROFILES:
        raise ValueError("Python UE profiles differ from the active YAML configuration")
    config = load_scheduler_config(ROOT / "Evaluation/config/semantic_scheduler_dynamic260.yaml")
    catalogue = load_catalog(manifest_path, run_root / "feature_cache.jsonl",
                             run_root / "payload_costs.csv")
    split = {row["image_id"]: row["split"] for row in reviewed}
    train_catalogue = tuple(item for item in catalogue
                            if split[item.candidate.image_id] == "train")
    validation_catalogue = tuple(item for item in catalogue
                                 if split[item.candidate.image_id] == "validation")
    if len(train_catalogue) != 156 or len(validation_catalogue) != 52:
        raise ValueError("Unexpected 260-image train/validation split")
    train_episodes = generate_episode_suite(
        train_catalogue, task_id=TASK_ID, prefix=f"class-align-train-s{args.seed}",
        episode_count=args.episodes, jobs_per_episode=52, seed=args.seed,
        ue_profiles=PRESENTATION_UE_PROFILES, queue_capacity=config.queue_capacity,
        horizon_ms=config.horizon_ms, deadline_mode=config.deadline_mode)
    validation_episodes = generate_episode_suite(
        validation_catalogue, task_id=TASK_ID, prefix="class-align-validation",
        episode_count=30, jobs_per_episode=52, seed=3001,
        ue_profiles=PRESENTATION_UE_PROFILES, queue_capacity=config.queue_capacity,
        horizon_ms=config.horizon_ms, deadline_mode=config.deadline_mode)
    output = (args.output.resolve() if args.output else
              ROOT / "results/scheduler-class-alignment" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    budget = ResourceBudgetSemantic(source_run["service_rate_units_per_ms"],
                                    resource_unit=source_run["resource_unit"])
    settings = DQNConfig(
        objective="class_priority_alignment", episodes=args.episodes,
        max_decisions=60, max_seconds=args.max_seconds,
    )
    provenance = {
        "source_training_run": source_run["run_id"],
        "source_commit": source_run["git_commit"],
        "manifest_sha256": file_sha256(manifest_path),
        "feature_cache_sha256": file_sha256(run_root / "feature_cache.jsonl"),
        "resource_unit": source_run["resource_unit"],
        "deadline_mode": config.deadline_mode,
        "validation_stream_seed": 3001,
        "class_rubric": "Evaluation.class_priority_profiles.CLASS_RUBRIC",
        "note": "Rubric-assisted class-order objective; labels hidden from model state.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_spec.json").write_text(
        json.dumps({"config": asdict(settings),
                    "provenance": provenance}, indent=2), encoding="utf-8")
    result = train_dqn(
        train_episodes, validation_episodes, budget, output,
        seed=args.seed, config=settings, provenance=provenance,
        device=args.device, enable_delivered_history=True,
    )
    print(json.dumps({"output": str(output), "completed_episodes": result["completed_episodes"],
                      "best_validation_priority_match_rate":
                      result["best_validation_priority_match_rate"]}, indent=2))


if __name__ == "__main__":
    main()
