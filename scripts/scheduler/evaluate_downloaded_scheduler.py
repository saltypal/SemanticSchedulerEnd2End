"""Evaluate downloaded semantic scheduler models locally without retraining.

The same frozen held-out episodes are replayed for every policy. The script
appends each completed policy/episode row immediately so interrupted work can
be inspected, then writes paired comparisons and a presentation plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "downloaded/train-final-cli/outputs/20260929T021341Z-d3338ef4"
DEFAULT_MANIFEST = ROOT / "results/ladi-curation/v2/ladi260_semantic_scheduler_v2.csv"
POLICIES = (
    "fifo", "ue_round_robin", "fixed_priority", "edf", "priority_edf",
    "sjf", "aging", "semantic_greedy", "semantic_weighted",
    "semantic_lexicographic", "ladi_classifier_priority",
    "clip_image_priority", "clip_image_priority_edf",
    "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn",
    "privileged_marginal_utility",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest_match(local_manifest: Path, downloaded_manifest: Path) -> None:
    columns = (
        "image_id", "source_sha256", "selection_stratum", "evidence_group_id",
        "split", "base_priority", "task_impact", "urgency", "evidence_sufficiency",
    )
    local = pd.read_csv(local_manifest).sort_values("image_id").reset_index(drop=True)
    remote = pd.read_csv(downloaded_manifest).sort_values("image_id").reset_index(drop=True)
    if not local[list(columns)].equals(remote[list(columns)]):
        raise ValueError("Local image manifest differs from downloaded training manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/scheduler-policy-local/image-disjoint-distinct")
    parser.add_argument("--policy-set", choices=("core", "all"), default="all")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    # Use the checked-out source, which is at the training commit. New review
    # scripts do not alter the policy or simulator modules.
    sys.path.insert(0, str(ROOT))
    from INFRA.experiments.semantic_scheduler_experiment import (
        TASK_ID, evaluate_policy_suite, load_catalog,
    )
    from INFRA.data.ladi_manifest import validate_ladi_manifest
    from INFRA.semantic_config import load_scheduler_config, load_ue_profiles
    from INFRA.scheduling.semantic_contracts import ResourceBudgetSemantic
    from INFRA.scheduling.streams import generate_episode_suite

    run_root = args.run_root.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    check_manifest_match(manifest_path, run_root / "dataset_manifest.csv")
    if run_manifest["git_commit"] != "d3338ef4faf553d83c2f0b868eb150aa9ab4a936":
        raise ValueError("This evaluation script expects the downloaded d3338ef run")
    for relative_path, expected in run_manifest["artifacts_sha256"].items():
        path = run_root / relative_path
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Missing or modified training artifact: {relative_path}")

    reviewed = validate_ladi_manifest(manifest_path, require_reviewed=True)
    config = load_scheduler_config(ROOT / "INFRA/config/semantic_scheduler_dynamic260.yaml")
    ue_profiles = load_ue_profiles(ROOT / "INFRA/config/ue_profiles.yaml")
    catalogue = load_catalog(manifest_path, run_root / "feature_cache.jsonl",
                             run_root / "payload_costs.csv")
    split_by_id = {row["image_id"]: row["split"] for row in reviewed}
    held_out = tuple(item for item in catalogue
                     if split_by_id[item.candidate.image_id] == "test")
    if len(held_out) != 52:
        raise ValueError(f"Expected 52 held-out images, got {len(held_out)}")
    episodes = generate_episode_suite(
        held_out, task_id=TASK_ID, prefix="image-disjoint-distinct",
        episode_count=args.episodes, jobs_per_episode=len(held_out), seed=4001,
        ue_profiles=ue_profiles, queue_capacity=config.queue_capacity,
        horizon_ms=config.horizon_ms, deadline_mode=config.deadline_mode)
    budget = ResourceBudgetSemantic(run_manifest["service_rate_units_per_ms"],
                                    resource_unit=run_manifest["resource_unit"])
    best_seed = max(run_manifest["dqn_best_validation"],
                    key=run_manifest["dqn_best_validation"].get)
    models = run_root / "models"
    artifacts = {
        "feature_ordinal_classifier": models / "ordinal_classifier.npz",
        "pairwise_linear_ranker": models / "pairwise_ranker.npz",
        "dqn": models / f"dqn_seed{best_seed}_best.npz",
    }
    ladi_predictions = json.loads((run_root / "ladi_predictions.json").read_text(encoding="utf-8"))
    clip_predictions = json.loads((models / "clip_priority_predictions.json").read_text(encoding="utf-8"))
    priority_by_image = {item.candidate.image_id: item.base_priority for item in catalogue}
    policy_names = (POLICIES if args.policy_set == "all" else (
        "fifo", "edf", "semantic_greedy", "ladi_classifier_priority",
        "clip_image_priority", "feature_ordinal_classifier",
        "pairwise_linear_ranker", "dqn", "privileged_marginal_utility"))

    provenance = {
        "training_run_id": run_manifest["run_id"],
        "training_commit": run_manifest["git_commit"],
        "training_manifest_sha256": run_manifest["manifest_sha256"],
        "local_manifest_sha256": file_sha256(manifest_path),
        "feature_cache_sha256": file_sha256(run_root / "feature_cache.jsonl"),
        "dqn_seed_selected_by_validation": int(best_seed),
        "episode_seed": 4001,
        "episode_count": len(episodes),
        "jobs_per_episode": len(held_out),
        "policy_names": policy_names,
        "payload_mode": run_manifest["payload_mode"],
        "resource_unit": run_manifest["resource_unit"],
        "scheduler_config": asdict(config),
        "notes": "Image-disjoint test; each episode uses the same 52 distinct test images in a seeded UE stream. Simulator cost is fixed, not measured SwinJSCC channel use.",
    }
    (output / "evaluation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    metrics_path = output / "all_policy_metrics.csv"
    completed = set()
    if metrics_path.is_file():
        existing = pd.read_csv(metrics_path)
        completed = set(zip(existing.episode_id, existing.policy))
    started = monotonic()
    for episode_number, episode in enumerate(episodes, start=1):
        for policy_name in policy_names:
            if (episode.episode_id, policy_name) in completed:
                continue
            retain = episode_number == 1 and policy_name in {
                "fifo", "edf", "semantic_greedy", "clip_image_priority",
                "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn"}
            results, rows = evaluate_policy_suite(
                (episode,), budget, priority_by_image=priority_by_image,
                predictions_by_image=ladi_predictions,
                image_priority_predictions_by_image=clip_predictions,
                model_artifacts=artifacts, policy_names=(policy_name,),
                retain_results=retain)
            row = {"track": "image_disjoint_distinct", "transport_p": 1.0,
                   "dqn_seed": int(best_seed), **rows[0]}
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            if retain:
                result = results[0]
                trace = [{"time_ms": item["time_ms"], "job_id": item["job_id"],
                          "score": item.get("score"), "reward": item.get("reward"),
                          "history_version": item.get("history_version"),
                          "candidate_snapshot": item.get("candidate_snapshot")}
                         for item in result["decision_trace"][:8]]
                (output / f"first_decisions_{policy_name}.json").write_text(
                    json.dumps(trace, indent=2, default=str), encoding="utf-8")
        print(f"episode {episode_number}/{len(episodes)} complete; elapsed {monotonic() - started:.1f}s", flush=True)

    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics.episode_id.isin({item.episode_id for item in episodes})
                      & metrics.policy.isin(policy_names)].copy()
    per_policy = metrics.groupby("policy", as_index=False).agg(
        semantic_utility_auc=("normalized_utility_auc", "mean"),
        priority_delivery_auc=("priority_delivery_auc", "mean"),
        critical_on_time_rate=("critical_on_time_rate", "mean"),
        weighted_inversions=("weighted_priority_inversion", "mean"),
        mean_inference_ms=("mean_policy_inference_ms", "mean"),
        episodes=("episode_id", "nunique"),
    ).sort_values("semantic_utility_auc", ascending=False)
    per_policy.to_csv(output / "policy_summary.csv", index=False)
    paired = metrics[["episode_id", "policy", "normalized_utility_auc"]].copy()
    for baseline in ("fifo", "edf"):
        reference = paired[paired.policy == baseline][["episode_id", "normalized_utility_auc"]]
        reference = reference.rename(columns={"normalized_utility_auc": f"{baseline}_auc"})
        paired = paired.merge(reference, on="episode_id", validate="many_to_one")
        paired[f"gain_vs_{baseline}"] = paired.normalized_utility_auc - paired[f"{baseline}_auc"]
    paired.to_csv(output / "dynamic_paired_priority_gains.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, max(5, .33 * len(per_policy))))
    shown = per_policy.sort_values("semantic_utility_auc")
    ax.barh(shown.policy, shown.semantic_utility_auc, color="#176b87")
    ax.set(xlabel="mean normalized semantic utility AUC (higher is better)",
           title=f"Same {len(held_out)} held-out images; {len(episodes)} frozen streams")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "policy_comparison.png", dpi=180)
    plt.close(fig)
    print(per_policy.to_string(index=False), flush=True)
    print("results:", output, flush=True)


if __name__ == "__main__":
    main()
