"""Compare the frozen DQN with schedulers on one combined dynamic scenario.

The 52 image-disjoint catalogue entries repeat within every episode.
Five UEs create staggered and burst arrivals; the 25-job queue can overflow.
Every policy receives the same episode. No model is trained or modified here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Evaluation.scheduling.event_simulation import run_episode
from Evaluation.experiments.semantic_scheduler_experiment import TASK_ID, load_catalog
from Evaluation.data.ladi_manifest import validate_ladi_manifest
from Evaluation.semantic_config import load_scheduler_config, load_ue_profiles
from Evaluation.scheduling.semantic_contracts import ResourceBudgetSemantic
from Evaluation.evaluation.semantic import aggregate_episode_metrics
from Evaluation.scheduling.streams import build_episode
from Scheduler.semantic.registry import create_semantic_policy

from evaluate_downloaded_scheduler import (
    DEFAULT_MANIFEST, DEFAULT_RUN, check_manifest_match, file_sha256,
)


POLICIES = (
    "fifo", "ue_round_robin", "edf", "sjf", "aging",
    "semantic_greedy", "semantic_weighted", "semantic_lexicographic",
    "ladi_classifier_priority", "clip_image_priority", "clip_image_priority_edf",
    "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn",
    "fixed_priority", "priority_edf", "privileged_marginal_utility",
)
PRIVILEGED = {"fixed_priority", "priority_edf", "privileged_marginal_utility"}


def priority_decision_matches(result: dict, priority_by_job: dict[str, int],
                              urgency_by_job: dict[str, int]) -> dict[str, int]:
    """Compare the action with the best *currently eligible* class/urgency.

    This is a rubric-order diagnostic, separate from the semantic-utility
    objective used to train the frozen checkpoint. Ties count as matches.
    """
    matches = contested_matches = contested = 0
    decisions = result["decision_trace"]
    for decision in decisions:
        eligible_ids = [str(item["job_id"]) for item in decision["candidate_snapshot"]
                        if item["eligible"]]
        selected_id = str(decision["job_id"])
        if selected_id not in eligible_ids:
            raise AssertionError("A policy selected an ineligible job")
        best_priority = min(priority_by_job[job_id] for job_id in eligible_ids)
        best_urgency = max(urgency_by_job[job_id] for job_id in eligible_ids
                           if priority_by_job[job_id] == best_priority)
        selected_matches = (priority_by_job[selected_id], urgency_by_job[selected_id]) == (
            best_priority, best_urgency)
        matches += int(selected_matches)
        choices_differ = len({(priority_by_job[job_id], urgency_by_job[job_id])
                              for job_id in eligible_ids}) > 1
        if choices_differ:
            contested += 1
            contested_matches += int(selected_matches)
    return {
        "decisions": len(decisions), "matches": matches,
        "mismatches": len(decisions) - matches,
        "contested_decisions": contested, "contested_matches": contested_matches,
        "contested_mismatches": contested - contested_matches,
    }


def observed_novelty_changes(result: dict) -> int:
    """Count queued images whose P3 changed across visible decision snapshots."""
    values_by_image: dict[str, set[float]] = {}
    for decision in result["decision_trace"]:
        for candidate in decision["candidate_snapshot"]:
            if candidate["availability"][2]:
                values_by_image.setdefault(str(candidate["image_id"]), set()).add(
                    round(float(candidate["raw_features"][2]), 6))
    return sum(len(values) > 1 for values in values_by_image.values())


def _episode(catalogue: tuple, *, ordinal: int, root_seed: int, repeats: int,
             ue_profiles: tuple, queue_capacity: int, horizon_ms: float):
    # Every image has exactly ``repeats`` arrivals. New seeds change order and UE
    # assignment, not the catalogue or class mix.
    templates = list(catalogue) * repeats
    episode_seed = random.Random(root_seed).randrange(2**31) if ordinal == 0 else (
        random.Random(f"{root_seed}:episode:{ordinal}").randrange(2**31))
    random.Random(episode_seed).shuffle(templates)
    return build_episode(
        templates, episode_id=f"heldout-overload-{ordinal:04d}",
        task_id=TASK_ID, seed=episode_seed, ue_profiles=ue_profiles,
        queue_capacity=queue_capacity, horizon_ms=horizon_ms,
        require_distinct_images=False, deadline_mode="independent_network_sla",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2,
                        help="Arrivals per held-out image; 2 approximates training multiplicity")
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "results/scheduler-policy-local/combined_dynamic_overload")
    args = parser.parse_args()
    if args.episodes <= 0 or args.repeats < 2:
        parser.error("--episodes must be positive and --repeats must be at least two")

    source = args.run_root.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    check_manifest_match(manifest_path, source / "dataset_manifest.csv")
    for relative_path, expected in run_manifest["artifacts_sha256"].items():
        path = source / relative_path
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Missing or changed source artifact: {relative_path}")
    rows = validate_ladi_manifest(manifest_path, require_reviewed=True)
    split_by_image = {row["image_id"]: row["split"] for row in rows}
    urgency_by_image = {row["image_id"]: int(row["urgency"]) for row in rows}
    full_catalogue = load_catalog(manifest_path, source / "feature_cache.jsonl",
                                  source / "payload_costs.csv")
    heldout = tuple(item for item in full_catalogue
                    if split_by_image[item.candidate.image_id] == "test")
    if len(heldout) != 52:
        raise ValueError(f"Expected 52 held-out images, found {len(heldout)}")
    settings = load_scheduler_config(ROOT / "Evaluation/config/semantic_scheduler_dynamic260.yaml")
    ue_profiles = load_ue_profiles(ROOT / "Evaluation/config/ue_profiles.yaml")
    if settings.queue_capacity != 25 or len(ue_profiles) != 5:
        raise ValueError("Frozen checkpoint comparison expects a 25-slot queue and five UEs")
    if run_manifest["resource_unit"] != "simulated_units":
        raise ValueError("This comparison was designed for the existing simulated-cost checkpoint")
    budget = ResourceBudgetSemantic(
        float(run_manifest["service_rate_units_per_ms"]),
        resource_unit=run_manifest["resource_unit"],
    )
    best_seed = max(run_manifest["dqn_best_validation"],
                    key=run_manifest["dqn_best_validation"].get)
    model_dir = source / "models"
    artifacts = {
        "feature_ordinal_classifier": model_dir / "ordinal_classifier.npz",
        "pairwise_linear_ranker": model_dir / "pairwise_ranker.npz",
        "dqn": model_dir / f"dqn_seed{best_seed}_best.npz",
    }
    predictions = json.loads((source / "ladi_predictions.json").read_text(encoding="utf-8"))
    image_predictions = json.loads((model_dir / "clip_priority_predictions.json").read_text(encoding="utf-8"))
    priority_by_image = {item.candidate.image_id: item.base_priority
                         for item in full_catalogue}
    oracle_facts = {item.candidate.image_id: {
        "base_priority": item.base_priority,
        "evidence_group_id": item.evidence_group_id,
        "evidence_sufficiency": item.candidate.features.evidence_sufficiency.value,
        "urgency": item.candidate.features.urgency.value,
    } for item in heldout}

    specification = {
        "scenario": "image-disjoint repeated-catalogue five-UE queue overload",
        "selection_rule": f"52 test images, exactly {args.repeats} arrivals per image; no cherry-picked images",
        "episode_count": args.episodes, "root_seed": args.seed,
        "jobs_per_episode": 52 * args.repeats, "catalogue_images": 52,
        "repeats_per_image": args.repeats,
        "queue_capacity": settings.queue_capacity,
        "horizon_ms": settings.horizon_ms,
        "deadline_mode": "independent_network_sla",
        "transport_success_probability": 1.0,
        "ue_profiles": [asdict(profile) for profile in ue_profiles],
        "policies": list(POLICIES), "privileged_policies": sorted(PRIVILEGED),
        "dqn_training_objective": "semantic_utility",
        "dqn_checkpoint": str(artifacts["dqn"]),
        "dqn_checkpoint_sha256": file_sha256(artifacts["dqn"]),
        "source_run_id": run_manifest["run_id"],
        "source_commit": run_manifest["git_commit"],
        "manifest_sha256": file_sha256(manifest_path),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource_unit": budget.resource_unit,
        "service_rate_units_per_ms": budget.service_rate_units_per_ms,
        "metric_note": "Priority match compares class and urgency among eligible jobs. "
                       "Semantic utility is the DQN training objective. "
                       "Policy-dependent drops make decision counts differ.",
    }
    (output / "scenario.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")

    metrics_rows = []
    for ordinal in range(args.episodes):
        episode = _episode(heldout, ordinal=ordinal, root_seed=args.seed,
                           repeats=args.repeats,
                           ue_profiles=ue_profiles,
                           queue_capacity=settings.queue_capacity,
                           horizon_ms=settings.horizon_ms)
        if (len(episode.jobs) != 52 * args.repeats
                or len({item.candidate.image_id for item in episode.jobs}) != 52):
            raise AssertionError("Repeated-catalogue episode construction changed")
        priority_by_job = {item.candidate.job_id: item.base_priority for item in episode.jobs}
        urgency_by_job = {item.candidate.job_id: urgency_by_image[item.candidate.image_id]
                          for item in episode.jobs}
        for name in POLICIES:
            policy = create_semantic_policy(
                name, priority_by_image=priority_by_image,
                predictions_by_image=predictions,
                image_priority_predictions_by_image=image_predictions,
                oracle_facts_by_image=oracle_facts,
                model_artifact=artifacts.get(name),
            )
            result = run_episode(episode, policy, budget,
                                 enable_delivered_history=True)
            counts = priority_decision_matches(result, priority_by_job, urgency_by_job)
            metrics = aggregate_episode_metrics(result)
            started_count = sum(row["started_ms"] is not None for row in result["outcomes"])
            if metrics["jobs"] != 52 * args.repeats or counts["decisions"] != started_count:
                raise AssertionError("Episode accounting or payload-cost assumption changed")
            metrics_rows.append({
                "episode_id": episode.episode_id, "policy": name,
                "privileged": name in PRIVILEGED,
                **counts,
                "semantic_utility_auc": metrics["normalized_utility_auc"],
                "priority_delivery_auc": metrics["priority_delivery_auc"],
                "final_utility": metrics["final_utility"],
                "critical_on_time_rate": metrics["critical_on_time_rate"],
                "deadline_miss_rate": metrics["deadline_miss_rate"],
                "completed_on_time": metrics["completed_on_time"],
                "completed_late": metrics["completed_late"],
                "overflow_drop": metrics["overflow_drop"],
                "expired_waiting": metrics["expired_waiting"],
                "novelty_changes": observed_novelty_changes(result),
            })
        print(f"combined dynamic episode {ordinal + 1}/{args.episodes} complete", flush=True)

    _write_csv(output / "by_episode.csv", metrics_rows)
    summary_rows = []
    for name in POLICIES:
        policy_rows = [row for row in metrics_rows if row["policy"] == name]
        decisions = sum(row["decisions"] for row in policy_rows)
        matches = sum(row["matches"] for row in policy_rows)
        contested = sum(row["contested_decisions"] for row in policy_rows)
        contested_matches = sum(row["contested_matches"] for row in policy_rows)
        summary_rows.append({
            "policy": name, "privileged": name in PRIVILEGED,
            "episodes": len(policy_rows), "offered_jobs": 52 * args.repeats * len(policy_rows),
            "decisions": decisions, "matches": matches,
            "mismatches": decisions - matches,
            "match_rate": matches / decisions if decisions else 0.0,
            "contested_decisions": contested,
            "contested_match_rate": contested_matches / contested if contested else 0.0,
            "semantic_utility_auc": sum(row["semantic_utility_auc"] for row in policy_rows) / len(policy_rows),
            "priority_delivery_auc": sum(row["priority_delivery_auc"] for row in policy_rows) / len(policy_rows),
            "critical_on_time_rate": sum(row["critical_on_time_rate"] for row in policy_rows) / len(policy_rows),
            "overflow_drops": sum(row["overflow_drop"] for row in policy_rows),
            "expired_waiting": sum(row["expired_waiting"] for row in policy_rows),
            "episodes_with_p3_change": sum(row["novelty_changes"] > 0 for row in policy_rows),
        })
    summary_rows.sort(key=lambda row: row["semantic_utility_auc"], reverse=True)
    _write_csv(output / "summary.csv", summary_rows)

    public = [row for row in summary_rows if not row["privileged"]]
    chart_rows = sorted(public, key=lambda row: row["semantic_utility_auc"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#f8fafc")
    names = [row["policy"].replace("_", " ") for row in chart_rows]
    colors = ["#df8a27" if row["policy"] == "dqn" else "#227e89" for row in chart_rows]
    axes[0].barh(names, [row["semantic_utility_auc"] for row in chart_rows], color=colors)
    axes[0].set(xlabel="Normalized semantic utility AUC", title="Delivered semantic utility over time")
    axes[1].barh(names, [100 * row["contested_match_rate"] for row in chart_rows], color=colors)
    axes[1].set(xlabel="Correct contested decisions (%)", title="Prescribed class/urgency alignment")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Frozen DQN on held-out repeated images, bursty five-UE overload", fontsize=15)
    fig.text(0.02, 0.015,
             f"Same {52 * args.repeats} offered jobs per episode and 25-slot queue. Drops/expiries make decision counts policy-dependent; "
             "see summary.csv. Privileged label-aware references excluded from chart. Simulator results.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(output / "combined_dynamic_comparison.png", dpi=180)
    plt.close(fig)
    print("\n" + "\n".join(
        f"{row['policy']:30s} utility={row['semantic_utility_auc']:.3f} "
        f"contested_match={row['contested_match_rate']:.1%} "
        f"decisions={row['decisions']} drops={row['overflow_drops']}"
        for row in summary_rows), flush=True)
    print(f"results: {output}", flush=True)


if __name__ == "__main__":
    main()
