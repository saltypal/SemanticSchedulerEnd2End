"""Compare actual scheduling choices against prescribed class priority.

Each selected job is compared with every simultaneously eligible job of a
different priority class. Equal-priority pairs and future arrivals are excluded.
The resulting counts measure scheduling decisions, not classifier accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from INFRA.scheduling.event_simulation import run_episode
from INFRA.experiments.semantic_scheduler_experiment import TASK_ID, load_catalog
from INFRA.data.ladi_manifest import validate_ladi_manifest
from INFRA.semantic_config import load_scheduler_config, load_ue_profiles
from INFRA.scheduling.semantic_contracts import PolicyDecision, ResourceBudgetSemantic
from INFRA.scheduling.streams import generate_episode_suite
from Scheduler.semantic.registry import create_semantic_policy

from evaluate_downloaded_scheduler import DEFAULT_MANIFEST, DEFAULT_RUN, check_manifest_match, file_sha256


POLICIES = (
    "fifo", "ue_round_robin", "edf", "sjf", "fixed_priority", "priority_edf",
    "qos_class_urgency", "semantic_greedy", "semantic_weighted",
    "semantic_lexicographic", "ladi_classifier_priority",
    "clip_image_priority", "clip_image_priority_edf",
    "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn",
)

LABELS = {
    "fifo": "FIFO", "ue_round_robin": "Round robin", "edf": "EDF",
    "sjf": "Shortest job", "fixed_priority": "Class priority (answer key)",
    "priority_edf": "Class priority + EDF (answer key)",
    "qos_class_urgency": "QoS: class + urgency (answer key)",
    "semantic_greedy": "Semantic greedy", "semantic_weighted": "Semantic weighted",
    "semantic_lexicographic": "Semantic lexicographic",
    "ladi_classifier_priority": "LADI image classifier",
    "clip_image_priority": "CLIP image classifier",
    "clip_image_priority_edf": "CLIP image classifier + EDF",
    "feature_ordinal_classifier": "Feature classifier",
    "pairwise_linear_ranker": "Pairwise ranker", "dqn": "DQN",
}


class QoSClassUrgencyPolicy:
    """Privileged QoS reference: prescribed class, then annotated urgency."""

    name = "qos_class_urgency"

    def __init__(self, class_by_image: dict[str, int], urgency_by_image: dict[str, int]):
        self.class_by_image = class_by_image
        self.urgency_by_image = urgency_by_image

    def reset(self, episode_metadata=None) -> None:
        pass

    def observe(self, outcome) -> None:
        pass

    def decide(self, snapshot, budget) -> PolicyDecision:
        def key(index):
            job = snapshot.jobs[index]
            return (self.class_by_image[job.image_id],
                    -self.urgency_by_image[job.image_id],
                    job.deadline_ms if job.deadline_ms is not None else float("inf"),
                    job.arrival_ms, job.job_id)

        chosen = min(snapshot.eligible_indices, key=key)
        return PolicyDecision(self.name, snapshot.now_ms, chosen, None,
                              "prescribed class, then annotated urgency and deadline")


def count_priority_pairs(result: dict) -> tuple[int, int, int]:
    """Return matches, mismatches, and decisions with a comparable pair."""
    priority = {str(row["job_id"]): int(row["base_priority"])
                for row in result["job_facts"]}
    matches = mismatches = comparable_decisions = 0
    for decision in result["decision_trace"]:
        selected_id = str(decision["job_id"])
        selected_priority = priority[selected_id]
        considered = False
        for candidate in decision["candidate_snapshot"]:
            candidate_id = str(candidate["job_id"])
            if not candidate["eligible"] or candidate_id == selected_id:
                continue
            other_priority = priority[candidate_id]
            if other_priority == selected_priority:
                continue
            considered = True
            if selected_priority < other_priority:
                matches += 1
            else:
                mismatches += 1
        comparable_decisions += int(considered)
    return matches, mismatches, comparable_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/scheduler-policy-local/test30_core")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    run_root = args.run_root.resolve()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    check_manifest_match(manifest, run_root / "dataset_manifest.csv")
    for relative, expected in run_manifest["artifacts_sha256"].items():
        artifact = run_root / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise ValueError(f"Training artifact missing or changed: {relative}")

    reviewed = validate_ladi_manifest(manifest, require_reviewed=True)
    catalogue = load_catalog(manifest, run_root / "feature_cache.jsonl",
                             run_root / "payload_costs.csv")
    split = {row["image_id"]: row["split"] for row in reviewed}
    held_out = tuple(item for item in catalogue if split[item.candidate.image_id] == "test")
    if len(held_out) != 52:
        raise ValueError(f"Expected 52 distinct held-out images, found {len(held_out)}")
    config = load_scheduler_config(ROOT / "INFRA/config/semantic_scheduler_dynamic260.yaml")
    profiles = load_ue_profiles(ROOT / "INFRA/config/ue_profiles.yaml")
    episodes = generate_episode_suite(
        held_out, task_id=TASK_ID, prefix="image-disjoint-distinct",
        episode_count=args.episodes, jobs_per_episode=len(held_out), seed=4001,
        ue_profiles=profiles, queue_capacity=config.queue_capacity,
        horizon_ms=config.horizon_ms, deadline_mode=config.deadline_mode)
    budget = ResourceBudgetSemantic(run_manifest["service_rate_units_per_ms"],
                                    resource_unit=run_manifest["resource_unit"])
    best_seed = max(run_manifest["dqn_best_validation"],
                    key=run_manifest["dqn_best_validation"].get)
    models = run_root / "models"
    model_artifacts = {
        "feature_ordinal_classifier": models / "ordinal_classifier.npz",
        "pairwise_linear_ranker": models / "pairwise_ranker.npz",
        "dqn": models / f"dqn_seed{best_seed}_best.npz",
    }
    ladi_predictions = json.loads((run_root / "ladi_predictions.json").read_text(encoding="utf-8"))
    clip_predictions = json.loads((models / "clip_priority_predictions.json").read_text(encoding="utf-8"))
    class_by_image = {item.candidate.image_id: item.base_priority for item in catalogue}
    urgency_by_image = {row["image_id"]: int(row["urgency"]) for row in reviewed}

    rows = []
    for episode_number, episode in enumerate(episodes, start=1):
        for name in POLICIES:
            if name == "qos_class_urgency":
                policy = QoSClassUrgencyPolicy(class_by_image, urgency_by_image)
            else:
                policy = create_semantic_policy(
                    name, priority_by_image=class_by_image,
                    predictions_by_image=ladi_predictions,
                    image_priority_predictions_by_image=clip_predictions,
                    model_artifact=model_artifacts.get(name))
            result = run_episode(episode, policy, budget)
            matches, mismatches, comparable_decisions = count_priority_pairs(result)
            rows.append({"episode_id": episode.episode_id, "policy": name,
                         "matches": matches, "mismatches": mismatches,
                         "comparable_decisions": comparable_decisions})
        print(f"priority match evaluation {episode_number}/{len(episodes)}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "priority_pair_matches_by_episode.csv", index=False)
    summary = raw.groupby("policy", as_index=False)[["matches", "mismatches",
                                                      "comparable_decisions"]].sum()
    summary["match_rate"] = summary.matches / (summary.matches + summary.mismatches)
    summary = summary.sort_values("match_rate", ascending=True)
    summary.to_csv(output / "priority_pair_match_summary.csv", index=False)
    colors = ["#d99a31" if "answer key" in LABELS[name] else
              "#137e88" if name.startswith("semantic_") else
              "#5858b7" if name in {"dqn", "feature_ordinal_classifier",
                                   "pairwise_linear_ranker", "clip_image_priority",
                                   "clip_image_priority_edf", "ladi_classifier_priority"} else
              "#75869a" for name in summary.policy]
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f9fbfd")
    positions = list(range(len(summary)))
    ax.barh(positions, summary.match_rate, color=colors, height=.72)
    ax.barh(positions, 1 - summary.match_rate, left=summary.match_rate,
            color="#e8bec1", height=.72)
    ax.set_yticks(positions, [LABELS[name] for name in summary.policy])
    ax.set_xlim(0, 1.17)
    ax.set_xlabel("Matched priority pairs / all comparable eligible pairs")
    ax.set_title("When two priority classes are waiting, which scheduler picks the more important job?",
                 fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks([i / 10 for i in range(11)], [f"{10*i}%" for i in range(11)])
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    for y, row in enumerate(summary.itertuples()):
        ax.text(1.015, y, f"{row.match_rate:.1%}  ({row.matches:,} / {row.matches+row.mismatches:,})",
                va="center", fontsize=9)
    fig.text(.02, .035,
             "30 identical seeded streams × 52 distinct held-out images; five simulated UEs. "
             "Each chosen job is compared with every other eligible job of a different prescribed class.\n"
             "Green/blue/gray/gold = matches; pale red = mismatches. "
             "Gold policies see prescribed class (answer key); learned policies do not. "
             "Feature classifier receives rubric-assisted impact/urgency.\n"
             "QoS here uses prescribed class, then annotated urgency and deadline. "
             "Ties and future arrivals are excluded; pair totals vary with queue states. "
             "Payload cost is simulated fixed.",
             fontsize=8.5, color="#394652", va="bottom")
    fig.subplots_adjust(left=.29, right=.83, top=.89, bottom=.17)
    for extension in ("png", "pdf"):
        target = output / f"dynamic_priority_matches.{extension}"
        fig.savefig(target, dpi=220, bbox_inches="tight")
        print(target)
    plt.close(fig)
    print(summary[["policy", "matches", "mismatches", "match_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
