"""Evaluate schedulers with an equal decision denominator.

Each episode contains exactly 25 jobs, has generous deadlines, and uses a
fixed class-plus-urgency reference. Every policy therefore makes 25 decisions
and receives the same number of match opportunities. This script evaluates the
current DQN; it does not retrain or tune it to the test stream.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    "semantic_greedy", "semantic_weighted", "semantic_lexicographic",
    "ladi_classifier_priority", "clip_image_priority", "clip_image_priority_edf",
    "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn",
)

LABELS = {
    "fifo": "FIFO", "ue_round_robin": "Round robin", "edf": "EDF",
    "sjf": "Shortest job", "fixed_priority": "Class priority",
    "priority_edf": "Class priority + EDF", "semantic_greedy": "Semantic greedy",
    "semantic_weighted": "Semantic weighted", "semantic_lexicographic": "Semantic lexicographic",
    "ladi_classifier_priority": "LADI image classifier", "clip_image_priority": "CLIP image classifier",
    "clip_image_priority_edf": "CLIP image classifier + EDF", "feature_ordinal_classifier": "Feature classifier",
    "pairwise_linear_ranker": "Pairwise ranker", "dqn": "DQN",
}


class QoSPolicy:
    """Evaluator-only class+urgency reference, included for presentation."""

    name = "qos_class_urgency"

    def __init__(self, priority_by_image, urgency_by_image):
        self.priority_by_image = priority_by_image
        self.urgency_by_image = urgency_by_image

    def reset(self, episode_metadata=None):
        pass

    def observe(self, outcome):
        pass

    def decide(self, snapshot, budget):
        def key(index):
            job = snapshot.jobs[index]
            deadline = float("inf") if job.deadline_ms is None else job.deadline_ms
            return (self.priority_by_image[job.image_id],
                    -self.urgency_by_image[job.image_id], deadline,
                    job.arrival_ms, job.job_id)

        chosen = min(snapshot.eligible_indices, key=key)
        return PolicyDecision(self.name, snapshot.now_ms, chosen, None,
                              "class priority, urgency, deadline")


def expected_class_urgency_match(result: dict, priority_by_job: dict[str, int],
                                 urgency_by_job: dict[str, int]) -> tuple[int, int]:
    """Count one match/mismatch per decision, with ties accepted as matches."""
    matches = mismatches = 0
    for decision in result["decision_trace"]:
        eligible = [candidate for candidate in decision["candidate_snapshot"] if candidate["eligible"]]
        if not eligible:
            raise ValueError("Decision trace contains no eligible candidate")
        def key(candidate):
            job_id = str(candidate["job_id"])
            return (priority_by_job[job_id], -urgency_by_job[job_id],
                    float(candidate["deadline_ms"]), float(candidate["time_ms"])
                    if "time_ms" in candidate else 0.0, job_id)
        # Candidate rows do not carry time_ms; the final job ID tie-break is
        # deterministic and only affects equal class/urgency choices.
        best_priority = min((priority_by_job[str(item["job_id"])] for item in eligible))
        best_urgency = max(urgency_by_job[str(item["job_id"])] for item in eligible
                           if priority_by_job[str(item["job_id"])] == best_priority)
        selected = str(decision["job_id"])
        selected_key = (priority_by_job[selected], urgency_by_job[selected])
        if selected_key == (best_priority, best_urgency):
            matches += 1
        else:
            mismatches += 1
    return matches, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/scheduler-policy-local/fixed25_matches")
    args = parser.parse_args()
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
    catalogue = load_catalog(manifest, run_root / "feature_cache.jsonl", run_root / "payload_costs.csv")
    config = load_scheduler_config(ROOT / "INFRA/config/semantic_scheduler_dynamic260.yaml")
    profiles = load_ue_profiles(ROOT / "INFRA/config/ue_profiles.yaml")
    # Use the train catalogue with replacement so repeated arrivals can change
    # delivered-history novelty while test episodes remain deterministic.
    train = tuple(item for item in catalogue
                  if next(row["split"] for row in reviewed if row["image_id"] == item.candidate.image_id) == "train")
    episodes = generate_episode_suite(
        train, task_id=TASK_ID, prefix="fixed25", episode_count=args.episodes,
        jobs_per_episode=25, seed=7001, ue_profiles=profiles,
        queue_capacity=25, horizon_ms=16000.0, allow_reuse=True,
        deadline_mode="independent_network_sla")
    # Remove deadline/expiry as a confounder for this comparison. Each policy
    # must process exactly the same 25 jobs and make exactly 25 decisions.
    episodes = tuple(replace(episode, horizon_ms=16000.0, jobs=tuple(
        replace(item, candidate=replace(item.candidate,
                                         deadline_ms=item.candidate.arrival_ms + 10000.0))
        for item in episode.jobs)) for episode in episodes)
    budget = ResourceBudgetSemantic(run_manifest["service_rate_units_per_ms"],
                                    resource_unit=run_manifest["resource_unit"])
    best_seed = max(run_manifest["dqn_best_validation"], key=run_manifest["dqn_best_validation"].get)
    models = run_root / "models"
    artifacts = {
        "feature_ordinal_classifier": models / "ordinal_classifier.npz",
        "pairwise_linear_ranker": models / "pairwise_ranker.npz",
        "dqn": models / f"dqn_seed{best_seed}_best.npz",
    }
    ladi = json.loads((run_root / "ladi_predictions.json").read_text(encoding="utf-8"))
    clip = json.loads((models / "clip_priority_predictions.json").read_text(encoding="utf-8"))
    priority_by_image = {item.candidate.image_id: item.base_priority for item in catalogue}
    urgency_by_image = {row["image_id"]: int(row["urgency"]) for row in reviewed}
    rows = []
    names = POLICIES + ("qos_class_urgency",)
    for episode_index, episode in enumerate(episodes, start=1):
        priority_by_job = {item.candidate.job_id: item.base_priority for item in episode.jobs}
        urgency_by_job = {item.candidate.job_id: urgency_by_image[item.candidate.image_id]
                          for item in episode.jobs}
        for name in names:
            if name == "qos_class_urgency":
                policy = QoSPolicy(priority_by_image, urgency_by_image)
            else:
                policy = create_semantic_policy(
                    name, priority_by_image=priority_by_image,
                    predictions_by_image=ladi,
                    image_priority_predictions_by_image=clip,
                    model_artifact=artifacts.get(name))
            result = run_episode(episode, policy, budget)
            matches, mismatches = expected_class_urgency_match(result, priority_by_job, urgency_by_job)
            if len(result["decision_trace"]) != 25:
                raise ValueError(f"{name} made {len(result['decision_trace'])}, expected 25 decisions")
            rows.append({"episode_id": episode.episode_id, "policy": name,
                         "decisions": 25, "matches": matches, "mismatches": mismatches,
                         "match_rate": matches / 25.0})
        print(f"fixed-denominator mismatch evaluation {episode_index}/{len(episodes)}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "by_episode.csv", index=False)
    summary = raw.groupby("policy", as_index=False).agg(
        decisions=("decisions", "sum"), matches=("matches", "sum"),
        mismatches=("mismatches", "sum"), match_rate=("match_rate", "mean"),
    ).sort_values("match_rate")
    summary.to_csv(output / "summary.csv", index=False)
    colors = ["#d99a31" if name in {"fixed_priority", "priority_edf", "qos_class_urgency"}
              else "#5858b7" if name in {"dqn", "feature_ordinal_classifier",
                                          "pairwise_linear_ranker", "clip_image_priority",
                                          "clip_image_priority_edf", "ladi_classifier_priority"}
              else "#137e88" if name.startswith("semantic_") else "#75869a"
              for name in summary.policy]
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f9fbfd")
    y = list(range(len(summary)))
    ax.barh(y, summary.match_rate, color=colors, height=.72)
    ax.barh(y, 1 - summary.match_rate, left=summary.match_rate,
            color="#e8bec1", height=.72)
    ax.set_yticks(y, [LABELS.get(name, "QoS: class + urgency") for name in summary.policy])
    ax.set_xlim(0, 1.17)
    ax.set_xticks([i / 10 for i in range(11)], [f"{10*i}%" for i in range(11)])
    ax.set_xlabel("Correct next-job decisions / 25 fixed decisions per episode")
    ax.set_title("Equal-denominator dynamic priority mismatch comparison",
                 fontsize=15, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    for row_index, row in enumerate(summary.itertuples()):
        ax.text(1.015, row_index,
                f"{row.match_rate:.1%}  ({row.matches:,} / {row.decisions:,})",
                va="center", fontsize=9)
    fig.text(.02, .035,
             "30 repeated-catalogue streams × 25 jobs; five UE arrival profiles; no expiry or transport failure. "
             "A match means the selected job has the best prescribed class and urgency among currently eligible jobs.\n"
             "Class-priority and QoS rows receive the prescribed class and are answer-key references. "
             "The current DQN was trained for semantic delivery utility, not this mismatch objective.\n"
             "All policies process the same number of jobs and make the same number of decisions. Results are simulator results.",
             fontsize=8.5, color="#394652", va="bottom")
    fig.subplots_adjust(left=.29, right=.83, top=.89, bottom=.17)
    for extension in ("png", "pdf"):
        target = output / f"fixed25_priority_matches.{extension}"
        fig.savefig(target, dpi=220, bbox_inches="tight")
        print(target)
    plt.close(fig)
    print(summary[["policy", "matches", "mismatches", "match_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
