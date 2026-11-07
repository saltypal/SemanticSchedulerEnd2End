"""Build the standalone, read-only review notebook for a downloaded Kaggle run."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Scheduler" / "notebooks" / "03_review_downloaded_run.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown("""
# Semantic scheduler: downloaded run review

This notebook reads one completed Kaggle output directory. It does **not** retrain or require images, a GPU, or repository imports. Run it top to bottom after downloading the kernel outputs. Tables and charts are computed from the files found in the selected run.

**Reading rule:** DQN validation scores are for checkpoint selection; classifier accuracy is for image priority prediction; scheduler utility requires `evaluation/all_policy_metrics.csv`. These are different quantities.
"""),
    markdown("""
## 1. Locate the run and check what was downloaded

Set `RUN_ROOT_OVERRIDE` if your files are outside the repository. The default selects the most complete matching run under `downloaded/`, preferring `train-final-cli`. A partial Kaggle download is common after cancelling a version, so the inventory is a required first step.
"""),
    code(r'''
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

RUN_ROOT_OVERRIDE = None  # Example: Path(r"D:\downloads\outputs\20260929T021341Z-d3338ef4")

def find_project_root(start):
    for parent in (start, *start.parents):
        if (parent / "PROJECT_MAP.md").is_file():
            return parent
    raise FileNotFoundError("Open this notebook from the repository, or set RUN_ROOT_OVERRIDE")

project_root = find_project_root(Path.cwd().resolve())
if RUN_ROOT_OVERRIDE is not None:
    run_root = Path(RUN_ROOT_OVERRIDE).expanduser().resolve()
else:
    candidates = list((project_root / "downloaded").glob("*/outputs/*/run_manifest.json"))
    if not candidates:
        raise FileNotFoundError("No downloaded run_manifest.json found; set RUN_ROOT_OVERRIDE")
    candidates.sort(key=lambda p: ("train-final-cli" in p.parts, len(list(p.parent.rglob("*")))), reverse=True)
    run_root = candidates[0].parent

if not (run_root / "run_manifest.json").is_file():
    raise FileNotFoundError(f"Run manifest missing under {run_root}")
run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
local_evaluation_root = project_root / "results/scheduler-policy-local/test30_core"
local_provenance_path = local_evaluation_root / "evaluation_provenance.json"
if local_provenance_path.is_file():
    local_provenance = json.loads(local_provenance_path.read_text(encoding="utf-8"))
    if local_provenance.get("training_run_id") != run_manifest.get("run_id"):
        raise ValueError("Local evaluation was created from a different training run")
downloaded_policy_path = run_root / "evaluation/all_policy_metrics.csv"
local_policy_path = local_evaluation_root / "all_policy_metrics.csv"
all_policy_path = downloaded_policy_path if downloaded_policy_path.is_file() else local_policy_path
all_files = sorted(p for p in run_root.rglob("*") if p.is_file())
inventory = pd.DataFrame([{"artifact": p.relative_to(run_root).as_posix(),
                           "bytes": p.stat().st_size} for p in all_files])
print("Run:", run_root)
print("Files:", len(inventory), "| Total MiB:", round(inventory.bytes.sum() / 2**20, 2))
display(inventory.assign(group=inventory.artifact.str.split("/").str[0])
        .groupby("group").agg(files=("artifact", "size"), bytes=("bytes", "sum")))
display(inventory[inventory.artifact.str.startswith("evaluation/")])
'''),
    code(r'''
required = {
    "training_metrics.csv": "DQN training curves",
    "models/dqn_seed42_manifest.json": "DQN seed metadata",
    "evaluation/priority_classifier_metrics.csv": "priority classification",
    "evaluation/all_policy_metrics.csv": "all-policy scheduler comparison",
    "evaluation/dynamic_paired_priority_gains.csv": "paired gains versus FIFO/EDF",
    "evaluation/evaluation_manifest.json": "finished evaluation provenance",
}
coverage = pd.DataFrame([{"artifact": name, "meaning": meaning,
                          "in_kaggle_download": (run_root / name).is_file(),
                          "in_local_evaluation": (local_evaluation_root / Path(name).name).is_file()}
                         for name, meaning in required.items()])
display(coverage)
expected_hashes = run_manifest.get("artifacts_sha256", {})
hash_rows = []
for relative_name, expected in expected_hashes.items():
    path = run_root / relative_name
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    hash_rows.append({"artifact": relative_name, "present": path.is_file(), "sha256_matches": actual == expected})
hash_check = pd.DataFrame(hash_rows)
display(hash_check)
if not hash_check.sha256_matches.all():
    display(Markdown("**Integrity warning:** one or more training artifacts are missing or differ from the run manifest."))
if not downloaded_policy_path.is_file():
    if all_policy_path.is_file():
        display(Markdown("**Local evaluation available:** the Kaggle download stopped before all-policy testing, so the policy comparison below comes from a separate local replay of the saved models and cached features."))
    else:
        display(Markdown("**Evaluation incomplete:** no all-policy scheduler results are available yet."))
'''),
    markdown("""
## 2. What experiment was run?

The fixed rubric supplies P5/P6/P7 and priority classes; it is a prototype task assumption. P1/P4 use a LADI classifier, P2 uses CLIP task similarity, and P3 is calculated from successfully delivered history during simulation. The download uses **simulated fixed payload units**, so it cannot support claims about actual SwinJSCC channel use or reconstruction quality.
"""),
    code(r'''
manifest = pd.read_csv(run_root / "dataset_manifest.csv")
payload = pd.read_csv(run_root / "payload_costs.csv")
assert manifest.image_id.is_unique, "The manifest contains duplicate image IDs"
assert set(manifest.image_id) == set(payload.image_id), "Payload rows do not match images"
summary = {
    "run_id": run_manifest.get("run_id"),
    "git_commit": run_manifest.get("git_commit"),
    "images": len(manifest),
    "selection_strata": manifest.selection_stratum.nunique(),
    "evidence_groups": manifest.evidence_group_id.nunique(),
    "payload_mode": run_manifest.get("payload_mode"),
    "resource_unit": run_manifest.get("resource_unit"),
    "dataset_annotation_source": ", ".join(sorted(manifest.annotation_source.dropna().unique())),
}
display(pd.DataFrame(summary.items(), columns=["item", "value"]))
display(pd.crosstab(manifest.selection_stratum, manifest.split))
display(pd.crosstab(manifest.base_priority, manifest.split).rename_axis("priority (0 = highest)"))
group_split_counts = manifest.groupby("evidence_group_id").split.nunique()
print("Evidence groups crossing splits:", int((group_split_counts > 1).sum()))
print("Simulated payload units per image:", sorted(payload.resource_units.unique().tolist()))
'''),
    markdown("""
## 3. Extracted semantic parameters and timing

This chart summarizes static per-image features. **P3 is expected to be unavailable in this cache**: it changes at each scheduling decision as the receiver history changes. P5/P6/P7 come from the fixed rubric, so their apparent predictive strength should not be interpreted as a fully automatic image understanding result.
"""),
    code(r'''
feature_rows = [json.loads(line) for line in (run_root / "feature_cache.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
parameter_names = ["scene_importance", "context_relevance", "delivered_history_novelty",
                   "interpretation_confidence", "task_impact", "urgency", "evidence_sufficiency"]
flat_rows = []
for row in feature_rows:
    output = {"image_id": row["image_id"], "extraction_latency_ms": row.get("extraction_latency_ms"),
              "process_cpu_ms": row.get("process_cpu_ms"), "gpu_peak_allocated_bytes": row.get("gpu_peak_allocated_bytes")}
    for name in parameter_names:
        item = row["features"].get(name, {})
        output[name] = item.get("value") if item.get("available") else np.nan
        output[name + "_available"] = bool(item.get("available", False))
    flat_rows.append(output)
features = pd.DataFrame(flat_rows).merge(manifest[["image_id", "split", "selection_stratum", "base_priority"]], on="image_id", validate="one_to_one")
assert len(features) == len(manifest)
availability = pd.DataFrame({"parameter": parameter_names,
                             "available_images": [int(features[name + "_available"].sum()) for name in parameter_names]})
display(availability)
display(features[["extraction_latency_ms", "process_cpu_ms", "gpu_peak_allocated_bytes"]].describe(percentiles=[.5, .95]).round(2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
latency_seconds = features.extraction_latency_ms / 1000
latency_seconds.plot.hist(bins=25, ax=axes[0], color="#19647e")
axes[0].set_xscale("symlog", linthresh=.1)
axes[0].set(title="Per-image extraction time (log-like scale)", xlabel="seconds", ylabel="images")
for name, label in [("scene_importance", "P1 scene"), ("context_relevance", "P2 relevance"),
                    ("interpretation_confidence", "P4 confidence")]:
    features[name].plot.hist(bins=25, ax=axes[1], histtype="step", linewidth=2, label=label)
axes[1].set(title="Automatically derived static feature values", xlabel="feature value", ylabel="images")
axes[1].legend(); fig.tight_layout(); plt.show()
'''),
    markdown("""
## 4. Priority classifier quality

Three classifiers predict a rubric-defined priority level (0 is highest). `macro_f1` gives each class equal weight; `balanced_accuracy` averages recall across classes. The feature ordinal model observes rubric-assisted impact and urgency, so compare the two image-only models separately when discussing automatic image interpretation.
"""),
    code(r'''
classifier_metrics = pd.read_csv(run_root / "evaluation/priority_classifier_metrics.csv")
predictions = pd.read_csv(run_root / "evaluation/priority_classifier_predictions.csv")
display(classifier_metrics.sort_values(["split", "macro_f1"], ascending=[True, False]).round(3))
test_scores = classifier_metrics[classifier_metrics.split == "test"].sort_values("macro_f1")
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(test_scores.model, test_scores.macro_f1, color="#19647e")
ax.set(xlim=(0, 1), xlabel="Macro F1 on 52 held-out images", title="Priority classification (not scheduler utility)")
for y, value in enumerate(test_scores.macro_f1):
    ax.text(value + .015, y, f"{value:.3f}", va="center")
fig.tight_layout(); plt.show()
for model in predictions.model.unique():
    subset = predictions[(predictions.split == "test") & (predictions.model == model)]
    table = pd.crosstab(subset.true_priority, subset.predicted_priority,
                        rownames=["true priority"], colnames=["predicted priority"])
    display(Markdown(f"**{model}: test confusion table**"))
    display(table)
'''),
    markdown("""
## 5. DQN training across three seeds

Each seed starts from a different random state. The best checkpoint is selected by validation semantic-utility AUC. A validation score is **not** an all-policy test comparison. The run used a wall-time cap: each seed can stop before all 300 configured episodes.
"""),
    code(r'''
training = pd.read_csv(run_root / "training_metrics.csv")
seed_rows = []
for path in sorted((run_root / "models").glob("dqn_seed*_manifest.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    seed_rows.append({"seed": record["seed"], "completed_episodes": record["completed_episodes"],
                      "configured_episodes": record["configured_episodes"],
                      "minutes": record["elapsed_seconds"] / 60,
                      "best_validation_utility_auc": record["best_validation_semantic_utility_auc"],
                      "stopped_by_time_budget": record["stopped_by_time_budget"],
                      "data_parallel": record["provenance"].get("dqn_data_parallel")})
seeds = pd.DataFrame(seed_rows).sort_values("seed")
display(seeds.round(3))
best_seed = int(seeds.loc[seeds.best_validation_utility_auc.idxmax(), "seed"])
print("Selected by validation:", best_seed)
print("Invalid actions during training:", int(training.invalid_action_count.sum()))
print("Loss trend is a diagnostic, not proof of convergence; use held-out policy evaluation for the scheduling claim.")
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for seed, group in training.groupby("seed"):
    group = group.sort_values("episode")
    label = f"seed {seed}"
    axes[0, 0].plot(group.episode, group.reward.rolling(10, min_periods=1).mean(), label=label)
    axes[0, 1].plot(group.episode, group.training_priority_delivery_auc.rolling(10, min_periods=1).mean(), label=label)
    axes[1, 0].plot(group.episode, group.loss.rolling(10, min_periods=1).mean(), label=label)
    axes[1, 1].plot(group.episode, group.validation_semantic_utility_auc, marker="o", label=label)
axes[0, 0].set(title="Training reward (10-episode mean)", xlabel="episode", ylabel="reward")
axes[0, 1].set(title="Training priority delivery AUC", xlabel="episode", ylabel="AUC")
axes[1, 0].set(title="Huber loss (10-episode mean)", xlabel="episode", ylabel="loss")
axes[1, 1].set(title="Fixed validation semantic utility AUC", xlabel="episode", ylabel="AUC")
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend()
fig.tight_layout(); plt.show()
'''),
    markdown("""
## 6. All-policy scheduler evaluation, if present

This section deliberately waits for the actual scheduler results. It compares policies only when the same evaluation artifact contains their outcomes. Normalized semantic utility AUC rewards useful early delivery; higher is better. Critical on-time rate is also higher-is-better. Weighted priority inversion and inference latency are lower-is-better. The paired-gains file measures differences on identical episodes against FIFO and EDF.
"""),
    code(r'''
policy_path = all_policy_path
if not policy_path.is_file():
    display(Markdown("**Unavailable in this download.** The Kaggle run was cancelled after classifier evaluation. No FIFO, EDF, semantic-greedy, ranker, or DQN test comparison is present; do not infer a winner from DQN validation scores."))
else:
    policy_metrics = pd.read_csv(policy_path)
    print("Source:", policy_path)
    print("Tracks:", policy_metrics.track.unique().tolist())
    print("Policies:", sorted(policy_metrics.policy.unique().tolist()))
    preferred = "image_disjoint_distinct"
    track = preferred if preferred in set(policy_metrics.track) else policy_metrics.track.iloc[0]
    comparison = policy_metrics[policy_metrics.track == track]
    numeric = [name for name in ("normalized_utility_auc", "critical_on_time_rate",
                                 "weighted_priority_inversion", "mean_policy_inference_ms")
               if name in comparison]
    table = comparison.groupby("policy")[numeric].agg(["mean", "std", "count"])
    display(Markdown(f"**Evaluation track:** `{track}`; means, spread, and number of episodes."))
    display(table.round(4))
    if "normalized_utility_auc" in numeric:
        means = comparison.groupby("policy").normalized_utility_auc.mean().sort_values()
        ax = means.plot.barh(figsize=(10, max(4, .38 * len(means))), color="#19647e")
        ax.set(title=f"Semantic utility AUC by policy — {track}", xlabel="mean normalized utility AUC", ylabel="policy")
        plt.tight_layout(); plt.show()
    paired_path = (run_root / "evaluation/dynamic_paired_priority_gains.csv"
                   if downloaded_policy_path.is_file()
                   else local_evaluation_root / "dynamic_paired_priority_gains.csv")
    if paired_path.is_file():
        paired = pd.read_csv(paired_path)
        columns = [name for name in ("gain_vs_fifo", "gain_vs_edf") if name in paired]
        paired_track = paired[paired.track == track] if "track" in paired else paired
        display(paired_track.groupby("policy")[columns].mean().round(4))
    pivot = comparison.pivot_table(index="episode_id", columns="policy",
                                   values="normalized_utility_auc", aggfunc="first")
    if {"fifo", "edf"}.issubset(pivot.columns):
        rng = np.random.default_rng(2026)
        effect_rows = []
        for policy in pivot.columns:
            for baseline in ("fifo", "edf"):
                differences = (pivot[policy] - pivot[baseline]).dropna().to_numpy()
                if len(differences) < 2:
                    continue
                samples = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
                effect_rows.append({"policy": policy, "baseline": baseline,
                                    "paired_gain": differences.mean(),
                                    "ci95_low": np.quantile(samples, .025),
                                    "ci95_high": np.quantile(samples, .975),
                                    "streams": len(differences)})
        display(Markdown("**Paired semantic utility gains.** Intervals resample traffic streams; the same 52 test images recur across streams."))
        display(pd.DataFrame(effect_rows).sort_values(["baseline", "paired_gain"], ascending=[True, False]).round(4))
'''),
    markdown("""
## 7. What can we say in the presentation?

Run the cell below for a short evidence ledger based on the actual downloaded artifacts. The key limitation is that simulated payload cost and prototype rubric labels constrain the interpretation of every result.
"""),
    code(r'''
observations = [
    f"The dataset contains {len(manifest)} unique images in {manifest.selection_stratum.nunique()} selection strata, split "
    + "/".join(str(int(v)) for v in manifest.split.value_counts().reindex(["train", "validation", "test"]).fillna(0))
    + " across train/validation/test.",
    f"Three DQN seeds trained for {seeds.minutes.sum():.1f} total minutes; seed {best_seed} had the highest recorded validation utility AUC ({seeds.best_validation_utility_auc.max():.4f}).",
    f"The best test image-only classifier macro F1 is {classifier_metrics[(classifier_metrics.split == 'test') & (classifier_metrics.model != 'feature_ordinal_oracle_assisted')].macro_f1.max():.3f}.",
    "The run used simulated fixed service units and a prototype priority rubric; no measured SwinJSCC reconstruction claim follows from it.",
]
if policy_path.is_file():
    observations.append("All-policy scheduler metrics are present; use the comparison table above for policy claims.")
else:
    observations.append("All-policy scheduler metrics are absent; this download cannot show whether DQN or semantic policies beat FIFO/EDF.")
display(Markdown("\n".join(f"- {sentence}" for sentence in observations)))
'''),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python"}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
