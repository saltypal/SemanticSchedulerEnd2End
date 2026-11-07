"""Create a presentation figure from the teammate's static notebook and local dynamic run."""

from __future__ import annotations

import argparse
from html import unescape
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STATIC_DEFAULT = Path(r"C:\Users\satya\Downloads\03_LADI_v2_Baseline_Scheduling.ipynb")
DYNAMIC_DEFAULT = ROOT / "results/scheduler-policy-local/test30_core/all_policy_metrics.csv"
OUTPUT_DEFAULT = ROOT / "results/scheduler-review"


def static_table_from_notebook(path: Path) -> pd.DataFrame:
    notebook = nbformat.read(path, as_version=4)
    html_table = None
    for cell in notebook.cells:
        if "ladi_v2_scheduler_comparison.csv" not in cell.source:
            continue
        for output in cell.get("outputs", []):
            html_table = output.get("data", {}).get("text/html")
            if html_table:
                break
    if not html_table:
        raise ValueError("Executed static comparison table is absent from the notebook")
    raw_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_table, flags=re.S)
    rows = [
        [unescape(re.sub(r"<[^>]+>", "", cell)).strip()
         for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)]
        for row in raw_rows
    ]
    header = rows[0][1:]
    frame = pd.DataFrame([row[1:] for row in rows[1:]], columns=header)
    frame["nDCG"] = pd.to_numeric(frame["nDCG"])
    frame["mismatches"] = pd.to_numeric(frame["mismatches"])
    frame["pair_inversions"] = pd.to_numeric(frame["pair_inversions"])
    return frame


def bootstrap_interval(values: np.ndarray, random_source: np.random.Generator) -> tuple[float, float]:
    samples = random_source.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [.025, .975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-notebook", type=Path, default=STATIC_DEFAULT)
    parser.add_argument("--dynamic-metrics", type=Path, default=DYNAMIC_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    static = static_table_from_notebook(args.static_notebook)
    dynamic = pd.read_csv(args.dynamic_metrics)
    if dynamic.episode_id.nunique() != 30 or dynamic.policy.nunique() != 17:
        raise ValueError("Expected completed 30-stream, 17-policy dynamic comparison")
    if dynamic.duplicated(["episode_id", "policy"]).any():
        raise ValueError("Duplicate dynamic policy/episode rows")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    static.to_csv(output / "static_results_from_teammate_notebook.csv", index=False)

    categories = {
        "conventional": "#8b98a8", "semantic rule": "#16a6a1",
        "learned": "#5b58bd", "reference": "#dfa041",
    }
    selected = [
        ("fifo", "FIFO", "conventional"),
        ("edf", "EDF", "conventional"),
        ("ue_round_robin", "UE round robin", "conventional"),
        ("semantic_greedy", "Semantic greedy", "semantic rule"),
        ("ladi_classifier_priority", "LADI classifier", "learned"),
        ("feature_ordinal_classifier", "Feature classifier*", "learned"),
        ("pairwise_linear_ranker", "Pairwise ranker*", "learned"),
        ("dqn", "DQN*", "learned"),
        ("clip_image_priority", "CLIP image classifier", "learned"),
        ("clip_image_priority_edf", "CLIP classifier + EDF", "learned"),
        ("privileged_marginal_utility", "Privileged reference", "reference"),
    ]
    rng = np.random.default_rng(2026)
    dynamic_rows = []
    for policy, label, category in selected:
        rows = dynamic.loc[dynamic.policy == policy]
        if rows.episode_id.nunique() != 30:
            raise ValueError(f"Missing episode results for {policy}")
        auc = rows.normalized_utility_auc.to_numpy(dtype=float)
        lower, upper = bootstrap_interval(auc, rng)
        dynamic_rows.append({
            "policy": policy, "label": label, "category": category,
            "utility_auc": auc.mean(), "ci_low": lower, "ci_high": upper,
            "critical_on_time": rows.critical_on_time_rate.mean(),
        })
    dynamic_view = pd.DataFrame(dynamic_rows).sort_values("utility_auc")
    dynamic_view.to_csv(output / "dynamic_presentation_summary.csv", index=False)

    static_labels = {
        "FCFS / FIFO": "FIFO", "Earliest Deadline First (EDF)": "EDF",
        "Round Robin completion": "Round robin",
        "Shortest Processing Time (SPT)": "Shortest job",
        "Weighted SPT (semantic density)": "Semantic density",
        "Static semantic priority": "Declared-priority reference",
    }
    static_view = static.assign(label=static.algorithm.map(static_labels)).sort_values("nDCG")
    static_colors = [categories["reference"] if "reference" in label else
                     categories["semantic rule"] if "Semantic" in label else
                     categories["conventional"] for label in static_view.label]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fafbfd"})
    figure = plt.figure(figsize=(18, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[.88, 1.12], height_ratios=[1, 1],
                              wspace=.25, hspace=.20)
    left = figure.add_subplot(grid[:, 0])
    right_top = figure.add_subplot(grid[0, 1])
    right_bottom = figure.add_subplot(grid[1, 1], sharey=right_top)

    left.barh(static_view.label, static_view.nDCG, color=static_colors, height=.68)
    left.set(xlim=(0, 1.13), xlabel="nDCG to declared order  ·  higher is better",
             title="A  STATIC: one fixed batch of 12 images")
    for y, value in enumerate(static_view.nDCG):
        left.text(value + .018, y, f"{value:.3f}", va="center", fontsize=10)

    ypos = np.arange(len(dynamic_view))
    colors = [categories[category] for category in dynamic_view.category]
    right_top.barh(ypos, dynamic_view.utility_auc, color=colors, height=.72)
    right_top.errorbar(dynamic_view.utility_auc, ypos,
                       xerr=[dynamic_view.utility_auc - dynamic_view.ci_low,
                             dynamic_view.ci_high - dynamic_view.utility_auc],
                       fmt="none", ecolor="#17212e", capsize=2.5, lw=1)
    right_top.set_yticks(ypos, dynamic_view.label)
    right_top.set(xlim=(0, .56), xlabel="mean normalized semantic utility AUC  ·  higher is better",
                  title="B  DYNAMIC: 30 arrival streams, 52 held-out images each")
    for y, value in enumerate(dynamic_view.utility_auc):
        right_top.text(value + .012, y, f"{value:.3f}", va="center", fontsize=9)

    right_bottom.barh(ypos, dynamic_view.critical_on_time, color=colors, height=.72)
    right_bottom.set_yticks(ypos, dynamic_view.label)
    right_bottom.set(xlim=(0, 1.13), xlabel="critical / emergency delivered before deadline  ·  higher is better",
                     title="C  DEADLINE CHECK: same dynamic streams")
    for y, value in enumerate(dynamic_view.critical_on_time):
        right_bottom.text(value + .018, y, f"{value:.0%}", va="center", fontsize=9)
    for axis in (left, right_top, right_bottom):
        axis.grid(axis="x", alpha=.16)
        axis.set_axisbelow(True)

    legend = [Patch(color=color, label=kind.title()) for kind, color in categories.items()]
    figure.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
                  bbox_to_anchor=(.51, 1.07))
    figure.suptitle("Semantic scheduling: from fixed ordering to changing queues",
                     fontsize=20, fontweight="bold", y=1.125)
    figure.text(.02, -.015,
        "Static: ranking agreement with the teammate's declared 12-image order; the reference is perfect by construction.  "
        "Dynamic: shared seeded arrivals/outcomes, five UEs, successful-delivery history, simulated fixed service cost.\n"
        "Dynamic whiskers: 95% bootstrap intervals across 30 traffic streams using the same 52 held-out images.  "
        "* Uses rubric-assisted P5/P6.  Panels measure different objectives and their numeric scores are not directly comparable.",
        ha="left", va="top", fontsize=9, color="#3e4855")
    for suffix in ("png", "pdf"):
        path = output / f"static_vs_dynamic_scheduler_comparison.{suffix}"
        figure.savefig(path, dpi=210, bbox_inches="tight", facecolor=figure.get_facecolor())
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
