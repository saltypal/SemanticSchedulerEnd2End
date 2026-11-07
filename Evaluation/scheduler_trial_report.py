"""Build the controlled development replay figure and result table."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.replay.read_text(encoding="utf-8"))
    out = args.output
    rows = data["results"]
    policies = (
        ("fifo", "FIFO", "#475569", "o", "-"),
        ("static", "Static + exact duplicate gate", "#2563eb", "s", "-"),
        ("conditional", "Receiver-conditioned", "#ca5800", "x", "--"),
    )

    fig, ax = plt.subplots(figsize=(7.7, 4.4), layout="constrained")
    for key, label, color, marker, style in policies:
        series = [row for row in rows if row["policy"] == key]
        ax.plot([row["load"] for row in series],
                [row["timely_facts"] / row["offered_facts"] for row in series],
                label=label, color=color, marker=marker, linestyle=style, linewidth=1.8)
    loads = sorted({row["load"] for row in rows})
    fig.suptitle('Timely road-damage evidence in development replay')
    ax.set_title('24 images, injected exact repeats, synthetic arrivals, measured service', fontsize=10)
    ax.set(
           xlabel="Nominal offered load", ylabel="Fraction of offered positive image IDs",
           xticks=loads, ylim=(0, 1))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left")
    fig.savefig(out / "dev_replay_coverage.png", dpi=180)
    plt.close(fig)

    fields = [key for key in rows[0] if key != "order"]
    with (out / "dev_replay_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    names = ("day1_probe.json", "day1_balanced_probe.json", "crasar_training_linkage.json",
             "dev_exact_repeat_replay.json", "dev_balanced_replay.json",
             "dev_replay_coverage.png", "dev_replay_table.csv")
    extra_names = ("day1_variant_probe.json", "dev_variant_replay.json", "dev_selector_timing.json",
                   "crasar_training_condition_candidates.json", "crasar_training_crop_manifest.json",
                   "crasar_crop_hash_check.json", "crasar_repeat_cue_probe.json")
    names += tuple(name for name in extra_names if (out / name).is_file())
    probe = json.loads((out / "day1_balanced_probe.json").read_text(encoding="utf-8"))
    scores = probe["records"]
    calibration = {}
    for key in ("source", "header", "decoded"):
        positives = false_positives = 0
        for row in scores:
            threshold = max(other[key] for other in scores
                            if other["id"] != row["id"] and not other["native_road_damage"])
            if row[key] > threshold:
                positives += int(row["native_road_damage"])
                false_positives += int(not row["native_road_damage"])
        calibration[key] = {"true_positives": positives, "false_positives": false_positives}

    header_budget_ms = 6 * data["service_estimate_ms"]
    header_reports = [row for row in scores if row["header"] >= probe["threshold"]
                      and row["sender_ms"] <= header_budget_ms]
    header_only = {
        "scope": "Synthetic no-transport control. Sender compute counted; decoder and JPEG encoding bypassed.",
        "budget_ms": header_budget_ms,
        "timely_true_positive_images": sum(row["native_road_damage"] for row in header_reports),
        "timely_false_positive_images": sum(not row["native_road_damage"] for row in header_reports),
        "offered_positive_images": sum(row["native_road_damage"] for row in scores),
    }
    result = {
        "scope": "Development and data qualification only. No holdout opened.",
        "replay_source": args.replay.name,
        "header_only_control": header_only,
        "leave_one_image_out_development_threshold_diagnostic": calibration,
        "artifact_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in names},
    }
    (out / "development_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
