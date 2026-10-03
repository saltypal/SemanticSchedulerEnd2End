"""Plot Stage 1A PSNR, SSIM, CBR, and resource use across the SNR sweep."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


COLORS = {
    "native_r32": "#4C78A8",
    "native_r96": "#1565C0",
    "native_r192": "#0B3D91",
    "vq_r96": "#7B3F98",
    "jpeg_q20": "#D97904",
    "raw": "#444444",
}


def optional_float(value: str | None) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    return float(value)


def optional_int(value: str | None) -> int | None:
    parsed = optional_float(value)
    return None if parsed is None else int(parsed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_series(row: dict[str, str]) -> str | None:
    method = row["method"]
    rate = optional_int(row.get("rate"))
    quality = optional_int(row.get("jpeg_quality"))
    if method == "native_swinjscc" and rate in (32, 96, 192):
        return f"native_r{rate}"
    if method == "vq_k256_ldpc_qpsk" and rate == 96:
        return "vq_r96"
    if method == "jpeg_ldpc_qpsk" and quality == 20:
        return "jpeg_q20"
    return None


def aggregate(rows: list[dict[str, str]]) -> dict[str, dict[float, dict[str, float | None]]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        series = selected_series(row)
        if series is not None:
            grouped[(series, float(row["snr_db"]))].append(row)

    output: dict[str, dict[float, dict[str, float | None]]] = defaultdict(dict)
    for (series, snr_db), group in grouped.items():
        successful = [
            row
            for row in group
            if row.get("frame_success", "").lower() == "true"
            and optional_float(row.get("psnr")) is not None
        ]

        def average(field: str) -> float | None:
            values = [optional_float(row.get(field)) for row in successful]
            values = [value for value in values if value is not None]
            return mean(values) if values else None

        output[series][snr_db] = {
            "psnr": average("psnr"),
            "ssim": average("ssim"),
            "cbr": average("cbr"),
            "channel_uses": average("channel_uses"),
            "success_rate": sum(
                row.get("frame_success", "").lower() == "true" for row in group
            ) / len(group),
        }
    return output


def style_axis(axis) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("Es/N0 SNR (dB)")


def plot_curves(output: Path, data: dict[str, dict[float, dict[str, float | None]]], raw_cbr: float) -> None:
    series_order = ["native_r32", "native_r96", "native_r192", "vq_r96", "jpeg_q20"]
    labels = {
        "native_r32": "Native SwinJSCC, rate 32",
        "native_r96": "Native SwinJSCC, rate 96",
        "native_r192": "Native SwinJSCC, rate 192",
        "vq_r96": "SwinJSCC + K=256 VQ, rate 96",
        "jpeg_q20": "JPEG Q20 + LDPC/QPSK",
    }

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_axes = {
        "psnr": (axes[0, 0], "PSNR versus SNR", "Mean PSNR of successful frames (dB)"),
        "ssim": (axes[0, 1], "SSIM versus SNR", "Mean SSIM of successful frames"),
        "cbr": (axes[1, 0], "CBR versus SNR", "CBR = channel uses / raw RGB samples"),
        "channel_uses": (axes[1, 1], "Channel uses versus SNR", "Complex channel uses per image"),
    }

    for metric, (axis, title, ylabel) in metric_axes.items():
        for series in series_order:
            values = data.get(series, {})
            points = sorted(
                (snr_db, record[metric])
                for snr_db, record in values.items()
                if record[metric] is not None
            )
            if not points:
                continue
            x_values, y_values = zip(*points)
            marker = "o" if series.startswith("native") else ("D" if series == "vq_r96" else "s")
            axis.plot(
                x_values,
                y_values,
                marker=marker,
                linewidth=2.1,
                color=COLORS[series],
                label=labels[series],
            )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        style_axis(axis)

    axes[1, 0].axhline(
        raw_cbr,
        color=COLORS["raw"],
        linestyle="--",
        linewidth=1.5,
        label="RAW + LDPC/QPSK resource reference",
    )
    axes[1, 0].annotate(
        "RAW reference\n(not SNR-simulated)",
        xy=(10, raw_cbr),
        xytext=(10.4, raw_cbr * 0.72),
        fontsize=8,
        color=COLORS["raw"],
    )

    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8, loc="best")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].legend(fontsize=8, loc="best")
    axes[1, 0].legend(fontsize=8, loc="best")

    figure.suptitle(
        "Stage 1A SNR sweep: quality and normalized resource consumption\n"
        "Digital-frame failures are omitted from PSNR/SSIM means and remain visible through missing points.",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_summary(path: Path, data: dict[str, dict[float, dict[str, float | None]]]) -> None:
    fields = ["series", "snr_db", "psnr", "ssim", "cbr", "channel_uses", "success_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for series, by_snr in sorted(data.items()):
            for snr_db, record in sorted(by_snr.items()):
                writer.writerow({"series": series, "snr_db": snr_db, **record})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(arguments.metrics)
    data = aggregate(rows)
    native_reference = data["native_r96"][10.0]
    native_uses = native_reference["channel_uses"]
    native_cbr = native_reference["cbr"]
    if native_uses is None or native_cbr is None:
        raise ValueError("Cannot derive RAW CBR reference from native rate-96 rows.")
    raw_cbr = (native_uses / native_cbr) * 8 / (native_uses / 2)
    # The expression above is equivalent to raw coded uses / raw RGB samples after
    # applying the fixed Stage 1A framing and rate-1/2 LDPC block accounting below.
    raw_rgb_samples = round(native_uses / native_cbr)
    raw_rgb_bits = raw_rgb_samples * 8
    framed_bits = raw_rgb_bits + 32 + 32
    blocks = math.ceil(framed_bits / 1024)
    raw_channel_uses = blocks * (2048 // 2)
    raw_cbr = raw_channel_uses / raw_rgb_samples

    output_plot = arguments.output_dir / "snr_psnr_ssim_cbr_channel_uses.png"
    plot_curves(output_plot, data, raw_cbr)
    write_summary(arguments.output_dir / "snr_metric_curves.csv", data)
    (arguments.output_dir / "README.txt").write_text(
        "SNR is the horizontal independent variable in every panel.\n"
        "RAW is shown only as a calculated CBR/resource reference because no RAW frame-level\n"
        "digital transmission experiment was run in Stage 1A. Failed digital frames are not\n"
        "assigned PSNR or SSIM. Native rows are continuous-symbol JSCC rows.\n",
        encoding="utf-8",
    )
    print("Created:")
    print(output_plot)
    print(arguments.output_dir / "snr_metric_curves.csv")


if __name__ == "__main__":
    main()
