"""Create honest Stage 1A comparison plots from the exported metrics CSV.

The script compares these four communication representations at a selected SNR:

1. RAW RGB bits carried through the same CRC + rate-1/2 LDPC + QPSK accounting.
   This is an accounting reference only: it was not simulated frame-by-frame.
2. JPEG + LDPC + QPSK.
3. Native continuous-symbol SwinJSCC.
4. SwinJSCC latent K=256 VQ + LDPC + QPSK.

It deliberately keeps native JSCC and packetized digital methods visibly distinct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


RAW_RGB_BITS_PER_SAMPLE = 8
TRANSPORT_HEADER_BITS = 32
CRC_BITS = 32
LDPC_INFORMATION_BLOCK_BITS = 1024
LDPC_CODEWORD_BITS = 2048
QPSK_BITS_PER_SYMBOL = 2

SELECTED_JPEG_QUALITY = 20
SELECTED_NATIVE_RATE = 96
SELECTED_VQ_RATE = 96

COLORS = {
    "raw": "#444444",
    "jpeg": "#D97904",
    "native": "#1565C0",
    "vq": "#7B3F98",
}


@dataclass(frozen=True)
class Aggregate:
    method: str
    snr_db: float
    rate: int | None
    jpeg_quality: int | None
    trial_count: int
    success_count: int
    mean_channel_uses: float | None
    mean_psnr_db: float | None
    mean_ssim: float | None
    mean_cbr: float | None

    @property
    def success_rate(self) -> float:
        return self.success_count / self.trial_count if self.trial_count else 0.0


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    parsed = parse_optional_float(value)
    return None if parsed is None else int(parsed)


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: list[dict[str, str]]) -> list[Aggregate]:
    grouped: dict[tuple[str, float, int | None, int | None], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            float(row["snr_db"]),
            parse_optional_int(row.get("rate")),
            parse_optional_int(row.get("jpeg_quality")),
        )
        grouped[key].append(row)

    aggregates: list[Aggregate] = []
    for (method, snr_db, rate, jpeg_quality), group in grouped.items():
        successful = [
            row
            for row in group
            if row.get("frame_success", "").lower() == "true" and parse_optional_float(row.get("psnr")) is not None
        ]

        def average(field: str) -> float | None:
            values = [parse_optional_float(row.get(field)) for row in successful]
            values = [value for value in values if value is not None]
            return mean(values) if values else None

        aggregates.append(
            Aggregate(
                method=method,
                snr_db=snr_db,
                rate=rate,
                jpeg_quality=jpeg_quality,
                trial_count=len(group),
                success_count=len(successful),
                mean_channel_uses=average("channel_uses"),
                mean_psnr_db=average("psnr"),
                mean_ssim=average("ssim"),
                mean_cbr=average("cbr"),
            )
        )
    return aggregates


def find_aggregate(
    aggregates: list[Aggregate],
    method: str,
    snr_db: float,
    rate: int | None = None,
    jpeg_quality: int | None = None,
) -> Aggregate:
    matches = [
        row
        for row in aggregates
        if row.method == method
        and math.isclose(row.snr_db, snr_db)
        and row.rate == rate
        and row.jpeg_quality == jpeg_quality
    ]
    if len(matches) != 1:
        raise LookupError(
            f"Expected one aggregate for method={method}, snr={snr_db}, rate={rate}, quality={jpeg_quality}; "
            f"found {len(matches)}."
        )
    return matches[0]


def raw_reference_from_native(native: Aggregate) -> dict[str, float | int | str]:
    if native.mean_channel_uses is None or native.mean_cbr is None:
        raise ValueError("Native row needs channel uses and CBR to derive the raw image dimensions.")

    raw_rgb_samples = round(native.mean_channel_uses / native.mean_cbr)
    raw_rgb_bits = raw_rgb_samples * RAW_RGB_BITS_PER_SAMPLE
    framed_bits = raw_rgb_bits + TRANSPORT_HEADER_BITS + CRC_BITS
    ldpc_blocks = math.ceil(framed_bits / LDPC_INFORMATION_BLOCK_BITS)
    coded_bits = ldpc_blocks * LDPC_CODEWORD_BITS
    channel_uses = coded_bits // QPSK_BITS_PER_SYMBOL

    return {
        "method": "raw_rgb_ldpc_qpsk_reference",
        "raw_rgb_samples": raw_rgb_samples,
        "raw_rgb_bits": raw_rgb_bits,
        "framed_bits": framed_bits,
        "ldpc_blocks": ldpc_blocks,
        "ldpc_coded_bits": coded_bits,
        "channel_uses": channel_uses,
        "cbr": channel_uses / raw_rgb_samples,
        "interpretation": "Resource-only lossless reference; frame success was not simulated in Stage 1A.",
    }


def require_actual(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing {label} for a successful aggregate.")
    return float(value)


def style_axis(axis) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def add_value_labels(axis, bars, formatter) -> None:
    for bar in bars:
        value = bar.get_width()
        axis.text(
            value * 1.13,
            bar.get_y() + bar.get_height() / 2,
            formatter(value),
            va="center",
            fontsize=9,
        )


def plot_resource_budget(output: Path, raw: dict, jpeg: Aggregate, native: Aggregate, vq: Aggregate, snr_db: float) -> None:
    labels = [
        "RAW RGB + LDPC/QPSK\n(resource reference)",
        f"JPEG Q{SELECTED_JPEG_QUALITY} + LDPC/QPSK",
        f"Native SwinJSCC\nlatent rate {SELECTED_NATIVE_RATE}",
        f"SwinJSCC + K=256 VQ\nlatent rate {SELECTED_VQ_RATE}",
    ]
    uses = [
        float(raw["channel_uses"]),
        require_actual(jpeg.mean_channel_uses, "JPEG channel uses"),
        require_actual(native.mean_channel_uses, "native channel uses"),
        require_actual(vq.mean_channel_uses, "VQ channel uses"),
    ]
    colors = [COLORS["raw"], COLORS["jpeg"], COLORS["native"], COLORS["vq"]]

    figure, axis = plt.subplots(figsize=(10, 5.6))
    bars = axis.barh(labels, uses, color=colors, hatch=["//", "", "", ""])
    axis.set_xscale("log")
    axis.set_xlabel("Complex channel uses per 768×512 RGB image (log scale)")
    axis.set_title(f"Resource budget at Es/N0 = {snr_db:g} dB")
    add_value_labels(axis, bars, lambda value: f"{value:,.0f}")
    style_axis(axis)
    axis.text(
        0.01,
        -0.20,
        "RAW is calculated with the same 32-bit length header, 32-bit CRC, rate-1/2 LDPC, and QPSK accounting. "
        "It was not frame-simulated, so it is a resource reference rather than a measured PSNR result.",
        transform=axis.transAxes,
        fontsize=8.5,
        wrap=True,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_quality_resource_tradeoff(output: Path, aggregates: list[Aggregate], raw: dict, snr_db: float) -> None:
    native_rows = sorted(
        [row for row in aggregates if row.method == "native_swinjscc" and math.isclose(row.snr_db, snr_db)],
        key=lambda row: row.rate or 0,
    )
    jpeg_rows = sorted(
        [row for row in aggregates if row.method == "jpeg_ldpc_qpsk" and math.isclose(row.snr_db, snr_db)],
        key=lambda row: row.jpeg_quality or 0,
    )
    vq = find_aggregate(aggregates, "vq_k256_ldpc_qpsk", snr_db, rate=SELECTED_VQ_RATE)

    figure, axis = plt.subplots(figsize=(10, 6.2))
    axis.plot(
        [require_actual(row.mean_channel_uses, "native uses") for row in native_rows],
        [require_actual(row.mean_psnr_db, "native PSNR") for row in native_rows],
        color=COLORS["native"], marker="o", linewidth=2.4, label="Native SwinJSCC",
    )
    for row in native_rows:
        axis.annotate(f"r={row.rate}", (require_actual(row.mean_channel_uses, "native uses"), require_actual(row.mean_psnr_db, "native PSNR")), xytext=(5, 5), textcoords="offset points", fontsize=8)

    axis.plot(
        [require_actual(row.mean_channel_uses, "JPEG uses") for row in jpeg_rows],
        [require_actual(row.mean_psnr_db, "JPEG PSNR") for row in jpeg_rows],
        color=COLORS["jpeg"], marker="s", linewidth=2.4, label="JPEG + LDPC/QPSK",
    )
    for row in jpeg_rows:
        axis.annotate(f"Q{row.jpeg_quality}", (require_actual(row.mean_channel_uses, "JPEG uses"), require_actual(row.mean_psnr_db, "JPEG PSNR")), xytext=(5, 5), textcoords="offset points", fontsize=8)

    axis.scatter(
        [require_actual(vq.mean_channel_uses, "VQ uses")],
        [require_actual(vq.mean_psnr_db, "VQ PSNR")],
        color=COLORS["vq"], marker="D", s=70, label="SwinJSCC + K=256 VQ + LDPC/QPSK", zorder=3,
    )
    axis.annotate("K=256, r=96", (require_actual(vq.mean_channel_uses, "VQ uses"), require_actual(vq.mean_psnr_db, "VQ PSNR")), xytext=(5, -14), textcoords="offset points", fontsize=8)

    # RAW is represented as a capped, explicitly conditional reference because lossless PSNR is infinite.
    raw_reference_y = 42.5
    axis.scatter([float(raw["channel_uses"])], [raw_reference_y], color=COLORS["raw"], marker="*", s=180, hatch="//", label="RAW lossless reference (conditional)", zorder=3)
    axis.annotate("RAW: lossless if decoded\n(not frame-simulated)", (float(raw["channel_uses"]), raw_reference_y), xytext=(-112, 8), textcoords="offset points", fontsize=8, ha="right")

    axis.set_xscale("log")
    axis.set_ylim(18, 45)
    axis.set_xlabel("Complex channel uses per image (log scale)")
    axis.set_ylabel("Mean PSNR (dB); RAW marker is a capped lossless reference")
    axis.set_title(f"Quality–resource trade-off at Es/N0 = {snr_db:g} dB")
    style_axis(axis)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_snr_robustness(output: Path, aggregates: list[Aggregate]) -> None:
    selected = [
        ("Native SwinJSCC, r=96", "native_swinjscc", SELECTED_NATIVE_RATE, None, COLORS["native"], "o"),
        ("SwinJSCC + K=256 VQ, r=96", "vq_k256_ldpc_qpsk", SELECTED_VQ_RATE, None, COLORS["vq"], "D"),
        (f"JPEG Q{SELECTED_JPEG_QUALITY}", "jpeg_ldpc_qpsk", None, SELECTED_JPEG_QUALITY, COLORS["jpeg"], "s"),
    ]

    figure, (quality_axis, success_axis) = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, method, rate, quality, color, marker in selected:
        rows = sorted(
            [
                row
                for row in aggregates
                if row.method == method and row.rate == rate and row.jpeg_quality == quality
            ],
            key=lambda row: row.snr_db,
        )
        snr = [row.snr_db for row in rows]
        psnr = [row.mean_psnr_db for row in rows]
        success = [100.0 * row.success_rate for row in rows]
        quality_axis.plot(snr, psnr, marker=marker, color=color, linewidth=2.2, label=label)
        success_axis.plot(snr, success, marker=marker, color=color, linewidth=2.2, label=label)

    quality_axis.set_title("Reconstruction quality versus channel SNR")
    quality_axis.set_xlabel("Es/N0 (dB)")
    quality_axis.set_ylabel("Mean PSNR of successful frames (dB)")
    quality_axis.legend(fontsize=8)
    style_axis(quality_axis)

    success_axis.set_title("Digital-frame reliability versus channel SNR")
    success_axis.set_xlabel("Es/N0 (dB)")
    success_axis.set_ylabel("Successful frames (%)")
    success_axis.set_ylim(-5, 105)
    success_axis.legend(fontsize=8)
    style_axis(success_axis)
    figure.suptitle("RAW is omitted: its Stage 1A frame-success experiment was not run.", y=0.02, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_four_way_dashboard(output: Path, raw: dict, jpeg: Aggregate, native: Aggregate, vq: Aggregate, snr_db: float) -> None:
    labels = ["RAW\nreference", f"JPEG\nQ{SELECTED_JPEG_QUALITY}", "Native\nr=96", "VQ\nK=256, r=96"]
    uses = [float(raw["channel_uses"]), require_actual(jpeg.mean_channel_uses, "JPEG uses"), require_actual(native.mean_channel_uses, "native uses"), require_actual(vq.mean_channel_uses, "VQ uses")]
    psnr = [None, require_actual(jpeg.mean_psnr_db, "JPEG PSNR"), require_actual(native.mean_psnr_db, "native PSNR"), require_actual(vq.mean_psnr_db, "VQ PSNR")]
    colors = [COLORS["raw"], COLORS["jpeg"], COLORS["native"], COLORS["vq"]]

    figure, (resource_axis, quality_axis) = plt.subplots(1, 2, figsize=(13, 5.4))
    bars = resource_axis.bar(labels, uses, color=colors, hatch=["//", "", "", ""])
    resource_axis.set_yscale("log")
    resource_axis.set_ylabel("Complex channel uses per image (log scale)")
    resource_axis.set_title(f"Resource requirement at {snr_db:g} dB")
    for bar, value in zip(bars, uses):
        resource_axis.text(bar.get_x() + bar.get_width() / 2, value * 1.35, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    style_axis(resource_axis)

    actual_labels = labels[1:]
    actual_psnr = [value for value in psnr[1:] if value is not None]
    actual_colors = colors[1:]
    bars = quality_axis.bar(actual_labels, actual_psnr, color=actual_colors)
    quality_axis.set_ylim(0, 45)
    quality_axis.set_ylabel("Mean PSNR of successful frames (dB)")
    quality_axis.set_title("Measured visual quality at 10 dB")
    for bar, value in zip(bars, actual_psnr):
        quality_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    quality_axis.text(0.5, 0.05, "RAW is lossless only if its digital frame decodes;\nno RAW frame experiment exists in the CSV.", transform=quality_axis.transAxes, ha="center", fontsize=8)
    style_axis(quality_axis)
    figure.suptitle("Stage 1A four-way comparison: RAW is a resource-only reference", fontsize=13)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_summary(path: Path, raw: dict, jpeg: Aggregate, native: Aggregate, vq: Aggregate) -> None:
    rows = [
        {
            "method": "RAW RGB + LDPC/QPSK reference",
            "channel_uses": raw["channel_uses"],
            "psnr_db": "lossless if decoded; not simulated",
            "ssim": "1.0 if decoded; not simulated",
            "frame_success_rate": "not measured",
            "notes": raw["interpretation"],
        },
        {
            "method": f"JPEG Q{SELECTED_JPEG_QUALITY} + LDPC/QPSK",
            "channel_uses": jpeg.mean_channel_uses,
            "psnr_db": jpeg.mean_psnr_db,
            "ssim": jpeg.mean_ssim,
            "frame_success_rate": jpeg.success_rate,
            "notes": "Measured Stage 1A CSV average at 10 dB.",
        },
        {
            "method": f"Native SwinJSCC, rate {SELECTED_NATIVE_RATE}",
            "channel_uses": native.mean_channel_uses,
            "psnr_db": native.mean_psnr_db,
            "ssim": native.mean_ssim,
            "frame_success_rate": native.success_rate,
            "notes": "Continuous-symbol JSCC. The current Stage 1A export is SA+RA, not a separately evaluated Base checkpoint.",
        },
        {
            "method": f"SwinJSCC + K=256 VQ + LDPC/QPSK, rate {SELECTED_VQ_RATE}",
            "channel_uses": vq.mean_channel_uses,
            "psnr_db": vq.mean_psnr_db,
            "ssim": vq.mean_ssim,
            "frame_success_rate": vq.success_rate,
            "notes": "Measured Stage 1A CSV average at 10 dB.",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_methodology_note(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Four-way Stage 1A comparison methodology",
                "",
                "RAW is not an executed Stage 1A method. It is calculated from the native CBR-derived Kodak image size,",
                "then framed with the same 32-bit length header and 32-bit CRC as the digital baseline, padded to",
                "1024 information-bit blocks, encoded into 2048-bit rate-1/2 LDPC codewords, and mapped by QPSK.",
                "Therefore it is a resource-only lossless reference. It has no measured frame-success or PSNR curve.",
                "",
                "JPEG Q20 is selected for the primary four-way comparison because its mean PSNR at 10 dB is close",
                "to native SwinJSCC rate 96. VQ rate 96 remains in the comparison even though its PSNR is lower,",
                "because the plot is meant to show the resource-quality trade-off rather than hide its degradation.",
                "",
                "Native SwinJSCC is continuous-symbol JSCC. VQ and JPEG are packetized 5G-LDPC/QPSK paths.",
                "Do not interpret the plot as paper-exact replication or as a semantic-task evaluation.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Path to stage1a_metrics.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plot and summary outputs")
    parser.add_argument("--snr-db", type=float, default=10.0, help="SNR operating point for the four-way plots")
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metrics(arguments.metrics)
    aggregates = aggregate_rows(rows)

    native = find_aggregate(aggregates, "native_swinjscc", arguments.snr_db, rate=SELECTED_NATIVE_RATE)
    jpeg = find_aggregate(aggregates, "jpeg_ldpc_qpsk", arguments.snr_db, jpeg_quality=SELECTED_JPEG_QUALITY)
    vq = find_aggregate(aggregates, "vq_k256_ldpc_qpsk", arguments.snr_db, rate=SELECTED_VQ_RATE)
    raw = raw_reference_from_native(native)

    plot_resource_budget(arguments.output_dir / "01_resource_budget_10db.png", raw, jpeg, native, vq, arguments.snr_db)
    plot_quality_resource_tradeoff(arguments.output_dir / "02_quality_vs_channel_uses_10db.png", aggregates, raw, arguments.snr_db)
    plot_snr_robustness(arguments.output_dir / "03_snr_robustness_and_frame_success.png", aggregates)
    plot_four_way_dashboard(arguments.output_dir / "04_four_way_summary_dashboard.png", raw, jpeg, native, vq, arguments.snr_db)
    write_summary(arguments.output_dir / "four_way_summary_10db.csv", raw, jpeg, native, vq)
    write_methodology_note(arguments.output_dir / "README_methodology.txt")

    with (arguments.output_dir / "plot_inputs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics": str(arguments.metrics.resolve()),
                "selected_snr_db": arguments.snr_db,
                "selected_jpeg_quality": SELECTED_JPEG_QUALITY,
                "selected_native_rate": SELECTED_NATIVE_RATE,
                "selected_vq_rate": SELECTED_VQ_RATE,
                "raw_reference": raw,
                "jpeg": asdict(jpeg),
                "native": asdict(native),
                "vq": asdict(vq),
            },
            handle,
            indent=2,
        )

    print("Created:")
    for path in sorted(arguments.output_dir.iterdir()):
        print(" -", path)
    print("\nFour-way summary:")
    print(f" RAW reference uses: {raw['channel_uses']:,}")
    print(f" JPEG Q20 uses / PSNR: {jpeg.mean_channel_uses:,.0f} / {jpeg.mean_psnr_db:.2f} dB")
    print(f" Native r96 uses / PSNR: {native.mean_channel_uses:,.0f} / {native.mean_psnr_db:.2f} dB")
    print(f" VQ r96 uses / PSNR: {vq.mean_channel_uses:,.0f} / {vq.mean_psnr_db:.2f} dB")


if __name__ == "__main__":
    main()
