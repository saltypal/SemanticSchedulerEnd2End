"""Append one compact measured-comparison cell to the Kodak-24 notebook."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TAG = "compact-kodak24-comparison-v1"
ORDER = [
    "RAW RGB24", "JPEG Q50", "DeepJSCC", "DeepJSCC + codebook",
    "NTSCC", "NTSCC + codebook", "SwinJSCC", "SwinJSCC + codebook",
]


def build(csv_path: Path):
    raw = pd.read_csv(csv_path)
    keep = (raw.method != "JPEG") | raw.operating_point.astype(str).str.startswith("quality=50;")
    raw = raw[keep].copy()
    labels = {
        "RAW_RGB24": "RAW RGB24", "JPEG": "JPEG Q50",
        "DeepJSCC": "DeepJSCC", "DeepJSCC codebook": "DeepJSCC + codebook",
        "NTSCC": "NTSCC", "NTSCC codebook": "NTSCC + codebook",
        "SwinJSCC": "SwinJSCC", "SwinJSCC codebook": "SwinJSCC + codebook",
    }
    rows = []
    for method, group in raw.groupby("method", sort=False):
        if method not in labels:
            continue
        mean = lambda column: pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        ssim = mean("ssim")
        cbr = mean("cbr")
        ratio = mean("compression_ratio")
        digital = np.isfinite(ratio)
        compression = ratio if digital else (1.0 / cbr if np.isfinite(cbr) and cbr > 0 else 1.0)
        payload_bits = mean("payload_bits")
        rows.append({
            "Method": labels[method],
            "PSNR (dB) ↑": "∞" if method == "RAW_RGB24" else f"{mean('psnr_db'):.3f}",
            "SSIM ↑": f"{ssim:.4f}",
            "LPIPS ↓": f"{mean('lpips'):.4f}",
            "CBR ↓": "—" if not np.isfinite(cbr) else f"{cbr:.5f}",
            "Total time/image ↓": f"{mean('total_time_ms') / 1000:.3f} s",
            "Quality loss ↓": f"{(1.0 - ssim) * 100:.2f}%",
            "Quality retained ↑": f"{ssim * 100:.2f}%",
            "Compression vs original ↑": f"{compression:.2f}× ({'payload' if digital else 'channel uses'})",
            "Payload/image ↓": (
                "Analog symbols; no bit payload" if not np.isfinite(payload_bits)
                else f"{payload_bits / 8 / 1024:.2f} KiB"
            ),
            "Model size ↓": f"{mean('model_size_mb'):.2f} MB",
            "_retained": ssim * 100,
            "_loss": (1.0 - ssim) * 100,
            "_compression": compression,
            "_basis": "payload bits" if digital else "channel uses",
        })
    frame = pd.DataFrame(rows).set_index("Method").loc[ORDER].reset_index()

    table = frame[[column for column in frame.columns if not column.startswith("_")]]
    html = """<style>
    .compact-wrap{overflow-x:auto;max-width:100%;font-family:Arial,sans-serif;background:#111417;padding:12px;border-radius:8px}
    .compact-table{border-collapse:collapse;font-size:12px;white-space:nowrap;color:#f1f4f6;width:100%}
    .compact-table th{background:#17324d!important;color:#ffffff!important;padding:9px 8px;border:1px solid #52687a;text-align:center}
    .compact-table td{background:#191e23!important;color:#f1f4f6!important;padding:8px;border:1px solid #3c4852;text-align:right}
    .compact-table td:first-child{text-align:left;font-weight:600}
    .compact-table tbody tr:nth-child(even) td{background:#222a31!important}
    .compact-table tbody tr:nth-child(4) td,
    .compact-table tbody tr:nth-child(6) td,
    .compact-table tbody tr:nth-child(8) td{background:#20364a!important;color:#dcedff!important}
    .compact-table tbody tr:hover td{background:#2c465d!important}
    .compact-note{font-size:12px;color:#c8d1d8;line-height:1.5;margin:10px 2px 2px}
    </style><div class="compact-wrap">""" + table.to_html(index=False, classes="compact-table", border=0, escape=False) + """
    <p class="compact-note"><b>Quality retained = SSIM × 100%; quality loss = (1 − SSIM) × 100%.</b>
    Digital compression uses original RGB24 bits / serialized payload bits. Native analog JSCC has no serialized bit payload,
    so its displayed compression is 1/CBR and is explicitly labeled channel-use based. Results are means over all 24 Kodak images.</p></div>"""

    colors = ["#7f8c8d", "#e67e22", "#3498db", "#85c1e9", "#27ae60", "#7dcea0", "#8e44ad", "#c39bd3"]
    fig1, ax1 = plt.subplots(figsize=(12, 5.5))
    bars = ax1.bar(frame.Method, frame._compression, color=colors, edgecolor="#263238")
    ax1.set_yscale("log")
    ax1.set_ylabel("Compression ratio vs original RGB24 (×, log scale)")
    ax1.set_title("Compression achieved by each representation")
    ax1.tick_params(axis="x", rotation=28)
    ax1.grid(axis="y", alpha=.25, which="both")
    for bar, value, basis in zip(bars, frame._compression, frame._basis):
        ax1.text(bar.get_x() + bar.get_width()/2, value * 1.08, f"{value:.1f}×\n{basis}", ha="center", va="bottom", fontsize=8)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 5.5))
    ax2.bar(frame.Method, frame._retained, label="Quality retained", color="#2e86c1")
    ax2.bar(frame.Method, frame._loss, bottom=frame._retained, label="Quality loss", color="#e74c3c")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("SSIM-derived quality share (%)")
    ax2.set_title("Reconstruction quality retained and lost")
    ax2.tick_params(axis="x", rotation=28)
    ax2.legend(ncol=2, loc="upper center")
    ax2.grid(axis="y", alpha=.25)
    for x, retained in enumerate(frame._retained):
        ax2.text(x, retained / 2, f"{retained:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    fig2.tight_layout()
    return frame, html, fig1, fig2


def png_output(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}, "metadata": {}, "output_type": "display_data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--remove-prior-from", type=Path)
    args = parser.parse_args()
    if args.remove_prior_from:
        prior = json.loads(args.remove_prior_from.read_text(encoding="utf-8"))
        old_tag = "kodak24-research-comparison-v1"
        before = len(prior["cells"])
        prior["cells"] = [cell for cell in prior["cells"] if old_tag not in cell.get("metadata", {}).get("tags", [])]
        args.remove_prior_from.write_text(json.dumps(prior, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Removed {before - len(prior['cells'])} prior comparison cells from {args.remove_prior_from}")
    frame, html, fig1, fig2 = build(args.csv)
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    notebook["cells"] = [cell for cell in notebook["cells"] if TAG not in cell.get("metadata", {}).get("tags", [])]
    source = '''# Compact comparison generated only from saved, measured Kodak-24 rows.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

repo_candidates = [Path.cwd(), Path.cwd().parent]
repo_root = next(p for p in repo_candidates if (p / "results" / "kodak24_benchmark.csv").exists())
csv_path = repo_root / "results" / "kodak24_benchmark.csv"

# The helper contains the exact aggregation and plotting definitions used for this saved output.
import importlib.util
helper_path = repo_root / "scripts" / "update_kodak_compact_comparison.py"
spec = importlib.util.spec_from_file_location("compact_comparison", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
comparison, table_html, compression_fig, quality_fig = helper.build(csv_path)
display(HTML(table_html))
display(compression_fig)
display(quality_fig)
'''
    cell = {
        "cell_type": "code", "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [
            {"data": {"text/html": [html]}, "metadata": {}, "output_type": "display_data"},
            png_output(fig1), png_output(fig2),
        ],
        "source": source.splitlines(keepends=True),
    }
    notebook["cells"].append(cell)
    args.notebook.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Appended one compact cell with {len(frame)} rows and two graphs to {args.notebook}")


if __name__ == "__main__":
    main()
