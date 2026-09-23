"""Render the six-model Kodak-24 comparison as a PNG."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("DeepJSCC", "DeepJSCC codebook", "NTSCC", "NTSCC codebook", "SwinJSCC", "SwinJSCC codebook")

def fmt(value, places=3):
    return "—" if pd.isna(value) else f"{value:.{places}f}"

def latest_method_rows(data, method):
    rows = data[data.method.eq(method)].copy()
    if rows.empty: return rows
    rows["timestamp_utc"] = pd.to_datetime(rows.timestamp_utc, errors="coerce")
    run_id = rows.sort_values("timestamp_utc").iloc[-1].run_id
    rows = rows[rows.run_id.eq(run_id)]
    if rows.image_id.nunique() != 24: raise ValueError(f"Incomplete {method} run")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "kodak24_benchmark.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "kodak24_model_comparison.png")
    args = parser.parse_args(); data = pd.read_csv(args.csv); table_rows = []
    for method in METHODS:
        rows = latest_method_rows(data, method)
        if rows.empty:
            table_rows.append([method, "—", "—", "—", "—", "—", "—", "—", "Not evaluated"]); continue
        mean = rows.mean(numeric_only=True)
        time = f"{mean.total_time_ms / 1000:.2f} s" if mean.total_time_ms >= 1000 else f"{mean.total_time_ms:.0f} ms"
        table_rows.append([method, fmt(mean.psnr_db), fmt(mean.ssim, 4), fmt(mean.lpips, 4), fmt(mean.cbr, 5), time, f"{mean.model_size_mb:.2f} MB", f"{mean.peak_cpu_rss_delta_mb:.2f} MB", "Measured"])
    fig, ax = plt.subplots(figsize=(15.5, 6.1), dpi=220); fig.patch.set_facecolor("#171717"); ax.axis("off")
    headers = ["Method", "PSNR\ndB", "SSIM", "LPIPS\n↓", "CBR\n↓", "Total\ntime/image", "Model\nsize", "Peak RSS\nΔ", "Status"]
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="left", colLoc="left", colWidths=[.205,.085,.085,.09,.09,.145,.11,.115,.1], bbox=[.015,.18,.97,.62]); table.auto_set_font_size(False); table.set_fontsize(10)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor("#303030"); cell.set_linewidth(.5); cell.set_facecolor("#171717"); cell.set_text_props(color="#E0E0E0")
        if r == 0: cell.set_text_props(color="#C8C8C8", weight="bold")
        elif table_rows[r-1][0].startswith("SwinJSCC"):
            cell.set_facecolor("#1E2B38")
            if c == 0: cell.set_text_props(color="#A9D5FF", weight="bold")
        elif table_rows[r-1][-1] == "Not evaluated": cell.set_text_props(color="#8F8F8F")
    fig.text(.02,.92,"Kodak-24 results",fontsize=17,weight="bold",color="#E6E6E6")
    fig.text(.02,.874,"Latest complete 24-image average for each model on CPU",fontsize=10.5,color="#B5B5B5")
    fig.text(.02,.055,"Rows may come from different completed runs. CBR is channel uses per source pixel; lower is better.",fontsize=8.7,color="#9E9E9E")
    args.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output,bbox_inches="tight",facecolor=fig.get_facecolor()); print(f"Saved {args.output}")
if __name__ == "__main__": main()
