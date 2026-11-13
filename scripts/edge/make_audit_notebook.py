"""Build Notebooks/Edge/edge-dataset-audit: 50 stratified samples of the sim table shown as a pandas DataFrame and as
image thumbnails with their tier, SPI, relevance, mission and STS. For human auditing. CPU only.

    python -m scripts.edge.make_audit_notebook
"""
import json
from pathlib import Path

OWNER, STEP = "satyapaladuguplus", "edge-dataset-audit"
CODE = [
'''import glob, json, os, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 60); pd.set_option("display.width", 250)
tab = glob.glob("/kaggle/input/**/simtable_v1.jsonl", recursive=True)[0]
idx = glob.glob("/kaggle/input/**/scheduler-dataset-v1-images/**/index.jsonl", recursive=True)[0]
root = os.path.dirname(idx)
df = pd.read_json(tab, lines=True)
files = {}
for line in open(idx):
    r = json.loads(line); files[r.get("uid") or r.get("image_id")] = r.get("file") or r.get("path")
T = ["Critical", "High", "Medium", "Low"]
M = {0: "general", 1: "flood_storm", 2: "wildfire", 3: "earthquake"}
df["true_tier"] = df["tier"].map(dict(enumerate(T)))
df["spi_top"] = df["tier_probs"].map(lambda p: T[max(range(4), key=lambda i: p[i])])
df["spi"] = df["tier_probs"].map(lambda p: [round(x, 2) for x in p])
df["mission_name"] = df["mission"].map(M)
df["sts_hist7"] = df["sts"].map(lambda s: [round(x, 2) for x in s[:7]])
print(len(df), "rows"); display(df.groupby(["split", "true_tier"]).size().unstack())''',
'''# 50 samples: stratified by split x true tier (as balanced as the data allows), fixed seed
s = df.groupby(["split", "true_tier"], group_keys=False).apply(lambda g: g.sample(min(len(g), 5), random_state=7))
s = s.sample(min(50, len(s)), random_state=7).reset_index(drop=True)
cols = ["image_id", "split", "event", "source", "true_tier", "spi_top", "spi", "relevance", "mission_name",
        "payload_bytes", "sts_hist7", "dup_group", "capture_time_s"]
display(s[cols])''',
'''fig, axes = plt.subplots(10, 5, figsize=(22, 46))
for ax, (_, r) in zip(axes.ravel(), s.iterrows()):
    f = files.get(r["image_id"])
    try:
        ax.imshow(Image.open(os.path.join(root, f)).convert("RGB"))
    except Exception as e:
        ax.text(0.5, 0.5, "missing: %s" % e, ha="center")
    ax.set_title("%s | true %s | SPI %s\\nP=%s rel %.2f %s" % (r["split"], r["true_tier"], r["spi_top"], r["spi"],
                 r["relevance"], r["mission_name"]), fontsize=8)
    ax.axis("off")
plt.tight_layout(); plt.savefig("/kaggle/working/audit_grid.jpg", dpi=60); plt.show()
s[cols].to_csv("/kaggle/working/audit_50.csv", index=False)''',
]


def main():
    out = Path("Notebooks/Edge") / STEP
    out.mkdir(parents=True, exist_ok=True)
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# Edge dataset audit\n\n50 stratified samples of the "
              "sim table with their images.\n"]}]
    cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in CODE]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / (STEP + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{STEP}", "title": "Edge dataset audit", "code_file": STEP + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [f"{OWNER}/scheduler-dataset-v1-simtable", f"{OWNER}/scheduler-dataset-v1-images"],
            "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
