"""Build Notebooks/Edge/edge-simtable-fill: on Kaggle, fill the sim table from the SQ-GAN v1 pass (C2), run the D6
codec-quality report and fit the SPI+STS fusion. Output = a few small files (no 429 on download).

Inputs: kernel output of edge-sqgan-v1-resume (all 55 shards) + public dataset scheduler-dataset-v1-simtable.
    python -m scripts.edge.make_fill_notebook [--out-dir Notebooks/Edge/edge-simtable-fill]
"""
import argparse
import json
from pathlib import Path

OWNER = "satyapaladuguplus"
STEP = "edge-simtable-fill"
REPO = "https://github.com/saltypal/SemanticSchedulerEnd2End.git"

CELLS = [
    ("markdown", "# Edge simtable fill\n\nFill STS / payload from the SQ-GAN v1 pass, D6 codec-quality report, SPI+STS fusion.\n"),
    ("code", f'''import glob, os, subprocess, sys
subprocess.run(["git", "clone", "--depth", "1", "-q", "{REPO}", "/tmp/repo"], check=True)
print("repo @", subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/tmp/repo"], check=False)
c2 = sorted({{os.path.dirname(p) for p in glob.glob("/kaggle/input/**/cache_sqgan/shards", recursive=True)}})
tab = sorted(glob.glob("/kaggle/input/**/simtable_v1.jsonl", recursive=True))
print("C2:", c2, "\\nTABLE:", tab)
assert c2 and tab, "inputs not mounted"
n = len(glob.glob(c2[0] + "/shards/shard_*/done.json")); print("finished shards:", n)
'''),
    ("code", '''def run(*a):
    r = subprocess.run([sys.executable, "-m", *a], cwd="/tmp/repo", capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-2000:])
    if r.returncode: raise SystemExit("failed: " + " ".join(a))
run("parameters.edge.simtable", "fill-codec", "--table", tab[0], "--c2", c2[0], "--out", "/kaggle/working/tab")
run("Evaluation.edge.report", "codec-quality", "--c2", c2[0], "--table", "/kaggle/working/tab/simtable_v1.jsonl",
    "--out", "/kaggle/working/quality")
run("parameters.edge.sts_fusion", "--table", "/kaggle/working/tab", "--out", "/kaggle/working/fusion")
'''),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="Notebooks/Edge/" + STEP)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / (STEP + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{STEP}", "title": "Edge simtable fill", "code_file": STEP + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [f"{OWNER}/scheduler-dataset-v1-simtable"],
            "kernel_sources": [f"{OWNER}/edge-sqgan-v1-resume"], "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
