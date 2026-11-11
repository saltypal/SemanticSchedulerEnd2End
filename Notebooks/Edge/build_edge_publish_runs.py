"""Builds the helper notebook edge-publish-runs: copies dataset-a, dataset-b and cache from the
dataset-semschedule notebook output (attached as kernel_sources) and publishes each as a Kaggle dataset.
Run: python Notebooks/Edge/build_edge_publish_runs.py <out_dir>   (writes notebook + kernel-metadata.json)
"""
import json
import sys
from pathlib import Path

OWNER = "satyapaladuguplus"
SOURCE_KERNEL = f"{OWNER}/dataset-semschedule"

INTRO = """# Edge publish runs (helper)

Attaches the output of `dataset-semschedule` (kernel_sources) and publishes its `run/` parts as Kaggle datasets:

| Folder in output | Dataset slug | Contents |
|---|---|---|
| `run/dataset-a` | `scheduler-dataset-v1-images` | raw images + index |
| `run/dataset-b` | `scheduler-dataset-v1-labels` | tier labels, counts, gates |
| `run/cache` | `scheduler-dataset-v1-cache` | embeddings, hashes, classifier grid |

Requires the Kaggle Secret `KAGGLE_API_TOKEN` (notebook Add-ons -> Secrets). Without it the cell stops and says so.
Existing dataset slugs get a new version; missing ones are created. Nothing else is touched.
"""

LOCATE = '''import glob, json, os, shutil, subprocess

ROOT = "/kaggle/input"
TARGETS = {"dataset-a": "scheduler-dataset-v1-images", "dataset-b": "scheduler-dataset-v1-labels",
           "cache": "scheduler-dataset-v1-cache"}
found = {}
for name in TARGETS:
    hits = [p for p in glob.glob(f"{ROOT}/**/run/{name}", recursive=True)
            if os.path.isdir(p) and "/repo/" not in p]
    found[name] = hits
    print(f"{name}: {hits[:3]}")

missing = [n for n, h in found.items() if not h]
if missing:
    raise SystemExit(f"Not found under {ROOT}: {missing}. Is dataset-semschedule attached as kernel_sources?")
print("OK, all three folders found.")
'''

TOKEN = '''import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
except Exception as e:
    print("could not read secret:", type(e).__name__)
if not os.environ.get("KAGGLE_API_TOKEN"):
    raise SystemExit("KAGGLE_API_TOKEN is not set. Add it: notebook Add-ons -> Secrets -> KAGGLE_API_TOKEN, then re-run.")
print("token present (value not printed)")
'''

PUBLISH = '''import time, zipfile

OWNER = "satyapaladuguplus"
ORDER = ["dataset-b", "cache", "dataset-a"]  # small first, images last


def make_zip(src, zpath):
    n = 0
    t0 = time.time()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for root, _, files in os.walk(src):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, src))
                n += 1
                if n % 2000 == 0:
                    print(f"  zipped {n} files, {time.time() - t0:.0f}s", flush=True)
    print(f"  zip done: {n} files, {os.path.getsize(zpath) / 1e6:.0f} MB, {time.time() - t0:.0f}s", flush=True)


for name in ORDER:
    slug = TARGETS[name]
    ds_id = f"{OWNER}/{slug}"
    staging = f"/kaggle/working/staging/{slug}"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    print(f"[{name}] -> {ds_id}", flush=True)
    make_zip(found[name][0], f"{staging}/{name}.zip")
    meta = {"title": slug.replace("-", " ").title(), "id": ds_id, "licenses": [{"name": "other"}]}
    with open(f"{staging}/dataset-metadata.json", "w") as f:
        json.dump(meta, f)
    exists = subprocess.run(["kaggle", "datasets", "status", ds_id], capture_output=True).returncode == 0
    if exists:
        cmd = ["kaggle", "datasets", "version", "-p", staging, "-m", "re-published from dataset-semschedule run"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", staging]
    print(f"  uploading ({'new version' if exists else 'create'})...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-600:])
    print(res.stderr[-600:])
    print(f"  upload took {time.time() - t0:.0f}s", flush=True)
    if res.returncode != 0:
        raise SystemExit(f"publish failed for {ds_id} (exit {res.returncode})")
    shutil.rmtree(staging, ignore_errors=True)
    print(f"done: https://www.kaggle.com/datasets/{ds_id}", flush=True)
'''

DONE = '''import subprocess
print(subprocess.run(["bash", "-lc", "du -sh /kaggle/working; ls /kaggle/working"], capture_output=True, text=True).stdout)
print("All three datasets published. Check each dataset page before deleting anything.")
'''


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [cell("markdown", INTRO), cell("code", LOCATE), cell("code", TOKEN),
                    cell("code", PUBLISH), cell("code", DONE)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / "edge-publish-runs.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/edge-publish-runs", "title": "Edge publish runs",
            "code_file": "edge-publish-runs.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [], "kernel_sources": [SOURCE_KERNEL], "competition_sources": [],
            "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
