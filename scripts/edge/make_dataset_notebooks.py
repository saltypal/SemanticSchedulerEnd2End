"""Emit Notebooks/Edge/edge-a-download-images.ipynb and edge-b-labeling.ipynb (+ kernel metadata).

Both are CPU, internet-on, thin notebooks built with scripts.edge.build_notebook; each publishes its
output folder as a Kaggle dataset from the entrypoint. Never pushes anything.
"""
import argparse
from pathlib import Path

from scripts.edge import build_notebook as bn

META_DIR = "/kaggle/working/repo/experiments/full_medic_labeling/runs/crisismmd4000_selection"
SPECS = {
    "edge-a-download-images": dict(
        entry="parameters.edge.dataset_a", title="Edge dataset A: download images",
        args=["--meta-dir", META_DIR, "--out", "/kaggle/working/dataset-a", "--workers", "16"], sources=()),
    "edge-b-labeling": dict(
        entry="parameters.edge.build_label_dataset", title="Edge dataset B: labeling",
        args=["--meta-dir", META_DIR, "--out", "/kaggle/working/dataset-b"],
        sources=("satyapaladuguplus/scheduler-dataset-v1-images",)),
}


def build(name, branch="main", account=None):
    """Return (notebook dict, kernel-metadata dict) for one of SPECS."""
    spec = SPECS[name]
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(name, spec["entry"], spec["args"], spec["title"], cfg["repo_url"], branch,
                           ("pillow",), False)
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, name, spec["title"], False, spec["sources"])
    meta["id"] = "%s/%s" % (meta["id"].split("/")[0], name)
    meta["code_file"] = name + ".ipynb"
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    for name in SPECS:
        nb, meta = build(name, a.branch)
        for p in bn.write_outputs(nb, meta, name, Path(a.out_dir)):
            print(p)


if __name__ == "__main__":
    main()
