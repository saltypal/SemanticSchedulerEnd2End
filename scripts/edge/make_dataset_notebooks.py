"""Emit Notebooks/Edge/dataset-semschedule-a.ipynb and edge-b-labeling.ipynb (+ kernel metadata).

Both are CPU, internet-on, thin notebooks built with scripts.edge.build_notebook; each publishes its
output folder as a Kaggle dataset from the entrypoint. Never pushes anything.
"""
import argparse
from pathlib import Path

from Evaluation.edge.edge_config import load_config
from scripts.edge import build_notebook as bn

_DATA = load_config("data")
CRISIS_DS = _DATA["crisismmd"]["kaggle_dataset"]   # attached to notebook A, with Places365 and COCO
PLACES_DS = _DATA["normals"]["places365_kaggle_dataset"]
COCO_DS = _DATA["normals"]["coco_kaggle_dataset"]
META_DIR = "/kaggle/working/repo/experiments/full_medic_labeling/runs/crisismmd4000_selection"
SPECS = {
    "dataset-semschedule-a": dict(
        title="Dataset SemSchedule A",
        entry="parameters.edge.dataset_a", about="Edge dataset A: download images",
        args=["--meta-dir", META_DIR, "--out", "/kaggle/working/dataset-a", "--workers", "16"], kernel_sources=(), dataset_sources=(CRISIS_DS, PLACES_DS, COCO_DS)),
    "edge-b-labeling": dict(
        entry="parameters.edge.build_label_dataset", about="Edge dataset B: labeling",
        args=["--meta-dir", META_DIR, "--out", "/kaggle/working/dataset-b"],
        kernel_sources=("satyapaladuguplus/dataset-semschedule-a",), dataset_sources=()),
}


def build(name, branch="main", account=None):
    """Return (notebook dict, kernel-metadata dict) for one of SPECS."""
    spec = SPECS[name]
    cfg = bn.load_kaggle_config()
    title = spec.get("title") or bn.title_for_slug(name)
    nb = bn.build_notebook(name, spec["entry"], spec["args"], title, cfg["repo_url"], branch,
                           ("pillow",), False)
    nb["cells"][0] = bn._md("# %s\n\n%s. Thin wrapper; logic lives in `%s`.\n" % (title, spec["about"], spec["entry"]))
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, name, title, False, spec["dataset_sources"], spec["kernel_sources"])
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
