"""Emit Notebooks/Edge/edge-s2-feature-pass.ipynb (+ kernel metadata) with scripts.edge.build_notebook.

The notebook is thin: clone, install, then in order
  feature_pass run -> compute_profile -> feature_pass export-grid -> embedding_grid grid -> embedding_grid select
all under /kaggle/working/runs/edge-s2/ (grid outputs under .../grid/). Never pushes anything.
"""
import argparse
from pathlib import Path

from scripts.edge import build_notebook as bn

STEP = "edge-s2"
NAME = "edge-s2-feature-pass"
OUT = "/kaggle/working/runs/edge-s2"
DATASET_SOURCES = (
    "satyapaladuguplus/scheduler-dataset-v1-images",
    "satyapaladuguplus/scheduler-dataset-v1-labels",
    "satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102",
    "satyapaladuguplus/semantic-lossless-private-sqgan-source",
    "satyapaladuguplus/semantic-lossless-private-sqgan-assets",
)
TITLE = "Edge S2 feature pass (embeddings, hashes, SQ-GAN, decode, timings) + SPI grid"

_NOTE = ("Provisional splits are built by `parameters.edge.splits.assign_event_splits` when "
         "`splits_v1.json` is absent from the labels dataset; the export step prints which was used.")


def build(branch="main", account=None):
    """Return (notebook dict, kernel-metadata dict)."""
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(STEP, "Codec.edge.feature_pass", ["run", "--out", OUT],
                           TITLE, cfg["repo_url"], branch, (), True)
    extra = [
        bn._md("## Compute profile (once per model)\n"),
        bn._run_cell("Codec.edge.compute_profile", ["--out", OUT]),
        bn._md("## Grid hand-off\n\n" + _NOTE + "\n"),
        bn._run_cell("Codec.edge.feature_pass", ["export-grid", "--out", OUT]),
        bn._run_cell("parameters.edge.embedding_grid",
                     ["grid", "--embeddings-dir", OUT + "/grid", "--labels", OUT + "/grid/labels_by_uid.jsonl",
                      "--splits", OUT + "/grid/splits_by_uid.json", "--out", OUT + "/grid"]),
        bn._run_cell("parameters.edge.embedding_grid", ["select", "--grid-dir", OUT + "/grid"]),
    ]
    nb["cells"][-1:-1] = extra                      # keep the manifest cell last
    nb["cells"] += [bn._md("## Publish cache (dataset scheduler-dataset-v1-cache)" + chr(10) + chr(10) + "Skipped with a reason if no credentials." + chr(10)),
                    bn._run_cell("Codec.edge.feature_pass", ["publish", "--out", OUT, "--slug", "scheduler-dataset-v1-cache"])]
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, NAME, TITLE, True, DATASET_SOURCES)
    meta["id"] = "%s/%s" % (meta["id"].split("/")[0], NAME)
    meta["code_file"] = NAME + ".ipynb"
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.branch)
    for p in bn.write_outputs(nb, meta, NAME, Path(a.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
