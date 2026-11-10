"""Emit Notebooks/Edge/edge-sqgan-finetune-v1.ipynb (+ kernel metadata) with scripts.edge.build_notebook.

Thin notebook: clone this repo's main, install the trainer's pinned dependencies (Semantic_Lossless
notebook_support.py:13,22), print the source banner (pinned GitHub clone or Kaggle fallback), then run
``python -m Codec.edge.sqgan_finetune``. Data roots are auto-discovered under /kaggle/input. Never pushes anything.
"""
import argparse
from pathlib import Path

from scripts.edge import build_notebook as bn

STEP = "edge-sqgan-finetune-v1"
CONFIG = "config/edge/sqgan_finetune.yaml"
TITLE = "Edge SQ-GAN fine-tune v1 (teammate hand-off)"
DATASET_SOURCES = (
    "satyapaladuguplus/scheduler-dataset-v1-images",
    "satyapaladuguplus/scheduler-dataset-v1-labels",
    "satyapaladuguplus/semantic-lossless-private-sqgan-source",
    "satyapaladuguplus/semantic-lossless-private-sqgan-assets",
    "satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102",
)
REQUIREMENTS = ("pytorch-lightning==1.9.5", "torchmetrics==0.11.4", "transformers==4.46.3", "scipy", "omegaconf",
                "pytorch-msssim==1.0.0", "lpips==0.1.4", "PyYAML", "einops")
ENTRY = "Codec.edge.sqgan_finetune"


def build(branch="main", account=None):
    """Return (notebook dict, kernel-metadata dict). Config is read from the cloned repo, edit only the YAML."""
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(STEP, ENTRY, ["--config", CONFIG], TITLE, cfg["repo_url"], branch, REQUIREMENTS, True)
    banner = [bn._md("## Trainer source\n\nPrints whether Semantic_Lossless came from the pinned GitHub commit or the Kaggle fallback.\n"),
              bn._run_cell(ENTRY, ["--config", CONFIG, "--stage", "source"])]
    nb["cells"][3:3] = banner                      # after clone + install, before the run
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, STEP, TITLE, True, DATASET_SOURCES)
    meta["id"] = "%s/%s" % (meta["id"].split("/")[0], STEP)   # slug is already edge-prefixed
    meta["machine_shape"] = "NvidiaTeslaT4"        # T4 x2: the trainer supports DataParallel (sqgan_train.py:129-131)
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.branch)
    for p in bn.write_outputs(nb, meta, STEP, Path(a.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
