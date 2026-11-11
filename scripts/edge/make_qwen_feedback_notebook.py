"""Emit Notebooks/Edge/edge-qwen-feedback.ipynb (+ kernel metadata): Qwen2.5-VL AI feedback on train images.

Thin notebook: clone main, install what is missing, run ``python -m Scheduler.training.edge.qwen_feedback``.
Account `wsl` (satyapaladuguplus), T4 x2, internet on (model download from Hugging Face). Never pushes anything.
"""
import argparse
from pathlib import Path

from scripts.edge import build_notebook as bn

STEP = "edge-qwen-feedback"
TITLE = "Edge Qwen feedback (S9 data + S3 kappa labels)"
DATASET_SOURCES = ("satyapaladuguplus/scheduler-dataset-v1-images", "satyapaladuguplus/scheduler-dataset-v1-labels")
REQUIREMENTS = ("transformers>=4.49.0", "accelerate", "bitsandbytes>=0.43.0", "PyYAML")
ENTRY = "Scheduler.training.edge.qwen_feedback"


def build(mode="pilot", branch="main", extra_args=(), account=None):
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(STEP, ENTRY, ["--mode", mode, *extra_args], TITLE, cfg["repo_url"], branch, REQUIREMENTS, True)
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, STEP, TITLE, True, DATASET_SOURCES)
    meta["id"] = "%s/%s" % (meta["id"].split("/")[0], STEP)
    meta["machine_shape"] = "NvidiaTeslaT4"
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    ap.add_argument("extra", nargs="*", help="extra entry-point args, e.g. --sec-point 1.2")
    a = ap.parse_args(argv)
    nb, meta = build(a.mode, a.branch, a.extra)
    for p in bn.write_outputs(nb, meta, STEP, Path(a.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
