"""Emit Notebooks/Edge/edge-pilot.ipynb (+ kernel metadata) with scripts.edge.build_notebook.

Thin notebook: clone main, install, run ``python -m parameters.edge.pilot`` (every pipeline stage on a tiny
sample plus one stage per pilot check), then copy PILOT_REPORT.md / pilot_report.json / checks/ to
/kaggle/working so they are the notebook output, and print the report. Never pushes anything.
"""
import argparse
from pathlib import Path

from scripts.edge import build_notebook as bn

STEP = "edge-pilot"
TITLE = "Edge pilot (all stages + unverified-item checks)"
ENTRY = "parameters.edge.pilot"
ARGS = ["--work", "/kaggle/working/pilot", "--per-source", "20"]
DATASET_SOURCES = (
    "satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102",
    "satyapaladuguplus/semantic-lossless-private-sqgan-source",
    "satyapaladuguplus/semantic-lossless-private-sqgan-assets",
)
REQUIREMENTS = ("pytorch-lightning==1.9.5", "torchmetrics==0.11.4", "transformers==4.46.3", "scipy", "omegaconf",
                "pytorch-msssim==1.0.0", "lpips==0.1.4", "PyYAML", "einops")
COLLECT = '''
import json, shutil
from pathlib import Path
W, OUT = Path("/kaggle/working/pilot"), Path("/kaggle/working")
for name in ("PILOT_REPORT.md", "pilot_report.json"):
    if (W / name).exists():
        shutil.copyfile(W / name, OUT / name)
if (W / "checks").is_dir():
    shutil.copytree(W / "checks", OUT / "checks", dirs_exist_ok=True)
print((OUT / "PILOT_REPORT.md").read_text() if (OUT / "PILOT_REPORT.md").exists() else "NO PILOT_REPORT.md PRODUCED")
'''


def build(branch="main", account=None):
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(STEP, ENTRY, ARGS, TITLE, cfg["repo_url"], branch, REQUIREMENTS, True)
    # the report must exist as output even if the pilot's exit code is non-zero: collect cell sits before the manifest
    nb["cells"].insert(5, bn._md("## Collect report\n\nCopies the report and per-check JSON to /kaggle/working and prints it.\n"))
    nb["cells"].insert(6, bn._code(COLLECT))
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, STEP, TITLE, True, DATASET_SOURCES)
    meta["id"] = "%s/%s" % (meta["id"].split("/")[0], STEP)
    meta["machine_shape"] = "NvidiaTeslaT4"        # T4 x2 (same key the fine-tune notebook uses); builder has no switch
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
