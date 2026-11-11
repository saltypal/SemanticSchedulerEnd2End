"""Emit Notebooks/Edge/dataset-semschedule.ipynb (+ kernel metadata): datasets A, B and C in ONE notebook.

Thin notebook: clone, quiet install, ``python -m parameters.edge.run_all --mode <MODE>``, then copy the
report, summary and logs to /kaggle/working top level. Regenerate for the full run with ``--mode full``.
Never pushes anything.
"""
import argparse
from pathlib import Path

from Evaluation.edge.edge_config import load_config
from scripts.edge import build_notebook as bn
from scripts.edge.make_sqgan_finetune_notebook import DATASET_SOURCES as _FT_SOURCES

MODE = "pilot"                      # default mode; override with --mode full
STEP = "dataset-semschedule"
TITLE = "Dataset SemSchedule"
ENTRY = "parameters.edge.run_all"
WORK = "/kaggle/working/run"
_DATA = load_config("data")
DATASET_SOURCES = (_DATA["crisismmd"]["kaggle_dataset"], _DATA["normals"]["places365_kaggle_dataset"],
                   _DATA["normals"]["coco_kaggle_dataset"],
                   *[s for s in _FT_SOURCES if "sqgan" in s])   # bundle, source, assets
REQUIREMENTS = ("pytorch-lightning==1.9.5", "torchmetrics==0.11.4", "transformers==4.46.3", "scipy", "omegaconf",
                "pytorch-msssim==1.0.0", "lpips==0.1.4", "PyYAML", "einops", "pillow")

RUN = '''
import os, subprocess
ARGV = {argv!r}
_p = subprocess.Popen(ARGV, cwd=REPO, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for _line in _p.stdout:
    print(_line, end="", flush=True)
RC = _p.wait()
print("run_all exit code:", RC)
'''
COLLECT = '''
import shutil
from pathlib import Path
W, OUT = Path("{work}"), Path("/kaggle/working")
for name in ("PILOT_REPORT.md", "pilot_report.json", "RUN_SUMMARY.md"):
    if (W / name).exists():
        shutil.copyfile(W / name, OUT / name)
if (W / "state" / "summary.json").exists():
    shutil.copyfile(W / "state" / "summary.json", OUT / "summary.json")
for sub in ("logs", "checks"):
    if (W / sub).is_dir():
        shutil.copytree(W / sub, OUT / sub, dirs_exist_ok=True)
print((OUT / "RUN_SUMMARY.md").read_text() if (OUT / "RUN_SUMMARY.md").exists() else "NO RUN_SUMMARY.md PRODUCED")
'''


def build(mode=None, branch="main", account=None):
    """Return (notebook dict, kernel-metadata dict)."""
    mode = mode or MODE
    cfg = bn.load_kaggle_config()
    args = ["--mode", mode, "--work", WORK]
    nb = bn.build_notebook(STEP, ENTRY, args, TITLE, cfg["repo_url"], branch, REQUIREMENTS, True)
    nb["cells"][0] = bn._md("# %s\n\nDatasets A (images), B (labels) and C (cache + grid) in one run, mode `%s`. "
                            "Logic lives in `%s`.\n" % (TITLE, mode, ENTRY))
    nb["cells"][4] = bn._code(RUN.format(argv=["python", "-u", "-m", ENTRY, *args]))
    nb["cells"] += [bn._md("## Collect outputs\n\nReport, summary and logs are copied to the top level of /kaggle/working.\n"),
                    bn._code(COLLECT.format(work=WORK))]
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, STEP, TITLE, True, DATASET_SOURCES)
    meta["machine_shape"] = "NvidiaTeslaT4"        # T4 x2, same key as the fine-tune notebook
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("pilot", "full"), default=MODE)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.mode, a.branch)
    for p in bn.write_outputs(nb, meta, STEP, Path(a.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
