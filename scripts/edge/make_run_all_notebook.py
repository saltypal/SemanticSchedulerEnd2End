"""Emit the two dataset notebooks (+ kernel metadata) into Notebooks/Edge/.

``--which ab-embed`` (default): ``dataset-semschedule`` = datasets A, B, embeddings cache C and the grid.
``--which sqgan``: ``dataset-semschedule-sqgan`` = the SQ-GAN pass; reads A and B from the public datasets (no kernel source).
Full mode also attaches the fine-tuned v1 bundle dataset (config/edge/codec.yaml) and runs ``--codec v1``; build it for the
second account with ``--account win``.

Thin notebooks: clone, quiet install, ``python -m parameters.edge.run_all --which W --mode <MODE>``, then copy
the report, summary and logs to /kaggle/working top level. Regenerate the full run with ``--mode full``.
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
SQGAN_STEP = "dataset-semschedule-sqgan"
SQGAN_TITLE = "Dataset SemSchedule Sqgan"
WHICH = {"ab-embed": (STEP, TITLE), "sqgan": (SQGAN_STEP, SQGAN_TITLE)}
ENTRY = "parameters.edge.run_all"
WORK = "/kaggle/working/run"
_DATA = load_config("data")
DATASET_SOURCES = (_DATA["crisismmd"]["kaggle_dataset"], _DATA["normals"]["places365_kaggle_dataset"],
                   _DATA["normals"]["coco_kaggle_dataset"])             # notebook 1: no SQ-GAN datasets
SQGAN_SOURCES = tuple(s for s in _FT_SOURCES if "sqgan" in s)            # source, assets, bundle
# notebook 2 reads A and B from the PUBLIC datasets (the second account cannot read notebook 1 private output)
SQGAN_NB_SOURCES = tuple(s for s in _FT_SOURCES if "dataset-v1" in s) + SQGAN_SOURCES
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
for name in ("PILOT_REPORT.md", "pilot_report.json", "RUN_SUMMARY.md", "located.json"):
    if (W / name).exists():
        shutil.copyfile(W / name, OUT / name)
if (W / "state" / "summary.json").exists():
    shutil.copyfile(W / "state" / "summary.json", OUT / "summary.json")
for sub in ("logs", "checks"):
    if (W / sub).is_dir():
        shutil.copytree(W / sub, OUT / sub, dirs_exist_ok=True)
print((OUT / "RUN_SUMMARY.md").read_text() if (OUT / "RUN_SUMMARY.md").exists() else "NO RUN_SUMMARY.md PRODUCED")
'''


def build(mode=None, branch="main", account=None, which="ab-embed"):
    """Return (notebook dict, kernel-metadata dict) for ``which`` in WHICH."""
    mode = mode or MODE
    step, title = WHICH[which]
    cfg = bn.load_kaggle_config()
    args = ["--which", which, "--mode", mode, "--work", WORK]
    nb = bn.build_notebook(step, ENTRY, args, title, cfg["repo_url"], branch, REQUIREMENTS, True)
    what = ("Datasets A (images), B (labels) and C embeddings cache + grid" if which == "ab-embed" else
            "SQ-GAN pass over dataset A and B read from the public datasets scheduler-dataset-v1-images / -labels")
    nb["cells"][0] = bn._md("# %s\n\n%s, mode `%s`. Logic lives in `%s`.\n" % (title, what, mode, ENTRY))
    nb["cells"][4] = bn._code(RUN.format(argv=["python", "-u", "-m", ENTRY, *args]))
    nb["cells"] += [bn._md("## Collect outputs\n\nReport, summary and logs are copied to the top level of /kaggle/working.\n"),
                    bn._code(COLLECT.format(work=WORK))]
    account = account or cfg.get("default_account", "wsl")
    sources = DATASET_SOURCES if which == "ab-embed" else SQGAN_NB_SOURCES
    if which == "sqgan" and mode == "full":            # fine-tuned v1 bundle (config/edge/codec.yaml), read by --codec v1
        v1 = load_config("codec")["codecs"]["v1"]["kaggle_dataset"]
        sources = sources + ((v1,) if v1 and v1 not in sources else ())
    meta = bn.build_metadata(cfg, account, step, title, True, sources)
    meta["machine_shape"] = "NvidiaTeslaT4"        # T4 x2, same key as the fine-tune notebook
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=tuple(WHICH), default="ab-embed")
    ap.add_argument("--mode", choices=("pilot", "full"), default=MODE)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--account", default=None, help="key of config/edge/kaggle.yaml accounts (default: default_account)")
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.mode, a.branch, account=a.account, which=a.which)
    for p in bn.write_outputs(nb, meta, WHICH[a.which][0], Path(a.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
