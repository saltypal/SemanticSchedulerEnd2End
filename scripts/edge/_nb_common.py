"""Shared pieces of the phase-2 notebooks (train, sweeps, live): dataset sources, input discovery, writing."""
from pathlib import Path

from scripts.edge import build_notebook as bn

SIMTABLE_DS = "satyapaladuguplus/scheduler-dataset-v1-simtable"   # PRIVATE now: a second-account run needs it public
LABELS_DS = "satyapaladuguplus/scheduler-dataset-v1-labels"
IMAGES_DS = "satyapaladuguplus/scheduler-dataset-v1-images"
SQGAN_V1_DS = "satyapaladugu/scheduler-sqgan-v1"                   # codec v1 bundle (later)
PRIVATE_NOTE = (f"**Inputs.** `{SIMTABLE_DS}` (simtable_v1.jsonl, latency_v1.json) is currently PRIVATE. "
                "A run on the second account (`--account win`) needs that dataset made public first.\n")

DISCOVER = '''
import os
def find_input(name, root="/kaggle/input"):
    """First file called ``name`` anywhere under ``root`` (mount depth differs between kernels)."""
    for dp, _dn, fn in os.walk(root):
        if name in fn:
            return os.path.join(dp, name)
    raise FileNotFoundError(name + " not found under " + root + "; attach the dataset (public for another account)")
TABLE = find_input("simtable_v1.jsonl")
PROFILE = find_input("latency_v1.json")
print("table  :", TABLE)
print("profile:", PROFILE)
'''


def discover_cell():
    return bn._code(DISCOVER)


def finish(nb, meta):
    return nb, meta


def write(nb, meta, step, out_dir):
    for p in bn.write_outputs(nb, meta, step, Path(out_dir)):
        print(p)
