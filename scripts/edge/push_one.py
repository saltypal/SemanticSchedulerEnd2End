"""Build and push one phase-2 notebook with a budget, seed count and optional GPU session (uses a GPU slot as
extra parallel capacity; the code runs on its CPUs). Prints 'PUSHED <step>' or 'FULL <step>'.

    python -m scripts.edge.push_one edge-train-scope-ppo --budget fix --seeds 2 [--gpu]
"""
import argparse
import json
from pathlib import Path

from scripts.edge import kaggle_ops as ko
from scripts.edge import launch_phase2 as lp
from scripts.edge.build_notebook import OUT_DIR, load_kaggle_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step")
    ap.add_argument("--budget", default="default", choices=sorted(lp.BUDGETS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_kaggle_config()
    e = next(x for x in lp.plan() if x.step == a.step)
    lp.build_entry(e, OUT_DIR, cfg, "main", a.seeds, a.budget)
    if a.gpu:
        mp = Path(OUT_DIR) / f"{e.step}.kernel-metadata.json"
        m = json.loads(mp.read_text())
        m["enable_gpu"] = True
        mp.write_text(json.dumps(m, indent=2))
        nbp = Path(OUT_DIR) / f"{e.step}.ipynb"
        nb = json.loads(nbp.read_text())
        nb["metadata"].setdefault("kaggle", {})["accelerator"] = "nvidiaTeslaT4"
        nbp.write_text(json.dumps(nb, indent=1))
    out = []
    ops = ko.Ops(cfg, e.account, out=out.append)
    rc = ops.push(e.step)
    print(("PUSHED " if rc == 0 else "FULL ") + e.step)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
