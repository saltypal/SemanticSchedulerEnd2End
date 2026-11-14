"""Push the analytic-policy compare sweep (val + final test) as sharded Kaggle notebooks (2026-11-14 report run).

    python -m scripts.edge.push_sweeps_today
"""
import json
from pathlib import Path

from scripts.edge import kaggle_ops as ko
from scripts.edge import make_scheduler_sweeps_notebook as sw
from scripts.edge.build_notebook import OUT_DIR, load_kaggle_config

GROUPS = {"svdpp-tuned": ["sv_dpp"], "genie-tuned": ["genie_sv_dpp"]}   # 2026-11-14 15:00: tuned SV-DPP config
VAL_DONE = False   # val shards already pushed 2026-11-14 13:2x
GRID = ["--workers", "1", "2", "4", "8", "--rhos", "0.8", "1.2", "1.5", "--realisations", "5"]


def main():
    cfg = load_kaggle_config()
    for split, acc, final, stage in (("val", "wsl", False, "compare"), ("test", "win", True, "test")):
        if split == "val" and VAL_DONE:
            continue
        for g, pol in GROUPS.items():
            step = "edge-sweep-%s-%s" % (split, g)
            nb, meta = sw.build(stage, acc, "main", "", pol, final, step=step, extra_args=GRID)
            meta["id"] = ko.Ops(cfg, acc).slug(step)
            Path(OUT_DIR, step + ".ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
            Path(OUT_DIR, step + ".kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            out = []
            rc = ko.Ops(cfg, acc, out=out.append).push(step)
            print(step, acc, "PUSHED" if rc == 0 else "FAILED " + (out[-1][:150] if out else ""))


if __name__ == "__main__":
    main()
