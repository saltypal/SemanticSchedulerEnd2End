"""Full-val and full-test compare sweeps for the frozen learners on Kaggle (checkpoints from their training kernels),
plus the STS-fusion ablation (SV-DPP, SPI-first + batch rule). 2026-11-14 report run.

    python -m scripts.edge.push_learned_sweeps
"""
import json
from pathlib import Path

from scripts.edge import kaggle_ops as ko
from scripts.edge import make_scheduler_sweeps_notebook as sw
from scripts.edge.build_notebook import OUT_DIR, load_kaggle_config

U = "satyapaladuguplus/"
GROUPS = {   # name: (policies, kernel sources, sts_fusion)
    "scopeppo": (["learned:scope_ppo:scope_ppo_seed0.pt", "learned:ppo:ppo_seed0.pt"],
                 [U + "edge-train-scope-ppo", U + "edge-train-ppo"], False),
    "lambdamart": (["learned:ltr:ltr_seed0.pt", "learned:bc:bc_seed0.pt"],
                   [U + "edge-train-ltr-lambdamart", U + "edge-train-bc"], False),
    "ranknet": (["learned:ltr:ltr_seed0.pt", "learned:scope_ltr:scope_ltr_seed1.pt"],
                [U + "edge-train-ltr-ranknet", U + "edge-train-scope-ltr"], False),
    "fusion": (["sv_dpp", "spi_edf_ratio"], [], True),
}
GRID = ["--workers", "1", "2", "4", "8", "--rhos", "0.8", "1.2", "1.5", "--realisations", "5"]


def main():
    cfg = load_kaggle_config()
    for split, stage, final in (("val", "compare", False), ("test", "test", True)):
        for g, (pol, kern, fus) in GROUPS.items():
            step = "edge-sweep-%s-%s" % (split, g)
            nb, meta = sw.build(stage, "wsl", "main", "", pol, final, step=step, sts_fusion=fus,
                                kernel_sources=kern, extra_args=GRID)
            meta["id"] = ko.Ops(cfg, "wsl").slug(step)
            Path(OUT_DIR, step + ".ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
            Path(OUT_DIR, step + ".kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rc = ko.Ops(cfg, "wsl", out=lambda *_: None).push(step)
            print(step, "PUSHED" if rc == 0 else "FULL/FAILED")


if __name__ == "__main__":
    main()
