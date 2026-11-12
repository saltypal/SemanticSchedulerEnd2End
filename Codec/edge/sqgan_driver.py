"""Subprocess driver: run Semantic_Lossless ``src.sqgan_train.execute`` with the project class map.

Run as a script (``python sqgan_driver.py pilot|train --sl-root ... --options ... --run ... --seg-classes ...``)
so the trainer's top-level ``src`` package (Semantic_Lossless) cannot clash with this repo's ``src``.
It monkeypatches ``sqgan_train.build`` (looked up as a module global at sqgan_train.py:110) to replace the
frozen teacher's forward with the 20-class merge; state_dict keys are unchanged. Untestable off-Kaggle (needs
torch, transformers, the upstream SQ-GAN source and GPUs).
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_seg_map():
    spec = importlib.util.spec_from_file_location("seg_map", Path(__file__).with_name("seg_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pilot", "train"))
    ap.add_argument("--sl-root", required=True)
    ap.add_argument("--options", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seg-classes", required=True)
    a = ap.parse_args(argv)
    sl = str(Path(a.sl_root).resolve())
    os.environ["SEMANTIC_LOSSLESS_ROOT"] = sl
    sys.path.insert(0, sl)
    os.chdir(sl)
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf          # same runtime shim the author notebooks use for NumPy 2
    from src import sqgan_train
    from src.assets import verified_sqgan_components
    seg = _load_seg_map()
    cfg = seg.load_class_map(a.seg_classes)
    options = json.loads(Path(a.options).read_text())
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    if not verified_sqgan_components(options["checkpoint"]):
        raise ValueError("author component checkpoints do not match the ledger digests (src/assets.py:23-39)")
    original = sqgan_train.build

    def build(opts):
        model = original(opts)
        seg.wrap_teacher(model.teacher, cfg)
        model.metadata["class_map"] = "project 20-class map from ADE20K teacher (seg_classes.yaml)"
        (run / "teacher_resolved.json").write_text(json.dumps(
            {"revision": getattr(model.teacher.config, "_commit_hash", None) or opts["teacher_revision"]}))
        return model

    sqgan_train.build = build
    if options.get("fast_loop"):
        spec = importlib.util.spec_from_file_location("sqgan_fast_train", Path(__file__).with_name("sqgan_fast_train.py"))
        fast = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fast)
        fast.execute_fast(sqgan_train, a.stage, options, run)
    else:
        sqgan_train.execute(a.stage, options, run)


if __name__ == "__main__":
    main()
