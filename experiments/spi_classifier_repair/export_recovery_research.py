"""Image-runtime parity check and unpromoted bundle, using existing local weights."""
import argparse
import gc
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch
from huggingface_hub import HfApi

from experiments.spi_classifier_repair.medic_campaign import load_encoder, export_bundle
from experiments.spi_classifier_repair.medic_deployment import SPIImagePredictor
from experiments.spi_classifier_repair.resume_store import atomic_json, file_hash


def export(run, campaign, images, snapshot):
    """Fail before inference if the existing .bin does not match pinned HF metadata."""
    run, campaign, snapshot = Path(run), Path(campaign), Path(snapshot)
    selected = json.loads((run/"selection_frozen.json").read_text())
    if selected["family"] != "balanced_logistic":
        raise ValueError("bounded exporter currently supports the selected linear recovery head only")
    old = json.loads((campaign/"comparison/clip_b32/identity.json").read_text())
    model_file = snapshot/"pytorch_model.bin"
    # Check the actually present checkpoint against the exact original revision.
    info = HfApi().model_info(old["repo"], revision=old["revision"], files_metadata=True)
    item = next(s for s in info.siblings if s.rfilename == model_file.name)
    expected = item.lfs.sha256
    actual = file_hash(model_file)
    if actual != expected:
        raise ValueError("local CLIP binary differs from pinned upstream LFS checksum")
    identity = {**old, "weights_sha256": {model_file.name: actual},
                "download_checkpoint_bytes": model_file.stat().st_size}
    # The loader must not download new weights for this bounded local check.
    # Passing pinned identity fixes repo/revision; HF cache is required below.
    from huggingface_hub import snapshot_download
    resolved = Path(snapshot_download(old["repo"], revision=old["revision"], local_files_only=True))
    if resolved.resolve() != snapshot.resolve():
        raise ValueError("unexpected local encoder snapshot")
    head = run/"balanced_logistic.joblib"
    if file_hash(head) != selected["model_sha256"]:
        raise ValueError("selected head identity changed")
    choice = {"encoder":"clip_b32", "head":"logistic", "kind":"vector", "artifact":str(head.resolve())}
    predictor = SPIImagePredictor(choice, identity, "cpu", temperature=1.)
    # Record the current processor/library identity instead of pretending to repair
    # the immutable Kaggle receipt. The parity check bridges cached/runtime inputs.
    identity = {**predictor.identity, "encoder_name":"clip_b32"}
    labels = json.loads((campaign/"labeling/labels.json").read_text())
    training = [r for r in labels if r["split"] == "train"][:10]
    with np.load(campaign/"comparison/clip_b32/features.npz", allow_pickle=False) as cache:
        lookup = {v:i for i,v in enumerate(cache["image_ids"].tolist())}
        x = cache["embeddings"]
    model = joblib.load(head)
    errors, predictions = [], []
    for row in training:
        path = Path(images)/(row["image_sha256"]+".img")
        if file_hash(path) != row["image_sha256"]:
            raise ValueError("image checksum mismatch")
        prediction = predictor.predict_image(path)
        reference = model.predict_proba(x[lookup[row["image_id"]]:lookup[row["image_id"]]+1])[0]
        error = float(np.max(np.abs(reference-np.asarray(prediction["probabilities"]))))
        errors.append(error)
        predictions.append({"image_id":row["image_id"], "prediction":prediction, "parity_error":error})
    if not errors or max(errors) > 1e-4:
        raise ValueError("cached vs image runtime parity failed; no bundle exported")
    receipt = {"schema":"medic_spi_release_v2", "runtime_promotion_allowed":False,
               "final_test_gate_passed":False, "independent_urgency_validated":False,
               "checks":{"all_class_support":False,"per_class_recall":False,"diagnostic_review":False,
                         "runtime_integration":False,"export_parity":True},
               "temperature":1., "calibration":"not fitted for this research bundle; raw logistic probabilities",
               "model_sha256":file_hash(head), "identity":identity, "max_parity_error":max(errors),
               "scope":"10 training-image cached-feature/runtime checks, not fresh urgency validation"}
    export_bundle(choice, identity, receipt, run/"bundle")
    atomic_json(run/"image_runtime_check.json", {
        "predictions":predictions, "max_parity_error":max(errors),
        "checkpoint_verification":{"source_revision":old["revision"],"hf_lfs_sha256":expected,
                                   "local_sha256":actual,"bytes":model_file.stat().st_size},
        "runtime_promotion_allowed":False,
        "note":"post-v20 local receipt, not a modification of immutable cloud provenance"})
    del predictor
    gc.collect()
    return receipt


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("run","campaign","images","snapshot"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    torch.set_num_threads(3)
    receipt=export(a.run,a.campaign,a.images,a.snapshot)
    print("RESEARCH BUNDLE EXPORTED",receipt["max_parity_error"],"promotion",receipt["runtime_promotion_allowed"],flush=True)
