"""Qualify a frozen reader on fresh source images before any codec replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score

from Codec.integration.sqgan_payload import sha256


READER_REVISION = "e7971f2700074d2cfe587c685a73de400b4f809f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reader", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration-screen", type=Path)
    args = parser.parse_args()
    if args.reader.name != READER_REVISION:
        raise ValueError("reader must be the frozen large-reference revision")
    raw = args.manifest.read_bytes()
    inputs = json.loads(raw)
    positive_count = sum("roads_damage" in row["positive_labels"] for row in inputs)
    if (positive_count not in ((9, 12) if args.calibration_screen else (12,))
            or len(inputs) - positive_count != 12
            or any(row["source_split"] != "validation" for row in inputs)):
        raise ValueError("development validation inputs must meet the positive and negative quotas")
    if (len({row["item_id"] for row in inputs}) != len(inputs)
            or len({row["source_sha256"] for row in inputs}) != len(inputs)):
        raise ValueError("duplicate source input")
    if args.output.exists():
        raise ValueError("output exists")
    reader_hash = sha256(args.reader / "model.safetensors")
    calibration_hash = None
    threshold = 0.5
    if args.calibration_screen:
        calibration_bytes = args.calibration_screen.read_bytes()
        calibration = json.loads(calibration_bytes)
        if calibration["reader_sha256"] != reader_hash or calibration["threshold"] != 0.5:
            raise ValueError("calibration screen has the wrong frozen reader or initial threshold")
        if {row["source_sha256"] for row in calibration["rows"]} & {row["source_sha256"] for row in inputs}:
            raise ValueError("calibration and evaluation images overlap")
        negative_scores = sorted((row["source_score"] for row in calibration["rows"] if not row["positive"]),
                                 reverse=True)
        threshold = float(np.nextafter(negative_scores[2], np.inf))
        calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.set_num_threads(2)
    processor = AutoImageProcessor.from_pretrained(args.reader, local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(args.reader, local_files_only=True).eval()
    label = next(i for i, name in model.config.id2label.items() if name == "roads_damage")

    @torch.inference_mode()
    def score(image):
        return float(model(**processor(images=image, return_tensors="pt")).logits[0, label].sigmoid())

    rows = []
    for row in inputs:
        path = args.manifest.parent / row["image_path"]
        if sha256(path) != row["source_sha256"]:
            raise ValueError("source hash mismatch")
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        resized = ImageOps.contain(source, (512, 256), Image.Resampling.BILINEAR)
        rows.append(dict(id=row["item_id"], source_sha256=row["source_sha256"],
                         positive="roads_damage" in row["positive_labels"],
                         source_score=score(source), resized_score=score(resized)))
    truth = np.array([row["positive"] for row in rows])
    minimum_true_positives = (2 * positive_count + 2) // 3
    metrics = {}
    for key in ("source_score", "resized_score"):
        scores = np.array([row[key] for row in rows])
        predicted = scores >= threshold
        metrics[key] = dict(tp=int((predicted & truth).sum()), fp=int((predicted & ~truth).sum()),
                            ap=float(average_precision_score(truth, scores)))
    base = metrics["source_score"]
    result = dict(scope="Fresh development validation cohort. No final test labels.",
                  manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  reader_revision=READER_REVISION, reader_sha256=reader_hash,
                  calibration_screen_sha256=calibration_hash,
                  threshold=threshold, minimum_true_positives=minimum_true_positives, maximum_false_positives=2,
                  source_gate_passed=base["tp"] >= minimum_true_positives and base["fp"] <= 2,
                  metrics=metrics, rows=rows)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
