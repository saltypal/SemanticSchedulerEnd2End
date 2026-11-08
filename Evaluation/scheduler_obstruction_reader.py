"""Train and qualify one visible-obstruction reader on disjoint disaster events."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from Codec.integration.sqgan_payload import sha256


CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
ROLES = {"Hurricane Ian": "train", "Champlain Towers Collapse": "calibration",
         "Hurricane Harvey": "screen", "Hurricane Ida": "screen"}


def summarize(scores, truth, threshold):
    predicted = scores >= threshold
    return dict(tp=int((predicted & truth).sum()), fp=int((predicted & ~truth).sum()),
                positives=int(truth.sum()), negatives=int((~truth).sum()),
                ap=float(average_precision_score(truth, scores)))


def meets_gate(metrics):
    return metrics["tp"] * 3 >= 2 * metrics["positives"] and metrics["fp"] <= 2


def image_views(image):
    width, height = image.size
    return [image.crop((width // 4, height // 4, width * 3 // 4, height * 3 // 4)), image]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort = json.loads(args.cohort.read_bytes())
    rows = cohort["rows"]
    if ({r["event"]: r["role"] for r in rows} != ROLES
            or any(ROLES[r["event"]] != r["role"] for r in rows)
            or len({r["source_sha256"] for r in rows}) != len(rows)):
        raise ValueError("event separation or image uniqueness failed")
    if args.snapshot.name != CLIP_REVISION:
        raise ValueError("unexpected backbone revision")
    import torch
    from transformers import AutoImageProcessor, CLIPModel

    torch.set_num_threads(2)
    processor = AutoImageProcessor.from_pretrained(args.snapshot, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(args.snapshot, local_files_only=True, use_safetensors=False).eval()
    vectors = {"center": [], "whole": []}
    times = []
    with torch.inference_mode():
        for r in rows:
            path = args.cohort.parent / r["image_path"]
            if sha256(path) != r["source_sha256"]:
                raise ValueError("crop hash mismatch")
            with Image.open(path) as source:
                image = source.convert("RGB")
            if image.size != (512, 512):
                raise ValueError("unexpected crop geometry")
            start = perf_counter()
            inputs = processor(images=image_views(image), return_tensors="pt")
            embeddings = model.visual_projection(model.vision_model(**inputs).pooler_output)
            center, whole = torch.nn.functional.normalize(embeddings, dim=-1).numpy()
            times.append((perf_counter() - start) * 1000)
            vectors["center"].append(center)
            vectors["whole"].append(whole)

    train = np.array([r["role"] == "train" for r in rows])
    calibration = np.array([r["role"] == "calibration" for r in rows])
    events = np.array([r["event"] for r in rows])
    truth = np.array([r["obstruction"] for r in rows])
    results = {}
    for name, features in vectors.items():
        x = np.asarray(features)
        head = LogisticRegression(C=1, class_weight="balanced", random_state=2084).fit(x[train], truth[train])
        scores = head.predict_proba(x)[:, 1]
        threshold = float(np.nextafter(sorted(scores[calibration & ~truth], reverse=True)[2], np.inf))
        metrics = {event: summarize(scores[events == event], truth[events == event], threshold)
                   for event in ROLES}
        passed = all(meets_gate(metrics[event])
                     for event, role in ROLES.items() if role in ("calibration", "screen"))
        results[name] = dict(threshold=threshold, weights=head.coef_[0].tolist(),
                             intercept=float(head.intercept_[0]), metrics=metrics, passed=passed,
                             scores=scores.tolist())
    receipt = dict(cohort_sha256=sha256(args.cohort), reader_revision=CLIP_REVISION,
                   reader_sha256=sha256(args.snapshot / "pytorch_model.bin"),
                   processor_sha256=hashlib.sha256(json.dumps(processor.to_dict(), sort_keys=True,
                                                              default=str).encode()).hexdigest(),
                   mean_two_views_ms=float(np.mean(times)), results=results)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
