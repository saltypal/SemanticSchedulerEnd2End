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
MULTIEVENT_ROLES = {**ROLES, "Hurricane Harvey": "train", "Hurricane Ida": "train",
                    "Mayfield Tornado": "screen", "Hurricane Michael": "screen"}


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
    parser.add_argument("--reference-receipt", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort = json.loads(args.cohort.read_bytes())
    rows = cohort["rows"]
    multievent = cohort.get("profile") == "multievent"
    roles = MULTIEVENT_ROLES if multievent else ROLES
    if ({r["event"]: r["role"] for r in rows} != roles
            or any(roles[r["event"]] != r["role"] for r in rows)
            or len({r["source_sha256"] for r in rows}) != len(rows)):
        raise ValueError("event separation or image uniqueness failed")
    if multievent:
        review = args.cohort.parent / "visual_review.json"
        if not review.is_file() or sha256(review) != cohort.get("visual_review_sha256"):
            raise ValueError("multi-event scoring requires the locked visual review")
        selected = {r["sha256"] for r in json.loads(review.read_bytes())["rows"] if r["selected"]}
        negatives = {r["source_sha256"] for r in rows if r["role"] == "screen" and not r["obstruction"]}
        if selected != negatives:
            raise ValueError("evaluation negatives differ from the visual review")
        for event, role in roles.items():
            if role == "screen":
                labels = [r["obstruction"] for r in rows if r["event"] == event]
                if len(labels) != 18 or sum(labels) != 6:
                    raise ValueError(f"evaluation quotas failed for {event}")
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
                   for event in roles}
        passed = all(meets_gate(metrics[event])
                     for event, role in roles.items() if role in ("calibration", "screen"))
        results[name] = dict(threshold=threshold, weights=head.coef_[0].tolist(),
                             intercept=float(head.intercept_[0]), metrics=metrics, passed=passed,
                             scores=scores.tolist())
    reader_sha = sha256(args.snapshot / "pytorch_model.bin")
    processor_sha = hashlib.sha256(json.dumps(processor.to_dict(), sort_keys=True,
                                               default=str).encode()).hexdigest()
    receipt = dict(cohort_sha256=sha256(args.cohort), reader_revision=CLIP_REVISION,
                   reader_sha256=reader_sha, processor_sha256=processor_sha,
                   mean_two_views_ms=float(np.mean(times)), results=results)
    if args.reference_receipt:
        reference = json.loads(args.reference_receipt.read_bytes())
        if reference["reader_sha256"] != reader_sha or reference["processor_sha256"] != processor_sha:
            raise ValueError("reference reader has different input features")
        head = reference["results"]["center"]
        logits = np.asarray(vectors["center"]) @ np.asarray(head["weights"]) + head["intercept"]
        scores = 1 / (1 + np.exp(-logits))
        receipt["ian_only_reference"] = dict(
            receipt_sha256=sha256(args.reference_receipt), threshold=head["threshold"],
            metrics={event: summarize(scores[events == event], truth[events == event], head["threshold"])
                     for event, role in roles.items() if role != "train"},
            scores=scores.tolist())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
