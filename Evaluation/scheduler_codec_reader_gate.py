"""Check the frozen road reader and one preregistered development threshold."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from sklearn.metrics import average_precision_score


def evaluate(rows, field, threshold):
    truth = [r["observation"]["reference_positive"] for r in rows]
    scores = [r[field] for r in rows]
    predictions = [v is not None and v >= threshold for v in scores]
    return dict(positive=sum(truth), negative=len(truth) - sum(truth),
                true_positive=sum(t and p for t, p in zip(truth, predictions)),
                false_positive=sum(not t and p for t, p in zip(truth, predictions)),
                average_precision=float(average_precision_score(
                    truth, [v if v is not None else -1. for v in scores])) if any(truth) else None)


def assess(calibration, extension):
    for field in ("codec", "sqgan_checkpoint_sha256", "reader_checkpoint_sha256", "preprocessing", "threshold"):
        if calibration[field] != extension[field]:
            raise ValueError(f"pilot configurations differ: {field}")
    if calibration.get("image_fraction", 0.25) != extension.get("image_fraction", 0.25):
        raise ValueError("pilot configurations differ: image_fraction")
    if calibration["codec"] != "sqgan":
        raise ValueError("requires native SQ-GAN measurements")
    a, b = calibration["rows"], extension["rows"]
    for field in ("source_sha256",):
        first, second = {r[field] for r in a}, {r[field] for r in b}
        if len(first) != len(a) or len(second) != len(b) or first & second:
            raise ValueError("duplicate or shared source images")
    if {r["observation"]["identity"] for r in a} & {r["observation"]["identity"] for r in b}:
        raise ValueError("calibration and extension share image identities")
    negative_scores = [r["reader_score"] for r in a
                       if not r["observation"]["reference_positive"] and r["reader_score"] is not None]
    if not negative_scores:
        raise ValueError("calibration needs at least one scored negative")
    candidate = math.nextafter(max(negative_scores), math.inf)
    frozen = {field: evaluate(b, field, calibration["threshold"])
              for field in ("source_reader_score", "resized_reader_score", "reader_score")}
    locked = evaluate(b, "reader_score", candidate)
    # This is a pilot gate, not an operational safety threshold.
    passes = (locked["positive"] >= 9 and locked["negative"] >= 9
              and locked["true_positive"] * 3 >= 2 * locked["positive"]
              and locked["false_positive"] <= 1)
    return dict(scope="Image-disjoint development extension. Event and near-duplicate separation unestablished.",
                frozen_threshold=calibration["threshold"], frozen=frozen,
                candidate_threshold=candidate, candidate_rule="Strictly above all scored calibration negatives",
                candidate=locked, candidate_gate="At least 2/3 recall and at most one false positive, "
                                                "with nine or more images per class",
                candidate_passes_pilot_gate=passes, protected_holdout_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("extension", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    first, second = args.calibration.read_bytes(), args.extension.read_bytes()
    result = assess(json.loads(first), json.loads(second))
    result["calibration_receipt_sha256"] = hashlib.sha256(first).hexdigest()
    result["extension_receipt_sha256"] = hashlib.sha256(second).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
