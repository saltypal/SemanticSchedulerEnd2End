"""Blinded training-label audit. AI reviews are diagnostics, never ground truth."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.spi_classifier_repair.resume_store import atomic_json, file_hash

LEVELS = ("Low", "Medium", "High", "Critical")


def select_panel(labels, per_class=12, seed=29):
    """Select only training rows; hide teacher targets in the review manifest."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    ids = [r["image_id"] for r in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate image identity")
    rng = np.random.default_rng(seed)
    selected = []
    for level in LEVELS:
        pool = sorted((r for r in labels if r.get("split") == "train"
                       and r.get("available") is True and r.get("level") == level),
                      key=lambda r: r["image_id"])
        if len(pool) < per_class:
            raise ValueError(f"insufficient training support for {level}: {len(pool)}")
        selected.extend(pool[i] for i in rng.choice(len(pool), per_class, replace=False))
    return [selected[i] for i in rng.permutation(len(selected))]


def prepare(labels_path, images, output, per_class=12, seed=29):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    labels = json.loads(Path(labels_path).read_text(encoding="utf8"))
    chosen = select_panel(labels, per_class, seed)
    manifest, key = [], []
    for i, row in enumerate(chosen):
        case_id = f"A{i+1:02d}"
        path = Path(images) / (row["image_sha256"] + ".img")
        if file_hash(path) != row["image_sha256"]:
            raise ValueError("audit image identity mismatch")
        manifest.append({"case_id": case_id, "image_path": str(path.resolve()),
                         "image_sha256": row["image_sha256"]})
        key.append({"case_id": case_id, "image_id": row["image_id"],
                    "teacher_level": row["level"], "group": row["group"],
                    "source_role": row["split"], "native_labels": row.get("native_labels", {})})
    atomic_json(output / "blind_manifest.json", manifest)
    atomic_json(output / "withheld_key.json", key)
    atomic_json(output / "provenance.json", {
        "labels_sha256": file_hash(labels_path), "seed": seed, "per_class": per_class,
        "source_role": "train", "image_count": len(manifest), "script_sha256": file_hash(__file__),
        "scope": "class-balanced qualitative AI diagnostic; not representative accuracy or verified urgency"})
    for start in range(0, len(manifest), 12):
        canvas = Image.new("RGB", (1440, 1408), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for j, record in enumerate(manifest[start:start+12]):
            left, top = (j % 3)*480, (j//3)*352
            with Image.open(record["image_path"]) as image:
                thumbnail = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (472, 316))
                canvas.paste(thumbnail, (left+(480-thumbnail.width)//2, top+32+(316-thumbnail.height)//2))
            draw.text((left+8, top+8), record["case_id"], fill="black")
        canvas.save(output / f"blind_panel_{start//12+1:02d}.png")
    return {"images": len(manifest), "output": str(output)}


def analyze(panel, review_path):
    """Join only after review is saved. A disagreement is not a corrected label."""
    panel = Path(panel)
    keys = {r["case_id"]: r for r in json.loads((panel/"withheld_key.json").read_text())}
    review = json.loads(Path(review_path).read_text(encoding="utf8"))
    if review.get("review_type") != "blinded_ai_diagnostic":
        raise ValueError("review must explicitly identify AI diagnostic scope")
    cases = review["cases"]
    ids = [r["case_id"] for r in cases]
    if len(ids) != len(set(ids)) or set(ids) != set(keys):
        raise ValueError("review must cover each panel case exactly once")
    results = []
    for row in cases:
        levels = row["plausible_levels"]
        if not levels or not set(levels) <= set(LEVELS) or len(levels) != len(set(levels)):
            raise ValueError("invalid plausible levels")
        if not str(row.get("visible_evidence", "")).strip():
            raise ValueError("visible evidence required")
        if row.get("immediate_harm_evidence") not in ("visible", "not_established", "indeterminate"):
            raise ValueError("immediate harm evidence must be explicit")
        key = keys[row["case_id"]]
        results.append({**row, "teacher_level": key["teacher_level"], "image_id": key["image_id"],
                        "teacher_within_review_range": key["teacher_level"] in levels})
    by_class = {}
    for level in LEVELS:
        rows = [r for r in results if r["teacher_level"] == level]
        by_class[level] = {"cases": len(rows),
                           "outside_reviewer_plausible_range": sum(not r["teacher_within_review_range"] for r in rows),
                           "immediate_harm_not_established": sum(r["immediate_harm_evidence"] != "visible" for r in rows)}
    result = {"review_type": review["review_type"], "review_sha256": file_hash(review_path),
              "source_role": "train", "per_teacher_class": by_class, "cases": results,
              "independent_urgency_validated": False, "automatic_relabeling": False,
              "limitation": "AI judgment on a balanced panel; disagreement prompts inspection, not truth or reliability clearance"}
    atomic_json(panel/"audit_results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--labels", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-class", type=int, default=12)
    p.add_argument("--seed", type=int, default=29)
    p = sub.add_parser("analyze")
    p.add_argument("--panel", required=True)
    p.add_argument("--review", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.labels, args.images, args.output, args.per_class, args.seed))
    else:
        print(json.dumps(analyze(args.panel, args.review)["per_teacher_class"], indent=2))
