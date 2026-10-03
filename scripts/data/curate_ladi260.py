"""Create the fixed 260-image, 12-stratum LADI dynamic benchmark."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="train")
    args = parser.parse_args()

    from datasets import Image, load_dataset
    from huggingface_hub import HfApi
    from PIL import Image as PILImage
    from INFRA.data.ladi_manifest import (
        DATASET_ID, LADI260_COUNTS, LADI260_RUBRIC, LADI260_STRATA,
        MANIFEST_FIELDS, NATIVE_LABELS, validate_ladi260_manifest,
    )

    revision = HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(DATASET_ID, split=args.source_split,
                           revision=revision, streaming=True)
    dataset = dataset.cast_column("image", Image(decode=False))
    output = args.output_dir
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    pools = {name: [] for name in LADI260_STRATA}
    seen = set()
    for source_index, row in enumerate(dataset):
        labels = {name: bool(row.get(name, False)) for name in NATIVE_LABELS}
        eligible = [name for name in LADI260_STRATA
                    if labels[name] and len(pools[name]) < LADI260_COUNTS[name]]
        if not eligible:
            continue
        # Assign multi-label images to the least-filled eligible stratum.
        primary = min(eligible, key=lambda name: (
            len(pools[name]) / LADI260_COUNTS[name], LADI260_STRATA.index(name)))
        raw = row["image"]["bytes"]
        with PILImage.open(BytesIO(raw)) as opened:
            buffer = BytesIO(); opened.convert("RGB").save(buffer, format="PNG")
        payload = buffer.getvalue(); digest = sha256(payload).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        image_id = f"ladi-{digest[:16]}"
        path = images / f"{image_id}.png"; path.write_bytes(payload)
        pools[primary].append({
            "image_id": image_id, "source_uri": f"hf://datasets/{DATASET_ID}@{revision}/{args.source_split}/{source_index}",
            "dataset_revision": revision, "source_split": args.source_split,
            "source_row_index": source_index, "image_path": str(Path("images") / path.name),
            "source_sha256": digest, "native_labels_json": json.dumps(labels, sort_keys=True),
        })
        total = sum(map(len, pools.values()))
        if total % 20 == 0:
            print(f"LADI-260 curation: {total}/260 images", flush=True)
        if all(len(pools[name]) == LADI260_COUNTS[name] for name in LADI260_STRATA):
            break
    missing = {name: LADI260_COUNTS[name] - len(pools[name]) for name in LADI260_STRATA
               if len(pools[name]) != LADI260_COUNTS[name]}
    if missing:
        raise RuntimeError(f"Dataset stream ended before quotas filled: {missing}")
    rows = []
    for stratum_index, stratum in enumerate(LADI260_STRATA):
        impact, urgency, priority = LADI260_RUBRIC[stratum]
        pair_count = LADI260_COUNTS[stratum] // 2
        for pair_index in range(pair_count):
            split = ("train" if pair_index < (7 if pair_count == 11 else 4)
                     else "validation" if pair_index < (9 if pair_count == 11 else 7)
                     else "test")
            for item in pools[stratum][2 * pair_index:2 * pair_index + 2]:
                rows.append({**item, "selection_stratum": stratum,
                    "evidence_group_id": f"{stratum}-group{pair_index + 1}", "split": split,
                    "base_priority": priority, "task_impact": impact, "urgency": urgency,
                    "evidence_sufficiency": 1,
                    "annotation_source": "prototype_default_no_human_review",
                    "reviewer": "", "rationale": "fixed 12-stratum task rubric",
                    "review_status": "prototype_fixed"})
    manifest = output / "ladi260_semantic_scheduler_v2.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    validate_ladi260_manifest(manifest, require_reviewed=True)
    print(f"Created {manifest} with 260 unique images across all 12 strata", flush=True)


if __name__ == "__main__":
    main()
