"""Package a genuinely reviewed LADI-60 manifest and images for private Kaggle input."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from INFRA.ladi_manifest import MANIFEST_FIELDS, validate_ladi60_manifest
from prepare_semantic_kaggle import kaggle_username


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", default="semantic-scheduler-ladi60-reviewed")
    args = parser.parse_args()
    rows = validate_ladi60_manifest(args.manifest, require_reviewed=True)
    target = args.output_dir.resolve()
    source = args.manifest.parent.resolve()
    if target == source or source in target.parents:
        raise ValueError("Output directory must not be inside the source image bundle")
    target.mkdir(parents=True, exist_ok=True)
    packed = []
    for row in rows:
        original = Path(row["image_path"])
        destination = target / f"{row['image_id']}.png"
        shutil.copy2(original, destination)
        packed.append({**row, "image_path": destination.name})
    manifest = target / "ladi60_semantic_scheduler_v1.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(packed)
    validate_ladi60_manifest(manifest, require_reviewed=True)
    metadata = {
        "title": "Semantic Scheduler LADI60 Reviewed",
        "id": f"{kaggle_username()}/{args.slug}",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Reviewed scheduling subset sourced from MITLL/LADI-v2-dataset. "
            "See the upstream dataset card for attribution and license terms."
        ),
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Private Kaggle dataset package: {target}")
    print(f"Create with: kaggle datasets create -p {target} --private")


if __name__ == "__main__":
    main()
