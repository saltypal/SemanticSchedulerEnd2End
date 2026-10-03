"""Package an experiment-ready LADI-60 manifest and images for private Kaggle input."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from INFRA.data.ladi_manifest import MANIFEST_FIELDS, validate_ladi60_manifest
from prepare_semantic_kaggle import kaggle_username


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", default="semantic-scheduler-ladi60-prototype")
    parser.add_argument("--zip", action="store_true", help="Also create <output-dir>.zip")
    parser.add_argument("--costs", type=Path,
                        help="Measured payload_costs.csv to include in the same Kaggle dataset")
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
    if args.costs is not None:
        with args.costs.open(newline="", encoding="utf-8") as handle:
            costs = list(csv.DictReader(handle))
        expected_ids = {row["image_id"] for row in rows}
        if len(costs) != 60 or {row["image_id"] for row in costs} != expected_ids:
            raise ValueError("Measured cost IDs must match all 60 manifest images exactly")
        if ({row["resource_unit"] for row in costs} != {"complex_channel_uses"}
                or len({row["checkpoint_sha256"] for row in costs}) != 1
                or any(int(row["channel_uses"]) <= 0 for row in costs)):
            raise ValueError("Payload costs must have one measured unit and checkpoint")
        shutil.copy2(args.costs, target / "payload_costs.csv")
    metadata = {
        "title": "Semantic Scheduler LADI60 Prototype",
        "id": f"{kaggle_username()}/{args.slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": (
            "Private prototype subset sourced from MITLL/LADI-v2-dataset. "
            "P5/P6 use a task rubric; P7 defaults to one; CLIP-proposed pairs "
            "are not human reviewed. Original LADI v2 data: MIT Lincoln "
            "Laboratory, CC BY 4.0; see SOURCE_AND_LICENSE.txt."
        ),
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (target / "SOURCE_AND_LICENSE.txt").write_text(
        "Source: MIT Lincoln Laboratory, LADI v2 dataset\n"
        "https://huggingface.co/datasets/MITLL/LADI-v2-dataset\n"
        f"Revision: {rows[0]['dataset_revision']}\n"
        "License for dataset imagery: Creative Commons Attribution 4.0 (CC BY 4.0).\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "This subset selects 60 images and converts them to RGB PNG; it adds\n"
        "prototype scheduling annotations and checkpoint-derived payload costs.\n"
        "P7=1 and evidence pairings have not been independently reviewed.\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path.read_bytes()).hexdigest()
              for path in sorted(target.glob("*.csv"))}
    (target / "bundle-hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Private Kaggle dataset package: {target}")
    if args.zip:
        print(f"Manual-upload ZIP: {shutil.make_archive(str(target), 'zip', root_dir=target)}")
    print(f"Create privately with: kaggle datasets create -p {target}")


if __name__ == "__main__":
    main()
