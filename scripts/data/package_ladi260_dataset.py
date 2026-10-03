"""Package the fixed LADI-260 benchmark for Kaggle."""

from __future__ import annotations
import argparse, csv, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from INFRA.data.ladi_manifest import MANIFEST_FIELDS, validate_ladi260_manifest
from prepare_semantic_kaggle import kaggle_username

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", default="semantic-scheduler-ladi260-public")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    rows = validate_ladi260_manifest(args.manifest, require_reviewed=True)
    target = args.output_dir.resolve(); target.mkdir(parents=True, exist_ok=True)
    packed = []
    for row in rows:
        destination = target / f"{row['image_id']}.png"
        shutil.copy2(row["image_path"], destination)
        packed.append({**row, "image_path": destination.name})
    manifest = target / "ladi260_semantic_scheduler_v2.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(packed)
    metadata = {"title": "Semantic Scheduler LADI260 12-Class Benchmark",
        "id": f"{kaggle_username()}/{args.slug}", "licenses": [{"name": "CC-BY-4.0"}],
        "description": "260-image LADI v2 prototype benchmark spanning all 12 native labels; task rubric and evidence pairing are prototype assumptions."}
    (target / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (target / "SOURCE_AND_LICENSE.txt").write_text(
        "MIT Lincoln Laboratory LADI v2, CC BY 4.0. Prototype scheduling annotations are not human reviewed.\n",
        encoding="utf-8")
    validate_ladi260_manifest(manifest, require_reviewed=True)
    if args.zip:
        print(shutil.make_archive(str(target), "zip", root_dir=target))
    print(target)

if __name__ == "__main__":
    main()
