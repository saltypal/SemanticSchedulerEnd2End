"""Freeze the pending LADI-60 draft as an explicitly unreviewed prototype set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from INFRA.ladi_manifest import MANIFEST_FIELDS, validate_ladi60_manifest


def freeze_prototype(source: Path, destination: Path) -> Path:
    if source.resolve() == destination.resolve():
        raise ValueError("Prototype output must not overwrite the draft manifest")
    if source.resolve().parent != destination.resolve().parent:
        raise ValueError("Prototype manifest must share the draft image directory")
    rows = validate_ladi60_manifest(source, require_reviewed=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixed_rows = []
    for row in rows:
        image_path = Path(row["image_path"])
        fixed_rows.append({
            **row,
            "image_path": str(Path("images") / image_path.name),
            "evidence_sufficiency": "1",
            "annotation_source": "prototype_default_no_human_review",
            "reviewer": "",
            "rationale": (
                "Prototype default: P7=1 and CLIP-proposed evidence pair; "
                "not independently human reviewed."
            ),
            "review_status": "prototype_fixed",
        })
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(fixed_rows)
    validate_ladi60_manifest(destination, require_reviewed=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze_prototype(args.source, args.output))


if __name__ == "__main__":
    main()
