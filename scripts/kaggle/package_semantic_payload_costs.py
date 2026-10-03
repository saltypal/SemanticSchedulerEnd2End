"""Package verified real SwinJSCC channel-use measurements for Kaggle training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from INFRA.data.ladi_manifest import validate_ladi60_manifest
from prepare_semantic_kaggle import kaggle_username


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", default="semantic-scheduler-measured-costs")
    args = parser.parse_args()
    images = validate_ladi60_manifest(args.manifest, require_reviewed=False)
    with args.costs.open(newline="", encoding="utf-8") as handle:
        costs = list(csv.DictReader(handle))
    if len(costs) != 60 or {row["image_id"] for row in costs} != {row["image_id"] for row in images}:
        raise ValueError("Measured payload costs must cover exactly the curated 60 images")
    if {row["resource_unit"] for row in costs} != {"complex_channel_uses"}:
        raise ValueError("Only measured complex channel uses may be packaged")
    hashes = {row.get("checkpoint_sha256") for row in costs}
    if len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError("A measured cost package needs one checkpoint hash")
    if any(int(row["channel_uses"]) <= 0 for row in costs):
        raise ValueError("Every image needs a positive measured channel-use count")
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.costs, target / "payload_costs.csv")
    metadata = {
        "title": "Semantic Scheduler Measured Costs",
        "id": f"{kaggle_username()}/{args.slug}",
        "licenses": [{"name": "unknown"}],
        "description": "Checkpoint-derived SwinJSCC whole-latent complex-channel-use measurements for 60 LADI v2 images.",
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Private measured-cost package: {target}")
    print(f"Create with: kaggle datasets create -p {target} --private")


if __name__ == "__main__":
    main()
