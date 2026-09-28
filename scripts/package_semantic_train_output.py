"""Verify and package compact learned-policy artifacts for private Kaggle evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from prepare_semantic_kaggle import kaggle_username


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", default="semantic-scheduler-train-output")
    args = parser.parse_args()
    source = args.run_dir.resolve()
    target = args.output_dir.resolve()
    if source == target or source in target.parents:
        raise ValueError("Output directory must be separate from the training run")
    run_manifest = source / "run_manifest.json"
    if not run_manifest.is_file():
        raise FileNotFoundError(run_manifest)
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    hashes = manifest.get("artifacts_sha256", {})
    if not hashes or manifest.get("status") != "trained_on_reviewed_manifest":
        raise ValueError("Run manifest does not describe reviewed-data training")
    target.mkdir(parents=True, exist_ok=True)
    packaged_hashes = {}
    for name, expected in hashes.items():
        source_path = source / name
        if not source_path.is_file() or sha256(source_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed training artifact: {name}")
        destination = target / Path(name).name
        if destination.name in packaged_hashes:
            raise ValueError(f"Packaged model name collides: {destination.name}")
        shutil.copy2(source_path, destination)
        packaged_hashes[destination.name] = expected
    manifest["artifacts_sha256"] = packaged_hashes
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "title": "Semantic Scheduler Train Output",
        "id": f"{kaggle_username()}/{args.slug}",
        "licenses": [{"name": "unknown"}],
        "description": "Private generated model weights and feature cache for semantic scheduler evaluation.",
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Private compact training-output package: {target}")
    print(f"Create with: kaggle datasets create -p {target} --private")


if __name__ == "__main__":
    main()
