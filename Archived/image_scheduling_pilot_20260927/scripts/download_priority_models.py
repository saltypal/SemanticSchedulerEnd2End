"""Download only scheduler-pilot model files into models/priority_classifiers.

Run: python scripts/download_priority_models.py --models clip ladi_small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from SemParam.priority_models import MODEL_SPECS, model_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "models" / "priority_classifiers")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.root.mkdir(parents=True, exist_ok=True)
    failures = []
    for key in args.models:
        spec = MODEL_SPECS[key]
        directory = model_directory(args.root, key)
        directory.mkdir(parents=True, exist_ok=True)
        print(f"{key}: syncing {spec.repository} -> {directory}", flush=True)
        try:
            snapshot_download(repo_id=spec.repository, local_dir=directory, max_workers=4)
            if not (directory / "config.json").is_file():
                raise FileNotFoundError(f"config.json missing after download in {directory}")
            (directory / "pilot_model_source.json").write_text(
                json.dumps({"repository": spec.repository, "model_key": key}, indent=2), encoding="utf-8"
            )
            print(f"{key}: ready", flush=True)
        except Exception as exc:
            failures.append(key)
            print(f"{key}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f"Failed models: {', '.join(failures)}. Rerun this command to resume downloads.")


if __name__ == "__main__":
    main()
