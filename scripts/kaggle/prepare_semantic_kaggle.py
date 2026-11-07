"""Stage the two canonical notebooks for Kaggle CLI without hard-coded users."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "train": ("01_train_semantic_scheduler.ipynb", "semantic-scheduler-train"),
    "evaluate": ("02_evaluate_all_policies.ipynb", "semantic-scheduler-evaluate"),
}


def kaggle_username() -> str:
    value = os.environ.get("KAGGLE_USERNAME")
    if value:
        return value
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    path = config_dir / "kaggle.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8")).get("username")
        if value:
            return str(value)
    raise RuntimeError("Set KAGGLE_USERNAME or configure ~/.kaggle/kaggle.json")


def stage_kernel(kind: str, *, manifest_dataset: str | None = None,
                 train_output_dataset: str | None = None,
                 payload_cost_dataset: str | None = None) -> Path:
    filename, slug = NOTEBOOKS[kind]
    source = ROOT / "Scheduler" / "notebooks" / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = ROOT / "kaggle" / kind
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / filename)
    sources = []
    if manifest_dataset:
        sources.append(manifest_dataset)
    if kind == "train" and payload_cost_dataset:
        sources.append(payload_cost_dataset)
    if kind == "evaluate" and train_output_dataset:
        sources.append(train_output_dataset)
    metadata = {
        "id": f"{kaggle_username()}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if kind == "train" else "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4" if kind == "train" else "",
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    path = destination / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("train", "evaluate", "both"))
    parser.add_argument("--manifest-dataset", help="username/private-LADI-prototype-slug")
    parser.add_argument("--train-output-dataset", help="username/private-training-output-slug")
    parser.add_argument("--payload-cost-dataset", help="username/private-measured-cost-slug")
    options = parser.parse_args()
    for kind in NOTEBOOKS if options.kind == "both" else (options.kind,):
        print(stage_kernel(kind, manifest_dataset=options.manifest_dataset,
                           train_output_dataset=options.train_output_dataset,
                           payload_cost_dataset=options.payload_cost_dataset))


if __name__ == "__main__":
    main()
