"""Stage a private, exact-source Kaggle cache-only experiment bundle."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=ROOT / "kaggle/static-latent")
    args = parser.parse_args()
    if not args.owner or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.owner):
        raise ValueError("Invalid Kaggle owner")
    from Evaluation.latent.experiment import read_batch, source_provenance
    read_batch(args.batch)
    dataset = args.stage / "dataset"
    kernel = args.stage / "kernel"
    dataset.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "PROJECT_MAP.md", ROOT / "Scheduler/config/static_latent_demo.json",
             ROOT / "docs/STATIC_LATENT_SCHEDULER.md"]
    for owner in ("Evaluation", "Scheduler", "Archived.owners_before_simplification_20261027.SemParam", "Codec"):
        files.extend(path for path in (ROOT / owner).rglob("*.py") if "__pycache__" not in path.parts)
    with zipfile.ZipFile(dataset / "static_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("snapshot_provenance.json", json.dumps(source_provenance(ROOT), indent=2))
    shutil.copyfile(args.batch, dataset / "latent_batch.json")
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    dataset_id = args.owner + "/static-latent-scheduler-cache"
    kernel_id = args.owner + "/static-latent-scheduler"
    write(dataset / "dataset-metadata.json", {
        "title": "Static Latent Scheduler Cache", "id": dataset_id,
        "licenses": [{"name": "other"}]})
    shutil.copyfile(ROOT / "Scheduler/notebooks/04_static_latent_scheduler.ipynb", kernel / "static_latent.ipynb")
    write(kernel / "kernel-metadata.json", {
        "id": kernel_id, "title": "Static Latent Scheduler", "code_file": "static_latent.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_internet": False, "dataset_sources": [dataset_id],
        "competition_sources": [], "kernel_sources": []})
    print(f"kaggle datasets create -p {dataset}")
    print(f"kaggle kernels push -p {kernel}")
    print(f"kaggle kernels logs {kernel_id} -f")
    print(f"kaggle kernels output {kernel_id} -p downloaded/static-latent")


if __name__ == "__main__":
    main()
