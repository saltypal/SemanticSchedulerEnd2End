from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

retrieval = Path("experiments/spi_four_class/runs/20261027_spi_evidence_v3a/retrieved_version2")
archive = retrieval / "experiments/spi_four_class/runs/20261027_spi_evidence_v3a/campaign/recovery.zip"
destination = retrieval / "recovery_extracted"
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:
    members = bundle.namelist()
    for name in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {name}")
    bundle.extractall(destination)
receipt = {
    "source_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "zip_entries": len(members),
    "extracted_files": sum(1 for path in destination.rglob("*") if path.is_file()),
    "members": members,
}
(destination / "extraction_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps({key: receipt[key] for key in ("source_zip_sha256", "zip_entries", "extracted_files")}, indent=2))
