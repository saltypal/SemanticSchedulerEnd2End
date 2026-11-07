from pathlib import Path
import hashlib
import json
import zipfile

retrieval = Path("experiments/spi_four_class/runs/20261027_spi_evidence_v3a/retrieved_version2")
run = retrieval / "experiments/spi_four_class/runs/20261027_spi_evidence_v3a"
campaign = run / "campaign"
checksums = json.loads((campaign / "checksums.json").read_text())
recovery = campaign / "recovery.zip"

direct_mismatches = []
for relative_path, expected in checksums.items():
    path = campaign / relative_path
    if not path.is_file():
        direct_mismatches.append(f"{relative_path}:missing")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        direct_mismatches.append(f"{relative_path}:hash")

with zipfile.ZipFile(recovery) as archive:
    crc_error = archive.testzip()
    names = archive.namelist()
    zip_mismatches = []
    for relative_path, expected in checksums.items():
        if relative_path not in names:
            zip_mismatches.append(f"{relative_path}:missing")
        elif hashlib.sha256(archive.read(relative_path)).hexdigest() != expected:
            zip_mismatches.append(f"{relative_path}:hash")

runtime = json.loads((run / "runtime.json").read_text())
source_dir = retrieval / "spi_source_20261027_spi_evidence_v3a"
source_manifest = json.loads((source_dir / "SOURCE_MANIFEST.json").read_text())
source_mismatches = [
    name for name, expected in source_manifest["files"].items()
    if not (source_dir / name).is_file()
    or hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != expected
]

summary = {
    "retrieved_files": sum(1 for path in retrieval.rglob("*") if path.is_file()),
    "checksum_entries": len(checksums),
    "direct_mismatches": direct_mismatches,
    "recovery_zip_sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
    "recovery_zip_entries": len(names),
    "recovery_zip_crc": "ok" if crc_error is None else crc_error,
    "zip_mismatches": zip_mismatches,
    "runtime": runtime,
    "runtime_source_tag": source_manifest.get("tag"),
    "runtime_source_files": len(source_manifest["files"]),
    "runtime_source_mismatches": source_mismatches,
    "campaign_status": json.loads((campaign / "status.json").read_text()),
    "cell_status": json.loads((run / "cell_evidence/cell_status.json").read_text()),
}
(retrieval / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
