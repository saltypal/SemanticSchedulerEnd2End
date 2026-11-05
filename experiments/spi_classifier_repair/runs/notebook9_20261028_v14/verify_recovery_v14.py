"""Verify and safely extract the terminal v14 Kaggle recovery archive."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


RUN = Path(__file__).resolve().parent
MONITOR = RUN / "monitor_v14"
ARCHIVE = MONITOR / "run_recovery.zip"
DESTINATION = MONITOR / "recovery_verified"

if not ARCHIVE.is_file():
    raise SystemExit(f"Recovery archive is missing: {ARCHIVE}")
if DESTINATION.exists():
    raise SystemExit(f"Preserving existing verification output: {DESTINATION}")

archive_sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
with zipfile.ZipFile(ARCHIVE) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"ZIP CRC failure: {bad}")
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate archive paths")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"Unsafe archive path: {name}")
    if "retrieval_checksums.json" not in names:
        raise SystemExit("Recovery checksum manifest is missing")
    checks = json.loads(archive.read("retrieval_checksums.json"))
    expected = set(checks)
    actual = set(names) - {"retrieval_checksums.json"}
    if expected != actual:
        raise SystemExit(
            f"Manifest membership mismatch missing={sorted(expected-actual)[:10]} "
            f"unexpected={sorted(actual-expected)[:10]}"
        )
    mismatches = []
    for name, expected_sha in checks.items():
        actual_sha = hashlib.sha256(archive.read(name)).hexdigest()
        if actual_sha != expected_sha:
            mismatches.append({"path": name, "expected": expected_sha, "actual": actual_sha})
    if mismatches:
        raise SystemExit(f"Recovery hash mismatch: {mismatches[:10]}")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for name in sorted(actual):
        target = DESTINATION.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
    (DESTINATION / "retrieval_checksums.json").write_bytes(
        archive.read("retrieval_checksums.json")
    )

report = {
    "schema": "notebook9_v14_recovery_verification_v1",
    "archive": str(ARCHIVE),
    "archive_bytes": ARCHIVE.stat().st_size,
    "archive_sha256": archive_sha,
    "zip_crc": "pass",
    "payload_files": len(checks),
    "manifest_hashes": f"{len(checks)}/{len(checks)} pass",
    "missing": 0,
    "unexpected": 0,
    "unsafe": 0,
    "hash_mismatches": 0,
    "verified_extraction": str(DESTINATION),
}
(MONITOR / "retrieval_verify.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
