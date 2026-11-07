import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath


OPS = Path(__file__).resolve().parent
RECOVERY = OPS / "recovery_inspection" / "spi_recovery.zip"
SOURCE_MANIFEST = OPS / "recovery_inspection" / "spi_recovery_manifest.json"
DOWNLOAD_RECEIPT = OPS / "recovery_inspection" / "recovery_download_receipt.json"
PROJECT = Path(__file__).resolve().parents[4]
DATASET = PROJECT / "experiments" / "spi_classifier_repair" / "data" / "semanticscheduler-spi-resume-v19"
OUT_PART = DATASET / "checkpoint.bin.part"
OUT_ARCHIVE = DATASET / "checkpoint.bin"
SOURCE_RUN = "medic_spi_20261028T214226Z"
RECORDS_SHA256 = "1b0feb7ec30c3d9fb033d9df8456daf37e56b5941f35e40aeb684d02e98845ae"
OMIT_PREFIXES = (
    "campaign/comparison/dino/checkpoints/feature_shard/",
    "campaign/comparison/siglip2_b16/checkpoints/feature_shard/",
)
OMIT_FILES = set()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream, output=None):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if output is not None:
            output.write(chunk)
    return size, digest.hexdigest()


if not RECOVERY.is_file() or not SOURCE_MANIFEST.is_file() or not DOWNLOAD_RECEIPT.is_file():
    raise RuntimeError("missing validated recovery archive, source manifest, or download receipt")

download_receipt = json.loads(DOWNLOAD_RECEIPT.read_text(encoding="utf-8"))
if download_receipt["source_run_marker"] != SOURCE_RUN:
    raise RuntimeError("download receipt belongs to a different run")
if download_receipt["downloaded_bytes"] != RECOVERY.stat().st_size:
    raise RuntimeError("downloaded archive size differs from receipt")
if download_receipt["sha256"] != sha256_file(RECOVERY):
    raise RuntimeError("downloaded archive SHA-256 differs from receipt")

source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
if SOURCE_RUN not in source_manifest.get("run", ""):
    raise RuntimeError("source manifest run identity mismatch")
expected = {entry["path"]: entry for entry in source_manifest["files"]}
if len(expected) != len(source_manifest["files"]):
    raise RuntimeError("duplicate paths in source manifest")

DATASET.mkdir(parents=True, exist_ok=True)
dataset_resolved = DATASET.resolve()
if PROJECT.resolve() not in dataset_resolved.parents:
    raise RuntimeError("resume dataset staging escaped the canonical project")
if OUT_ARCHIVE.exists() or OUT_PART.exists():
    raise RuntimeError("refusing to overwrite an existing resume checkpoint package")

with zipfile.ZipFile(RECOVERY, "r") as source_zip:
    infos = [info for info in source_zip.infolist() if not info.is_dir()]
    paths = [info.filename for info in infos]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate paths in recovery archive")
    if set(paths) != set(expected):
        missing = sorted(set(expected) - set(paths))[:10]
        extra = sorted(set(paths) - set(expected))[:10]
        raise RuntimeError(f"archive/manifest path mismatch; missing={missing}, extra={extra}")
    if source_zip.testzip() is not None:
        raise RuntimeError("source recovery archive CRC test failed")

    # Only drop shards when the corresponding complete aggregate cache has a
    # successful checkpoint receipt and a manifest-verified feature payload.
    complete_encoders = set()
    for encoder in ("dino", "siglip2_b16"):
        prefix = f"campaign/comparison/{encoder}/checkpoints/all_features/"
        receipts = [name for name in paths if name.startswith(prefix) and name.endswith("/receipt.json")]
        features = [name for name in paths if name.startswith(prefix) and name.endswith("/features.npz")]
        if len(receipts) != 1 or len(features) != 1:
            raise RuntimeError(f"expected one all-features receipt and payload for {encoder}")
        receipt_entry = expected[receipts[0]]
        with source_zip.open(receipts[0]) as stream:
            size, digest = sha256_stream(stream)
        if size != receipt_entry["bytes"] or digest != receipt_entry["sha256"]:
            raise RuntimeError(f"all-features receipt checksum mismatch for {encoder}")
        with source_zip.open(receipts[0]) as stream:
            receipt = json.load(stream)
        if receipt.get("status") != "complete":
            raise RuntimeError(f"all-features cache is not complete for {encoder}")
        aggregate_entry = expected[features[0]]
        output_path = f"campaign/comparison/{encoder}/features.npz"
        if output_path not in expected:
            raise RuntimeError(f"duplicate output feature file is missing for {encoder}")
        output_entry = expected[output_path]
        if (output_entry["bytes"], output_entry["sha256"]) != (aggregate_entry["bytes"], aggregate_entry["sha256"]):
            raise RuntimeError(f"output feature file differs from completed aggregate cache for {encoder}")
        OMIT_FILES.add(output_path)
        complete_encoders.add(encoder)
    if complete_encoders != {"dino", "siglip2_b16"}:
        raise RuntimeError("not all expected duplicate-shard encoders were verified complete")

    included = []
    omitted = []
    seen = set()
    try:
        with zipfile.ZipFile(OUT_PART, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as dest_zip:
            for info in infos:
                name = info.filename
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise RuntimeError(f"unsafe checkpoint path: {name}")
                if name.startswith("campaign/") and name.lower().endswith((".img", ".jpg", ".jpeg", ".png", ".webp")):
                    raise RuntimeError(f"unexpected image payload in checkpoint archive: {name}")
                entry = expected[name]
                if not name.startswith("campaign/"):
                    skip_reason = "non_campaign_run_artifact"
                elif name in OMIT_FILES:
                    skip_reason = "duplicate_output_feature"
                elif any(name.startswith(prefix) for prefix in OMIT_PREFIXES):
                    skip_reason = "duplicate_feature_shard"
                else:
                    skip_reason = None
                skip = skip_reason is not None
                if skip_reason in ("duplicate_output_feature", "duplicate_feature_shard"):
                    valid_duplicate = name in OMIT_FILES or any(
                        name.startswith(f"campaign/comparison/{encoder}/checkpoints/feature_shard/")
                        for encoder in complete_encoders
                    )
                    if not valid_duplicate:
                        raise RuntimeError(f"attempted to omit an unverified feature shard: {name}")
                out_info = copy.copy(info)
                out_info.compress_type = zipfile.ZIP_DEFLATED
                out_info.compress_level = 1
                output_stream = None
                try:
                    if not skip:
                        output_stream = dest_zip.open(out_info, "w", force_zip64=True)
                    with source_zip.open(info, "r") as input_stream:
                        size, digest = sha256_stream(input_stream, output_stream)
                finally:
                    if output_stream is not None:
                        output_stream.close()
                if size != entry["bytes"] or digest != entry["sha256"]:
                    raise RuntimeError(f"source member hash mismatch: {name}")
                seen.add(name)
                record = {"path": name, "bytes": size, "sha256": digest}
                if skip:
                    record["reason"] = skip_reason
                    omitted.append(record)
                else:
                    included.append(record)
                if (len(included) + len(omitted)) % 50 == 0:
                    print(f"verified {len(included) + len(omitted)}/{len(infos)} members", flush=True)
    except BaseException:
        OUT_PART.unlink(missing_ok=True)
        raise

if seen != set(expected):
    raise RuntimeError("not all source manifest entries were verified")
OUT_PART.replace(OUT_ARCHIVE)
if zipfile.ZipFile(OUT_ARCHIVE, "r").testzip() is not None:
    raise RuntimeError("curated checkpoint ZIP CRC test failed")

archive_sha256 = sha256_file(OUT_ARCHIVE)
index = {
    "schema": "medic_spi_checkpoints_v1",
    "archive": "checkpoint.bin",
    "sha256": archive_sha256,
    "archive_bytes": OUT_ARCHIVE.stat().st_size,
    "records_sha256": RECORDS_SHA256,
    "source": {
        "kernel": "satyapaladugu/betterscheduler-spi-20261027finalc/19",
        "run_id": SOURCE_RUN,
        "source_archive_sha256": download_receipt["sha256"],
        "source_manifest_path": "source_recovery_manifest.json",
    },
    "included_file_count": len(included),
    "omitted_duplicate_feature_shard_count": sum(entry["reason"] == "duplicate_feature_shard" for entry in omitted),
    "omitted_duplicate_feature_shard_bytes": sum(entry["bytes"] for entry in omitted if entry["reason"] == "duplicate_feature_shard"),
    "omitted_duplicate_output_feature_count": sum(entry["reason"] == "duplicate_output_feature" for entry in omitted),
    "omitted_non_campaign_run_artifact_count": sum(entry["reason"] == "non_campaign_run_artifact" for entry in omitted),
    "verified_complete_aggregate_caches": sorted(complete_encoders),
    "omitted_duplicate_output_feature_files": sorted(OMIT_FILES),
    "member_prefix": "campaign/",
}
(DATASET / "resume_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(DATASET / "source_recovery_manifest.json").write_text(SOURCE_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
resume_manifest = {
    "schema": "medic_spi_checkpoints_v1",
    "source_run_id": SOURCE_RUN,
    "source_recovery_archive_sha256": download_receipt["sha256"],
    "records_sha256": RECORDS_SHA256,
    "included": included,
    "omitted": omitted,
}
(DATASET / "resume_manifest.json").write_text(json.dumps(resume_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(json.dumps({
    "dataset_stage": str(DATASET),
    "source_archive_bytes": RECOVERY.stat().st_size,
    "source_archive_sha256": download_receipt["sha256"],
    "included_files": len(included),
    "included_uncompressed_bytes": sum(entry["bytes"] for entry in included),
    "omitted_files": len(omitted),
    "omitted_uncompressed_bytes": sum(entry["bytes"] for entry in omitted),
    "omitted_duplicate_feature_shard_count": sum(entry["reason"] == "duplicate_feature_shard" for entry in omitted),
    "omitted_duplicate_output_feature_count": sum(entry["reason"] == "duplicate_output_feature" for entry in omitted),
    "omitted_non_campaign_run_artifact_count": sum(entry["reason"] == "non_campaign_run_artifact" for entry in omitted),
    "checkpoint_bin_bytes": OUT_ARCHIVE.stat().st_size,
    "checkpoint_bin_sha256": archive_sha256,
    "records_sha256": RECORDS_SHA256,
}, indent=2), flush=True)
