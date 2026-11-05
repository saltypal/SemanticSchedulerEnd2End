import concurrent.futures
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


OWNER = "satyapaladugu"
SLUG = "crisismmd1000-spi-labeling"
VERSION = 6
KERNEL_ID = 141416127
SOURCE_SHA256 = "8AD229ADB25034DDB0673EC5A26D88C8AD8E33CF5A3D714BB106E3FF3DF97C85"
PROJECT = Path(r"D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End")
RESULTS = PROJECT / "experiments/full_medic_labeling/runs/v6_results"
RUN_DIR = Path(__file__).resolve().parent
STAGING = RUN_DIR / "staging"
RUNTIME = "experiments/full_medic_labeling/runs/main_20261029T154606Z"
CHUNK_BYTES = 1024 * 1024
FILES = [
    f"{RUNTIME}/identity.json",
    f"{RUNTIME}/progress.json",
    f"{RUNTIME}/SYSTEM_PROMPT.txt",
]
for encoder in ("clip_b32", "siglip2_b16", "dino", "convnextv2_tiny"):
    for name in ("identity.json", "profile.json", "extraction_times.json"):
        FILES.append(f"{RUNTIME}/comparison/features/{encoder}/{name}")


def api_response(path):
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    req = ApiDownloadKernelOutputRequest()
    req.owner_slug = OWNER
    req.kernel_slug = SLUG
    req.file_path = path
    req.version_number = VERSION
    return client, client.kernels.kernels_api_client.download_kernel_output(req)


def digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            h.update(block)
    return size, h.hexdigest().upper()


def transfer(path):
    client = None
    response = None
    try:
        client, response = api_response(path)
        if response.status_code != 200:
            return {"path": path, "status": "http_error", "status_code": response.status_code}
        expected_text = response.headers.get("Content-Length")
        expected = int(expected_text) if expected_text and expected_text.isdigit() else None
        relative = Path(*path.split("/"))
        target = STAGING / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".part")
        h = hashlib.sha256()
        size = 0
        with temp.open("wb") as out:
            for block in response.iter_content(chunk_size=CHUNK_BYTES):
                if not block:
                    continue
                size += len(block)
                if expected is not None and size > expected:
                    raise RuntimeError("actual response exceeded content length")
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if expected is None or size != expected:
            temp.unlink(missing_ok=True)
            return {"path": path, "status": "length_unverified", "expected_bytes": expected, "actual_bytes": size}
        temp.replace(target)
        return {"path": path, "status": "downloaded", "actual_bytes": size, "sha256": h.hexdigest().upper()}
    except Exception as error:
        return {"path": path, "status": "error", "error_type": type(error).__name__, "status_code": getattr(getattr(error, "response", None), "status_code", None)}
    finally:
        if response is not None:
            response.close()
        close = getattr(client, "close", None)
        if callable(close):
            close()


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    print(f"ENCODER METADATA DOWNLOAD START count={len(FILES)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(transfer, FILES))
    merged = []
    conflicts = []
    for record in records:
        if record.get("status") != "downloaded":
            continue
        relative = Path(*record["path"].split("/"))
        source = STAGING / relative
        size, h = digest(source)
        if size != record["actual_bytes"] or h != record["sha256"]:
            conflicts.append({"path": record["path"], "reason": "staging rehash mismatch"})
            continue
        target = RESULTS / relative
        if target.exists():
            old_size, old_hash = digest(target)
            if old_size != size or old_hash != h:
                conflicts.append({"path": record["path"], "reason": "existing target differs; both preserved"})
                continue
            source.unlink()
            merged.append({"path": record["path"], "status": "identical_existing", "bytes": size, "sha256": h})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            merged.append({"path": record["path"], "status": "merged_verified", "bytes": size, "sha256": h})
    manifest = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "actual_transfer_bytes": sum(r.get("actual_bytes", 0) for r in records),
        "complete": len(records) == len(FILES) and all(r.get("status") == "downloaded" for r in records) and not conflicts,
        "files": records,
        "merged": merged,
        "conflicts": conflicts,
    }
    path = RUN_DIR / "v6_encoder_metadata_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ENCODER METADATA DOWNLOAD END complete={manifest['complete']} count={len(merged)}/{len(FILES)} bytes={manifest['actual_transfer_bytes']}", flush=True)


if __name__ == "__main__":
    main()
