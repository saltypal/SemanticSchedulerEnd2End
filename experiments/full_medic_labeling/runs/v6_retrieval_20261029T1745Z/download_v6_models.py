import concurrent.futures
import hashlib
import json
import os
import threading
import time
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
PLAN = RUN_DIR / "v6_model_size_plan.json"
HARD_CAP = 5 * 1024**3
MAX_WORKERS = 4
CHUNK_BYTES = 1024 * 1024


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest().upper()


def fetch(record, reserved, lock):
    remote_path = record["path"]
    relative = Path(*remote_path.split("/"))
    target = STAGING / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = record["content_length"]

    for attempt in range(1, 4):
        api = KaggleApi()
        api.authenticate()
        client = api.build_kaggle_client()
        temporary = target.with_name(target.name + ".part")
        response = None
        try:
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug = OWNER
            request.kernel_slug = SLUG
            request.file_path = remote_path
            request.version_number = VERSION
            response = client.kernels.kernels_api_client.download_kernel_output(request)
            if response.status_code != 200:
                status = response.status_code
                response.close()
                if status in (408, 429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                return {**record, "status": "http_error", "status_code": status, "attempts": attempt}

            actual = 0
            digest = hashlib.sha256()
            with temporary.open("wb") as out:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    actual += len(chunk)
                    if actual > expected:
                        raise RuntimeError("actual bytes exceeded probed content length")
                    with lock:
                        if reserved["downloaded"] + actual > reserved["available"]:
                            raise RuntimeError("hard 5 GiB retrieval cap reached")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            response.close()
            if actual != expected:
                raise RuntimeError("actual bytes differ from probed content length")
            temporary.replace(target)
            with lock:
                reserved["downloaded"] += actual
            return {
                **record,
                "status": "downloaded",
                "attempts": attempt,
                "actual_bytes": actual,
                "sha256": digest.hexdigest().upper(),
            }
        except Exception as error:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            status = getattr(getattr(error, "response", None), "status_code", None)
            transient = status in (408, 429, 500, 502, 503, 504) or type(error).__name__ in {
                "Timeout",
                "ConnectTimeout",
                "ReadTimeout",
                "ConnectionError",
            }
            if transient and attempt < 3:
                time.sleep(2**attempt)
                continue
            return {
                **record,
                "status": "error",
                "attempts": attempt,
                "error_type": type(error).__name__,
                "status_code": status,
            }
        finally:
            if response is not None:
                response.close()
            close = getattr(client, "close", None)
            if callable(close):
                close()


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    records = plan["files"]
    if plan.get("missing_or_unprobed_count") != 0:
        raise SystemExit("At least one model artifact lacks a verified size probe")
    if any("/checkpoints/" in item["path"] for item in records):
        raise SystemExit("Refusing to retrieve duplicate checkpoint artifacts")

    existing = sum(path.stat().st_size for path in RESULTS.rglob("*") if path.is_file())
    existing_paths = {
        item["path"]: item
        for item in json.loads((RUN_DIR / "v6_priority_download_manifest.json").read_text(encoding="utf-8"))["files"]
        if item.get("status") == "downloaded"
    }
    missing = []
    already = []
    for item in records:
        relative = Path(*item["path"].split("/"))
        local = RESULTS / relative
        if not local.is_file():
            missing.append(item)
            continue
        actual, digest = digest_file(local)
        prior = existing_paths.get(item["path"])
        if prior and actual == prior.get("actual_bytes") and digest == prior.get("sha256"):
            already.append({**item, "status": "identical_existing_verified", "actual_bytes": actual, "sha256": digest})
        elif actual == item["content_length"]:
            # Existing local candidate output is not replaced; content identity
            # will be compared against the remote response only if staged later.
            already.append({**item, "status": "existing_unverified_against_remote", "actual_bytes": actual, "sha256": digest})
        else:
            missing.append(item)

    available = HARD_CAP - existing
    planned = sum(item["content_length"] for item in missing)
    if planned > available:
        raise SystemExit(f"Refusing transfer: planned {planned} bytes exceed remaining cap {available}")
    print(
        f"MODEL DOWNLOAD START total={len(records)} missing={len(missing)} existing={len(already)} "
        f"planned_bytes={planned} available={available} workers={MAX_WORKERS}",
        flush=True,
    )

    reserved = {"downloaded": 0, "available": available}
    lock = threading.Lock()
    fetched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch, item, reserved, lock) for item in missing]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            fetched.append(result)
            if index % 20 == 0 or index == len(futures):
                print(f"MODEL DOWNLOAD progress={index}/{len(futures)} downloaded_bytes={reserved['downloaded']}", flush=True)

    merged = []
    conflicts = []
    for record in fetched:
        if record.get("status") != "downloaded":
            continue
        relative = Path(*record["path"].split("/"))
        source = STAGING / relative
        actual, digest = digest_file(source)
        if actual != record["actual_bytes"] or digest != record["sha256"]:
            conflicts.append({"path": record["path"], "reason": "staging rehash mismatch"})
            continue
        target = RESULTS / relative
        if target.exists():
            old_size, old_hash = digest_file(target)
            if old_size != actual or old_hash != digest:
                conflicts.append({"path": record["path"], "reason": "existing target differs; both preserved"})
                continue
            source.unlink()
            merged.append({"path": record["path"], "status": "identical_existing", "bytes": actual, "sha256": digest})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            merged.append({"path": record["path"], "status": "merged_verified", "bytes": actual, "sha256": digest})

    all_records = already + fetched
    manifest = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "hard_cap_bytes": HARD_CAP,
        "existing_results_bytes_before_model_transfer": existing,
        "planned_transfer_bytes": planned,
        "actual_transfer_bytes": reserved["downloaded"],
        "completed_models": sum(r.get("status") in ("downloaded", "identical_existing_verified") for r in all_records),
        "candidate_model_count": len(records),
        "status": "complete" if len(all_records) == len(records) and not conflicts and all(r.get("status") in ("downloaded", "identical_existing_verified") for r in all_records) else "partial",
        "files": sorted(all_records, key=lambda item: item["path"]),
        "merged": merged,
        "conflicts": conflicts,
    }
    manifest_path = RUN_DIR / "v6_models_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"MODEL DOWNLOAD END status={manifest['status']} actual_bytes={reserved['downloaded']} "
        f"complete={manifest['completed_models']}/{len(records)} conflicts={len(conflicts)} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
