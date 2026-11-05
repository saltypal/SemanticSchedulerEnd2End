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
RUNTIME_PREFIX = "experiments/full_medic_labeling/runs/main_20261029T154606Z"
PROJECT = Path(r"D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End")
RESULTS = PROJECT / "experiments/full_medic_labeling/runs/v6_results"
RUN_DIR = Path(__file__).resolve().parent
STAGING = RUN_DIR / "staging"
HARD_CAP = 5 * 1024**3
MAX_WORKERS = 4
CHUNK_BYTES = 1024 * 1024

REMOTE_FILES = [
    f"{RUNTIME_PREFIX}/label_cache.sqlite",
    f"{RUNTIME_PREFIX}/labels.jsonl",
    f"{RUNTIME_PREFIX}/selected_manifest.json",
    f"{RUNTIME_PREFIX}/comparison/records.json",
    f"{RUNTIME_PREFIX}/comparison/split_protocol.json",
    f"{RUNTIME_PREFIX}/comparison/features/siglip2_b16/identity.json",
    f"{RUNTIME_PREFIX}/comparison/class_weighted/siglip2_b16/rbf_svm.joblib",
    f"{RUNTIME_PREFIX}/comparison/features/clip_b32/features.npz",
    f"{RUNTIME_PREFIX}/comparison/features/siglip2_b16/features.npz",
    f"{RUNTIME_PREFIX}/comparison/features/dino/features.npz",
    f"{RUNTIME_PREFIX}/comparison/features/convnextv2_tiny/features.npz",
]


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest().upper()


def api_download_response(path):
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = OWNER
    request.kernel_slug = SLUG
    request.file_path = path
    request.version_number = VERSION
    return client, client.kernels.kernels_api_client.download_kernel_output(request)


def probe(path):
    client = None
    try:
        client, response = api_download_response(path)
        length = response.headers.get("Content-Length")
        result = {
            "path": path,
            "status_code": response.status_code,
            "content_length": int(length) if length and length.isdigit() else None,
            "content_type": response.headers.get("Content-Type"),
        }
        response.close()
        return result
    except Exception as error:
        response = getattr(error, "response", None)
        return {
            "path": path,
            "status_code": getattr(response, "status_code", None),
            "error_type": type(error).__name__,
        }
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def fetch(item, reservations, lock):
    path = item["path"]
    relative = Path(*path.split("/"))
    target = STAGING / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        size, digest = digest_file(target)
        return {**item, "status": "already_staged", "actual_bytes": size, "sha256": digest}

    for attempt in range(1, 4):
        client = None
        temp = target.with_name(target.name + ".part")
        try:
            client, response = api_download_response(path)
            status = response.status_code
            if status != 200:
                response.close()
                if status in (408, 429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                return {**item, "status": "http_error", "status_code": status, "attempts": attempt}

            expected = item["content_length"]
            observed = 0
            digest = hashlib.sha256()
            with temp.open("wb") as out:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    observed += len(chunk)
                    if expected is not None and observed > expected:
                        raise RuntimeError("response exceeded advertised content length")
                    with lock:
                        if reservations["actual_bytes"] + observed > reservations["available_bytes"]:
                            raise RuntimeError("hard retrieval byte cap would be exceeded")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            response.close()
            if expected is not None and observed != expected:
                raise RuntimeError("download size differs from advertised content length")
            temp.replace(target)
            with lock:
                reservations["actual_bytes"] += observed
            return {
                **item,
                "status": "downloaded",
                "attempts": attempt,
                "actual_bytes": observed,
                "sha256": digest.hexdigest().upper(),
            }
        except Exception as error:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
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
                **item,
                "status": "error",
                "attempts": attempt,
                "error_type": type(error).__name__,
                "status_code": status,
            }
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    return {**item, "status": "retry_exhausted"}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    current_bytes = sum(p.stat().st_size for p in RESULTS.rglob("*") if p.is_file())
    available_bytes = HARD_CAP - current_bytes
    if available_bytes <= 0:
        raise SystemExit("Existing results already meet/exceed the 5 GiB hard cap")

    print(f"PROBE START files={len(REMOTE_FILES)} version={VERSION}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        probes = list(pool.map(probe, REMOTE_FILES))
    plan_path = RUN_DIR / "v6_priority_size_plan.json"
    plan = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "hard_cap_bytes": HARD_CAP,
        "existing_results_bytes": current_bytes,
        "available_bytes": available_bytes,
        "files": probes,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if any(p.get("status_code") != 200 or p.get("content_length") is None for p in probes):
        raise SystemExit("A requested v6 artifact failed size probe; inspect size plan before continuing")
    planned = sum(p["content_length"] for p in probes)
    plan["planned_bytes"] = planned
    plan["remaining_after_plan_bytes"] = available_bytes - planned
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if planned > available_bytes:
        raise SystemExit("Planned transfer exceeds the 5 GiB hard cap; no downloads started")
    print(f"PROBE END planned_bytes={planned} available_bytes={available_bytes}", flush=True)

    reservations = {"actual_bytes": 0, "available_bytes": available_bytes}
    lock = threading.Lock()
    print("DOWNLOAD START workers=4", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch, item, reservations, lock) for item in probes]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result.get("status") == "downloaded":
                print(
                    f"DOWNLOADED {Path(result['path']).name} bytes={result['actual_bytes']} sha256={result['sha256']}",
                    flush=True,
                )
            else:
                print(
                    f"FETCH_RESULT {Path(result['path']).name} status={result.get('status')} "
                    f"type={result.get('error_type')} http={result.get('status_code')}",
                    flush=True,
                )

    manifest = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_prefix": RUNTIME_PREFIX,
        "hard_cap_bytes": HARD_CAP,
        "existing_results_bytes_before_transfer": current_bytes,
        "planned_transfer_bytes": planned,
        "actual_transfer_bytes": reservations["actual_bytes"],
        "status": "complete" if all(r.get("status") == "downloaded" for r in results) else "partial",
        "files": sorted(results, key=lambda x: x["path"]),
    }
    manifest_path = RUN_DIR / "v6_priority_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Merge only files with verified byte count/hash; never overwrite an existing result.
    merged = []
    conflicts = []
    for record in manifest["files"]:
        if record.get("status") != "downloaded":
            continue
        relative = Path(*record["path"].split("/"))
        source = STAGING / relative
        size, digest = digest_file(source)
        if size != record["actual_bytes"] or digest != record["sha256"]:
            conflicts.append({"path": record["path"], "reason": "staging rehash mismatch"})
            continue
        target = RESULTS / relative
        if target.exists():
            existing_size, existing_digest = digest_file(target)
            if existing_size != size or existing_digest != digest:
                conflicts.append({"path": record["path"], "reason": "existing target differs; preserved both"})
                continue
            source.unlink()
            merged.append({"path": record["path"], "status": "identical_existing"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        merged.append({"path": record["path"], "status": "merged_verified", "bytes": size, "sha256": digest})

    manifest["merged"] = merged
    manifest["conflicts"] = conflicts
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"MANIFEST {manifest_path} status={manifest['status']} actual_bytes={manifest['actual_transfer_bytes']} "
        f"merged={len(merged)} conflicts={len(conflicts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
