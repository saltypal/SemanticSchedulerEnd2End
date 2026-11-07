import concurrent.futures
import hashlib
import json
import os
import re
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
LOG = RESULTS / "crisismmd1000-spi-labeling.log"
HARD_CAP = 5 * 1024**3
MAX_WORKERS = 4
CHUNK_BYTES = 1024 * 1024
PATH_RE = re.compile(r"/kaggle/working/(experiments/[^\s\"']+?\.json)")


def client_response(path):
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = OWNER
    request.kernel_slug = SLUG
    request.file_path = path
    request.version_number = VERSION
    return client, client.kernels.kernels_api_client.download_kernel_output(request)


def safe_request(path):
    client = None
    try:
        client, response = client_response(path)
        result = {
            "path": path,
            "status_code": response.status_code,
            "content_length": int(response.headers["Content-Length"])
            if response.headers.get("Content-Length", "").isdigit()
            else None,
        }
        response.close()
        return result
    except Exception as error:
        response = getattr(error, "response", None)
        return {"path": path, "status_code": getattr(response, "status_code", None), "error_type": type(error).__name__}
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest().upper()


def fetch(record, budget, lock):
    remote_path = record["path"]
    relative = Path(*remote_path.split("/"))
    target = STAGING / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    expected = record["content_length"]
    for attempt in range(1, 4):
        client = None
        response = None
        try:
            client, response = client_response(remote_path)
            if response.status_code != 200:
                code = response.status_code
                response.close()
                if code in (408, 429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                return {**record, "status": "http_error", "status_code": code, "attempts": attempt}
            size = 0
            digest = hashlib.sha256()
            with temporary.open("wb") as out:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > expected:
                        raise RuntimeError("response exceeded probed content length")
                    with lock:
                        if budget["downloaded"] + size > budget["available"]:
                            raise RuntimeError("hard retrieval byte cap would be exceeded")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            response.close()
            if size != expected:
                raise RuntimeError("actual size differs from probed content length")
            temporary.replace(target)
            with lock:
                budget["downloaded"] += size
            return {**record, "status": "downloaded", "actual_bytes": size, "sha256": digest.hexdigest().upper(), "attempts": attempt}
        except Exception as error:
            temporary.unlink(missing_ok=True)
            code = getattr(getattr(error, "response", None), "status_code", None)
            transient = code in (408, 429, 500, 502, 503, 504) or type(error).__name__ in {
                "Timeout", "ConnectTimeout", "ReadTimeout", "ConnectionError"
            }
            if transient and attempt < 3:
                time.sleep(2**attempt)
                continue
            return {**record, "status": "error", "error_type": type(error).__name__, "status_code": code, "attempts": attempt}
        finally:
            if response is not None:
                response.close()
            close = getattr(client, "close", None)
            if callable(close):
                close()


def main():
    obj = json.loads(LOG.read_text(encoding="utf-8"))
    paths = sorted({m.group(1) for row in obj for m in PATH_RE.finditer(row.get("data", ""))})
    if len(paths) != 96 or any("full_pipeline_profiles/" not in p for p in paths):
        raise SystemExit(f"Expected 96 log-evidenced full-pipeline profile paths; found {len(paths)}")
    existing = sum(path.stat().st_size for path in RESULTS.rglob("*") if path.is_file())
    available = HARD_CAP - existing
    print(f"PROFILE SIZE PROBE START count={len(paths)} existing={existing}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        probes = list(pool.map(safe_request, paths))
    planned = sum(p.get("content_length") or 0 for p in probes)
    bad = [p for p in probes if p.get("status_code") != 200 or p.get("content_length") is None]
    plan = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "hard_cap_bytes": HARD_CAP,
        "existing_results_bytes": existing,
        "planned_bytes": planned,
        "available_bytes": available,
        "within_cap": planned <= available,
        "unavailable_count": len(bad),
        "files": probes,
    }
    (RUN_DIR / "v6_pipeline_profile_size_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if bad or planned > available:
        raise SystemExit("Profile size plan has unavailable files or exceeds hard cap; no download started")
    STAGING.mkdir(parents=True, exist_ok=True)
    budget = {"downloaded": 0, "available": available}
    lock = threading.Lock()
    print(f"PROFILE DOWNLOAD START planned={planned} workers={MAX_WORKERS}", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch, p, budget, lock) for p in probes]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if index % 24 == 0 or index == len(futures):
                print(f"PROFILE DOWNLOAD progress={index}/{len(futures)} bytes={budget['downloaded']}", flush=True)

    merged = []
    conflicts = []
    for record in results:
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
            old_size, old_hash = digest_file(target)
            if old_size != size or old_hash != digest:
                conflicts.append({"path": record["path"], "reason": "existing output differs; preserved both"})
                continue
            source.unlink()
            merged.append({"path": record["path"], "status": "identical_existing", "bytes": size, "sha256": digest})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            merged.append({"path": record["path"], "status": "merged_verified", "bytes": size, "sha256": digest})

    manifest = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "hard_cap_bytes": HARD_CAP,
        "existing_results_bytes_before_transfer": existing,
        "planned_transfer_bytes": planned,
        "actual_transfer_bytes": budget["downloaded"],
        "status": "complete" if len(results) == 96 and all(r.get("status") == "downloaded" for r in results) and not conflicts else "partial",
        "files": sorted(results, key=lambda x: x["path"]),
        "merged": merged,
        "conflicts": conflicts,
    }
    manifest_path = RUN_DIR / "v6_pipeline_profile_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PROFILE DOWNLOAD END status={manifest['status']} count={len(results)} bytes={budget['downloaded']} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
