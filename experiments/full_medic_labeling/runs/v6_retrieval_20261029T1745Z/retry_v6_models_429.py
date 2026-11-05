import concurrent.futures
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


OWNER = "satyapaladugu"
SLUG = "crisismmd1000-spi-labeling"
VERSION = 6
PROJECT = Path(r"D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End")
RESULTS = PROJECT / "experiments/full_medic_labeling/runs/v6_results"
RUN_DIR = Path(__file__).resolve().parent
STAGING = RUN_DIR / "staging"
MANIFEST_PATH = RUN_DIR / "v6_models_download_manifest.json"
CHUNK_BYTES = 1024 * 1024


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest().upper()


def fetch(record):
    remote_path = record["path"]
    relative = Path(*remote_path.split("/"))
    target = STAGING / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    for attempt in range(1, 3):
        api = KaggleApi()
        api.authenticate()
        client = api.build_kaggle_client()
        response = None
        try:
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug = OWNER
            request.kernel_slug = SLUG
            request.file_path = remote_path
            request.version_number = VERSION
            response = client.kernels.kernels_api_client.download_kernel_output(request)
            if response.status_code != 200:
                code = response.status_code
                response.close()
                if code in (408, 429, 500, 502, 503, 504) and attempt == 1:
                    time.sleep(15)
                    continue
                return {**record, "status": "http_error", "status_code": code, "attempts": attempt}
            expected = record["content_length"]
            digest = hashlib.sha256()
            actual = 0
            with temporary.open("wb") as out:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    actual += len(chunk)
                    if actual > expected:
                        raise RuntimeError("response exceeded probed content length")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            response.close()
            if actual != expected:
                raise RuntimeError("download length mismatch")
            temporary.replace(target)
            result = {
                **record,
                "status": "downloaded",
                "attempts": attempt,
                "actual_bytes": actual,
                "sha256": digest.hexdigest().upper(),
            }
            result.pop("status_code", None)
            return result
        except Exception as error:
            temporary.unlink(missing_ok=True)
            status = getattr(getattr(error, "response", None), "status_code", None)
            transient = status in (408, 429, 500, 502, 503, 504) or type(error).__name__ in {
                "Timeout",
                "ConnectTimeout",
                "ReadTimeout",
                "ConnectionError",
            }
            if transient and attempt == 1:
                time.sleep(15)
                continue
            return {**record, "status": "error", "attempts": attempt, "error_type": type(error).__name__, "status_code": status}
        finally:
            if response is not None:
                response.close()
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return {**record, "status": "retry_exhausted"}


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    retry = [item for item in manifest["files"] if item.get("status_code") == 429]
    if not retry:
        print("NO_RATE_LIMITED_MODEL_FILES", flush=True)
        return
    print(f"RATE_LIMIT_RETRY START count={len(retry)} workers=1", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for index, result in enumerate(pool.map(fetch, retry), 1):
            results.append(result)
            print(
                f"RATE_LIMIT_RETRY {index}/{len(retry)} path={Path(result['path']).name} "
                f"status={result.get('status')} code={result.get('status_code')} bytes={result.get('actual_bytes')}",
                flush=True,
            )

    old_by_path = {item["path"]: item for item in manifest["files"]}
    for result in results:
        old_by_path[result["path"]] = result
    merged = list(manifest.get("merged", []))
    conflicts = list(manifest.get("conflicts", []))
    for result in results:
        if result.get("status") != "downloaded":
            continue
        relative = Path(*result["path"].split("/"))
        source = STAGING / relative
        size, digest = digest_file(source)
        if size != result["actual_bytes"] or digest != result["sha256"]:
            conflicts.append({"path": result["path"], "reason": "retry staging rehash mismatch"})
            continue
        target = RESULTS / relative
        if target.exists():
            old_size, old_hash = digest_file(target)
            if old_size != size or old_hash != digest:
                conflicts.append({"path": result["path"], "reason": "existing target differs; both preserved"})
                continue
            source.unlink()
            merged.append({"path": result["path"], "status": "identical_existing", "bytes": size, "sha256": digest})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            merged.append({"path": result["path"], "status": "merged_verified", "bytes": size, "sha256": digest})

    all_files = sorted(old_by_path.values(), key=lambda item: item["path"])
    manifest["files"] = all_files
    manifest["actual_transfer_bytes"] = sum(item.get("actual_bytes", 0) for item in all_files if item.get("status") == "downloaded")
    manifest["completed_models"] = sum(item.get("status") in ("downloaded", "identical_existing_verified") for item in all_files)
    manifest["status"] = "complete" if manifest["completed_models"] == manifest["candidate_model_count"] and not conflicts else "partial"
    manifest["merged"] = merged
    manifest["conflicts"] = conflicts
    manifest["retry_completed_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"RATE_LIMIT_RETRY END status={manifest['status']} complete={manifest['completed_models']}/{manifest['candidate_model_count']}", flush=True)


if __name__ == "__main__":
    main()
