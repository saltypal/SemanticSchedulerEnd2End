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
MANIFEST_PATH = RUN_DIR / "v6_pipeline_profile_download_manifest.json"
CHUNK_BYTES = 1024 * 1024


def download(path, expected):
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = OWNER
    request.kernel_slug = SLUG
    request.file_path = path
    request.version_number = VERSION
    response = None
    try:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        if response.status_code != 200:
            return {"path": path, "status": "http_error", "status_code": response.status_code}
        relative = Path(*path.split("/"))
        target = STAGING / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as out:
            for block in response.iter_content(chunk_size=CHUNK_BYTES):
                if not block:
                    continue
                size += len(block)
                if size > expected:
                    raise RuntimeError("received more bytes than size probe")
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if size != expected:
            temporary.unlink(missing_ok=True)
            return {"path": path, "status": "length_mismatch", "actual_bytes": size, "expected_bytes": expected}
        temporary.replace(target)
        return {"path": path, "status": "downloaded", "actual_bytes": size, "sha256": digest.hexdigest().upper()}
    except Exception as error:
        return {"path": path, "status": "error", "error_type": type(error).__name__, "status_code": getattr(getattr(error, "response", None), "status_code", None)}
    finally:
        if response is not None:
            response.close()
        close = getattr(client, "close", None)
        if callable(close):
            close()


def digest_file(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            h.update(block)
    return size, h.hexdigest().upper()


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["files"]}
    pending = [item for item in manifest["files"] if item.get("status") != "downloaded"]
    print(f"PROFILE RETRY START remaining={len(pending)} concurrency=1", flush=True)
    results = []
    consecutive_rate_limits = 0
    for index, item in enumerate(pending, 1):
        result = download(item["path"], item["content_length"])
        if result.get("status_code") == 429:
            consecutive_rate_limits += 1
            if consecutive_rate_limits == 1:
                time.sleep(10)
                result = download(item["path"], item["content_length"])
            if result.get("status_code") == 429:
                print(f"PROFILE RETRY HALT persistent_429 at {Path(item['path']).name}", flush=True)
                results.append(result)
                break
        else:
            consecutive_rate_limits = 0
        results.append(result)
        if result.get("status") == "downloaded":
            print(f"PROFILE RETRIED {index}/{len(pending)} {Path(item['path']).name} bytes={result['actual_bytes']}", flush=True)
        elif result.get("status") != "downloaded":
            print(f"PROFILE RETRY RESULT {index}/{len(pending)} {Path(item['path']).name} status={result.get('status')} code={result.get('status_code')}", flush=True)
            if result.get("status_code") in (401, 403):
                break

    merged = list(manifest.get("merged", []))
    conflicts = list(manifest.get("conflicts", []))
    for result in results:
        records[result["path"]] = result
        if result.get("status") != "downloaded":
            continue
        source = STAGING / Path(*result["path"].split("/"))
        size, digest = digest_file(source)
        if size != result["actual_bytes"] or digest != result["sha256"]:
            conflicts.append({"path": result["path"], "reason": "staging rehash mismatch"})
            continue
        target = RESULTS / Path(*result["path"].split("/"))
        if target.exists():
            old_size, old_digest = digest_file(target)
            if old_size != size or old_digest != digest:
                conflicts.append({"path": result["path"], "reason": "existing target differs; preserved both"})
                continue
            source.unlink()
            merged.append({"path": result["path"], "status": "identical_existing", "bytes": size, "sha256": digest})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            merged.append({"path": result["path"], "status": "merged_verified", "bytes": size, "sha256": digest})

    files = sorted(records.values(), key=lambda item: item["path"])
    manifest["files"] = files
    manifest["merged"] = merged
    manifest["conflicts"] = conflicts
    manifest["actual_transfer_bytes"] = sum(item.get("actual_bytes", 0) for item in files if item.get("status") == "downloaded")
    manifest["status"] = "complete" if len(files) == 96 and all(item.get("status") == "downloaded" for item in files) and not conflicts else "partial"
    manifest["last_retry_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PROFILE RETRY END status={manifest['status']} downloaded={sum(x.get('status')=='downloaded' for x in files)}/96 bytes={manifest['actual_transfer_bytes']}", flush=True)


if __name__ == "__main__":
    main()
