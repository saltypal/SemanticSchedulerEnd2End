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
RUNTIME_PREFIX = "experiments/full_medic_labeling/runs/main_20261029T154606Z"
RESULTS = Path(r"D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\experiments\full_medic_labeling\runs\v6_results")
RUN_DIR = Path(__file__).resolve().parent
STAGING = RUN_DIR / "staging"
MAX_WORKERS = 4
CHUNK_BYTES = 1024 * 1024
BALANCES = ("original", "class_weighted", "random_oversampling", "moderate_undersampling")
NAMES = (
    "candidates.json",
    "failures.json",
    "coverage.json",
    "selection_frozen.json",
    "locked_test_status.json",
    "sklearn_floppy_estimates.jsonl",
)
FILES = [f"{RUNTIME_PREFIX}/comparison/{balance}/{name}" for balance in BALANCES for name in NAMES]


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


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest().upper()


def retrieve(path):
    client = None
    relative = Path(*path.split("/"))
    local = STAGING / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        client, response = client_response(path)
        if response.status_code != 200:
            code = response.status_code
            response.close()
            return {"path": path, "status": "http_error", "status_code": code}
        expected_text = response.headers.get("Content-Length")
        expected = int(expected_text) if expected_text and expected_text.isdigit() else None
        temporary = local.with_name(local.name + ".part")
        digest = hashlib.sha256()
        actual = 0
        with temporary.open("wb") as out:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                actual += len(chunk)
                if expected is not None and actual > expected:
                    raise RuntimeError("actual bytes exceeded response content length")
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        response.close()
        if expected is None or actual != expected:
            temporary.unlink(missing_ok=True)
            return {"path": path, "status": "length_unverified", "advertised_bytes": expected, "actual_bytes": actual}
        temporary.replace(local)
        return {
            "path": path,
            "status": "downloaded",
            "advertised_bytes": expected,
            "actual_bytes": actual,
            "sha256": digest.hexdigest().upper(),
        }
    except Exception as error:
        return {"path": path, "status": "error", "error_type": type(error).__name__}
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"METRICS DOWNLOAD START files={len(FILES)} workers={MAX_WORKERS}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        records = list(pool.map(retrieve, FILES))
    records.sort(key=lambda item: item["path"])
    actual = sum(item.get("actual_bytes", 0) for item in records)
    manifest = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "actual_transfer_bytes": actual,
        "files": records,
    }
    manifest_path = RUN_DIR / "v6_metrics_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    merged = []
    conflicts = []
    for record in records:
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
                conflicts.append({"path": record["path"], "reason": "existing target differs; both preserved"})
                continue
            source.unlink()
            merged.append({"path": record["path"], "status": "identical_existing"})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            merged.append({"path": record["path"], "status": "merged_verified", "bytes": size, "sha256": digest})

    manifest["merged"] = merged
    manifest["conflicts"] = conflicts
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"METRICS DOWNLOAD END bytes={actual} downloaded={len(merged)} conflicts={len(conflicts)} manifest={manifest_path}",
        flush=True,
    )

    artifact_paths = set()
    for balance in BALANCES:
        candidate_path = RESULTS / RUNTIME_PREFIX / "comparison" / balance / "candidates.json"
        if not candidate_path.is_file():
            continue
        try:
            candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for candidate in candidates:
            artifact = candidate.get("artifact")
            if isinstance(artifact, str) and artifact.startswith("/kaggle/working/"):
                artifact_paths.add(artifact.removeprefix("/kaggle/working/").replace("\\", "/"))
    (RUN_DIR / "v6_candidate_artifact_paths.json").write_text(
        json.dumps(sorted(artifact_paths), indent=2) + "\n", encoding="utf-8"
    )
    print(f"CANDIDATE_ARTIFACT_PATHS count={len(artifact_paths)}", flush=True)


if __name__ == "__main__":
    main()
