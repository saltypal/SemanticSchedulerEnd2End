import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "compact_outputs_v20"
MANIFEST_PATH = OUTPUT_ROOT / "spi_recovery_manifest.json"
TARGET_ROOT = OUTPUT_ROOT / "campaign" / "comparison" / "clip_b32"
OWNER = "satyapaladugu"
SLUG = "betterscheduler-spi-20261027finalc"
RUN_ID = "medic_spi_20261028T221515Z"
RUN_PREFIX = f"experiments/spi_classifier_repair/runs/{RUN_ID}/"
MAX_BYTES = 100_000_000
SUFFIXES = (
    "campaign/comparison/clip_b32/features.npz",
    "campaign/comparison/clip_b32/identity.json",
    "campaign/comparison/clip_b32/profile.json",
    "campaign/comparison/clip_b32/logistic.joblib",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
if RUN_ID not in manifest.get("run", ""):
    raise RuntimeError("recovery manifest is not for the requested v20 runtime run")
manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}
if any(suffix not in manifest_by_path for suffix in SUFFIXES):
    raise RuntimeError("one or more requested CLIP artifacts are absent from the v20 manifest")
expected_total = sum(manifest_by_path[suffix]["bytes"] for suffix in SUFFIXES)
if expected_total > MAX_BYTES:
    raise RuntimeError(f"requested artifacts exceed the 100 MB retrieval cap: {expected_total}")

TARGET_ROOT.mkdir(parents=True, exist_ok=True)
existing_verified = []
to_download = []
for suffix in SUFFIXES:
    relative = PurePosixPath(suffix)
    target = OUTPUT_ROOT.joinpath(*relative.parts)
    expected = manifest_by_path[suffix]
    if target.exists():
        actual_size = target.stat().st_size
        actual_sha = sha256_file(target)
        if actual_size != expected["bytes"] or actual_sha != expected["sha256"]:
            raise RuntimeError(f"existing file differs from v20 manifest; refusing to overwrite: {target}")
        existing_verified.append({"path": suffix, "bytes": actual_size, "sha256": actual_sha})
    else:
        to_download.append((suffix, target, expected))

api = KaggleApi()
api.authenticate()
request = ApiListKernelSessionOutputRequest()
request.user_name = OWNER
request.kernel_slug = SLUG
request.page_size = 500
files = []
token = None
log_has_run_id = False
with api.build_kaggle_client() as client:
    while True:
        request.page_token = token
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        files.extend(response.files or [])
        if response.log and RUN_ID in response.log:
            log_has_run_id = True
        token = response.next_page_token
        if not token:
            break
if not log_has_run_id:
    raise RuntimeError("latest Kaggle output log does not identify the requested v20 run")
output_by_name = {item.file_name: item for item in files}

downloaded = []
for suffix, target, expected in to_download:
    remote_name = RUN_PREFIX + suffix
    item = output_by_name.get(remote_name)
    if item is None:
        raise RuntimeError(f"exact-run output file is missing: {remote_name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    written = 0
    response = requests.get(item.url, stream=True, timeout=(30, 180))
    response.raise_for_status()
    header_size = int(response.headers.get("Content-Length", "-1"))
    if header_size != expected["bytes"]:
        response.close()
        raise RuntimeError(f"size header mismatch for {suffix}: {header_size} != {expected['bytes']}")
    try:
        with partial.open("wb") as stream:
            for chunk in response.iter_content(2 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    finally:
        response.close()
    actual_sha = digest.hexdigest()
    if written != expected["bytes"] or actual_sha != expected["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 or size mismatch for {suffix}")
    if target.exists():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"target appeared during retrieval; refusing to overwrite: {target}")
    partial.replace(target)
    downloaded.append({"path": suffix, "bytes": written, "sha256": actual_sha})

total_downloaded = sum(item["bytes"] for item in downloaded)
if total_downloaded > MAX_BYTES:
    raise RuntimeError("downloaded byte cap exceeded")
receipt = {
    "observed_utc": utc_now(),
    "kernel": f"{OWNER}/{SLUG}/20",
    "runtime_run_id": RUN_ID,
    "recovery_manifest_run": manifest["run"],
    "max_download_bytes": MAX_BYTES,
    "expected_selected_total_bytes": expected_total,
    "downloaded_bytes": total_downloaded,
    "existing_verified": existing_verified,
    "downloaded": downloaded,
    "no_other_files_requested": True,
}
(ROOT / "clip_v20_retrieval_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "run_id": RUN_ID,
    "expected_total_bytes": expected_total,
    "downloaded_bytes": total_downloaded,
    "downloaded": downloaded,
    "existing_verified": existing_verified,
}, indent=2), flush=True)
