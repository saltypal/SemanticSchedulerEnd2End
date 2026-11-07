import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


ROOT = Path(__file__).resolve().parent / "recovery_inspection"
ROOT.mkdir(parents=True, exist_ok=True)
DEST = ROOT / "spi_recovery.zip"
PART = ROOT / "spi_recovery.zip.part"
EXPECTED_BYTES = 875_993_157
OWNER = "satyapaladugu"
SLUG = "betterscheduler-spi-20261027finalc"
RUN_MARKER = "medic_spi_20261028T214226Z"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


api = KaggleApi()
api.authenticate()
request = ApiListKernelSessionOutputRequest()
request.user_name = OWNER
request.kernel_slug = SLUG
request.page_size = 500
token = None
archive_item = None
while True:
    request.page_token = token
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    for item in response.files or []:
        if item.file_name == "spi_recovery.zip":
            archive_item = item
            break
    if archive_item or not response.next_page_token:
        break
    token = response.next_page_token

if archive_item is None:
    raise RuntimeError("spi_recovery.zip is absent from the latest kernel output index")
manifest_path = ROOT / "spi_recovery_manifest.json"
if not manifest_path.is_file():
    raise RuntimeError("v19 recovery manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if RUN_MARKER not in manifest.get("run", ""):
    raise RuntimeError("recovery manifest does not identify the expected v19 run")
log_path = ROOT / "betterscheduler-spi-20261027finalc.log"
if not log_path.is_file() or RUN_MARKER not in log_path.read_text(encoding="utf-8", errors="replace"):
    raise RuntimeError("saved v19 kernel log does not identify the recovery run")

started = time.monotonic()
digest = hashlib.sha256()
downloaded = 0
last_report = 0
response = requests.get(archive_item.url, stream=True, timeout=(30, 180))
response.raise_for_status()
header_length = int(response.headers.get("Content-Length", "-1"))
if header_length != EXPECTED_BYTES:
    response.close()
    raise RuntimeError(f"unexpected recovery archive size header: {header_length}")

try:
    with PART.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if not chunk:
                continue
            stream.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 64 * 1024 * 1024:
                elapsed = time.monotonic() - started
                rate = downloaded / max(elapsed, 1e-9)
                entry = {
                    "observed_utc": utc_now(),
                    "downloaded_bytes": downloaded,
                    "expected_bytes": EXPECTED_BYTES,
                    "elapsed_seconds": round(elapsed, 3),
                    "average_bytes_per_second": round(rate, 1),
                }
                with (ROOT / "download_progress.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(entry, sort_keys=True) + "\n")
                print(json.dumps(entry, sort_keys=True), flush=True)
                last_report = downloaded
        stream.flush()
        os.fsync(stream.fileno())
finally:
    response.close()

elapsed = time.monotonic() - started
if downloaded != EXPECTED_BYTES:
    raise RuntimeError(f"recovery archive byte count mismatch: {downloaded} != {EXPECTED_BYTES}")
PART.replace(DEST)
receipt = {
    "kernel": f"{OWNER}/{SLUG}/19",
    "source_run_marker": RUN_MARKER,
    "remote_output_name": archive_item.file_name,
    "expected_content_length": EXPECTED_BYTES,
    "downloaded_bytes": downloaded,
    "sha256": digest.hexdigest(),
    "started_utc": datetime.fromtimestamp(time.time() - elapsed, timezone.utc).isoformat(),
    "completed_utc": utc_now(),
    "elapsed_seconds": round(elapsed, 3),
    "average_bytes_per_second": round(downloaded / max(elapsed, 1e-9), 1),
    "local_path": str(DEST),
}
(ROOT / "recovery_download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True), flush=True)
