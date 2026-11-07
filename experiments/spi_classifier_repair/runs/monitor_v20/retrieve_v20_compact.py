import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "compact_outputs_v20"
DEST.mkdir(parents=True, exist_ok=True)
OWNER = "satyapaladugu"
SLUG = "betterscheduler-spi-20261027finalc"
KERNEL = f"{OWNER}/{SLUG}/20"
RUN_ID = "medic_spi_20261028T221515Z"
RUN_PREFIX = f"experiments/spi_classifier_repair/runs/{RUN_ID}/"
SOURCE_SHA256 = "6c5758f5a9da0b4b433e1ae6eda07557187a84b885bd493d154fadd874b0b364"
MAX_SELECTED_MODEL_BYTES = 250_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


api = KaggleApi()
api.authenticate()
status_object = api.kernels_status(KERNEL)
status = {
    "status": status_object.status.name,
    "failure_message": status_object.failure_message,
}
if status.get("status") != "COMPLETE":
    raise RuntimeError(f"v20 is not terminal COMPLETE: {status}")

log_text = api.kernels_logs(KERNEL)
(ROOT / "logs_raw_v20.json").write_text(log_text, encoding="utf-8")
events = json.loads(log_text)
stdout_parts = []
stderr_parts = []
for event in events:
    stream = stdout_parts if event.get("stream_name") == "stdout" else stderr_parts
    stream.append(f"[{event.get('time', '')}] {event.get('data', '')}")
(ROOT / "stdout_v20.log").write_text("".join(stdout_parts), encoding="utf-8")
(ROOT / "stderr_v20.log").write_text("".join(stderr_parts), encoding="utf-8")
all_log_text = "".join(event.get("data", "") for event in events)
if SOURCE_SHA256 not in all_log_text or RUN_ID not in all_log_text:
    raise RuntimeError("latest kernel logs do not match the v20 source/run identity")

request = ApiListKernelSessionOutputRequest()
request.user_name = OWNER
request.kernel_slug = SLUG
request.page_size = 500
files = []
token = None
latest_log = ""
with api.build_kaggle_client() as client:
    while True:
        request.page_token = token
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        files.extend(response.files or [])
        if response.log:
            latest_log = response.log
        token = response.next_page_token
        if not token:
            break

if RUN_ID not in latest_log:
    raise RuntimeError("current kernel output log is not tied to the v20 runtime run")

output_by_name = {item.file_name: item for item in files}
manifest_item = output_by_name.get("spi_recovery_manifest.json")
if manifest_item is None:
    raise RuntimeError("v20 recovery manifest is not present in the output listing")
manifest_path = DEST / "spi_recovery_manifest.json"
response = requests.get(manifest_item.url, stream=True, timeout=(30, 120))
response.raise_for_status()
with manifest_path.open("wb") as stream:
    for chunk in response.iter_content(2 * 1024 * 1024):
        if chunk:
            stream.write(chunk)
response.close()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if RUN_ID not in manifest.get("run", ""):
    raise RuntimeError("v20 recovery manifest belongs to a different run")
manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}

suffixes = {
    "campaign/terminal.json",
    "campaign/REPORT.md",
    "campaign/run_identity.json",
    "input_receipt.json",
    "environment.json",
    "campaign/labeling/labels.json",
    "campaign/labeling/teacher_identity.json",
    "campaign/labeling/timings.json",
    "campaign/splits.json",
    "campaign/records.json",
    "campaign/comparison/candidates.json",
    "campaign/comparison/coverage.json",
    "campaign/comparison/failures.json",
    "campaign/comparison/selection_frozen.json",
    "campaign/comparison/locked_test_status.json",
    "campaign/release/readiness.json",
    "campaign/release/development.json",
    "campaign/release/locked_test.json",
    "campaign/release/profiles.json",
    "campaign/release/calibration.json",
    "campaign/release/bundle/readiness.json",
    "campaign/release/bundle/bundle.json",
    "campaign/release/bundle/encoder_identity.json",
    "campaign/release/bundle/selection.json",
    "campaign/release/bundle/model.joblib",
    "campaign/final_parameter_cache/status.json",
    "campaign/development_comparison.png",
}
for encoder in ("clip_b32", "convnextv2_tiny", "dino", "siglip2_b16"):
    suffixes.add(f"campaign/comparison/{encoder}/identity.json")
    suffixes.add(f"campaign/comparison/{encoder}/profile.json")
for encoder in ("dino", "siglip2_b16"):
    for head in ("gated_abmil", "mean_mlp", "single_query_cross_attention"):
        suffixes.add(f"campaign/comparison/{encoder}/{head}_curves.json")
        suffixes.add(f"campaign/comparison/{encoder}/{head}_curves.png")
for item in manifest["files"]:
    path = item["path"]
    if path.startswith("campaign/release/checkpoints/profile/") and path.endswith(("/profile.json", "/receipt.json")):
        suffixes.add(path)

selected_model = "campaign/release/bundle/model.joblib"
model_entry = manifest_by_path.get(selected_model)
if model_entry and model_entry["bytes"] > MAX_SELECTED_MODEL_BYTES:
    suffixes.remove(selected_model)

downloads = []
missing = []
tasks = []
for suffix in sorted(suffixes):
    remote_name = RUN_PREFIX + suffix
    item = output_by_name.get(remote_name)
    expected = manifest_by_path.get(suffix)
    if item is None or expected is None:
        missing.append(suffix)
        continue
    relative = PurePosixPath(suffix)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe output path: {suffix}")
    tasks.append((suffix, item, expected, DEST.joinpath(*relative.parts)))


def download_one(task):
    suffix, item, expected, target = task
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    response = requests.get(item.url, stream=True, timeout=(30, 180))
    response.raise_for_status()
    header_size = int(response.headers.get("Content-Length", "-1"))
    if header_size != expected["bytes"]:
        response.close()
        raise RuntimeError(f"unexpected content length for {suffix}: {header_size} != {expected['bytes']}")
    with target.open("wb") as stream:
        for chunk in response.iter_content(2 * 1024 * 1024):
            if chunk:
                stream.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    response.close()
    actual_sha = digest.hexdigest()
    if written != expected["bytes"] or actual_sha != expected["sha256"]:
        raise RuntimeError(f"v20 artifact hash mismatch for {suffix}")
    return {"path": suffix, "bytes": written, "sha256": actual_sha}


with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(download_one, task) for task in tasks]
    for future in as_completed(futures):
        downloads.append(future.result())

summary = {
    "observed_utc": utc_now(),
    "kernel": KERNEL,
    "status": status,
    "runtime_run_id": RUN_ID,
    "source_sha256": SOURCE_SHA256,
    "notebook_log_sha256": hashlib.sha256(log_text.encode("utf-8")).hexdigest(),
    "notebook_log_event_count": len(events),
    "runtime_seconds_last_log": max((float(e.get("time", 0)) for e in events), default=0),
    "output_file_count_for_run": sum(name.startswith(RUN_PREFIX) for name in output_by_name),
    "recovery_manifest_sha256": sha256_file(manifest_path),
    "recovery_member_count": len(manifest["files"]),
    "recovery_uncompressed_bytes": sum(entry["bytes"] for entry in manifest["files"]),
    "recovery_archive_name": next((name for name in output_by_name if name == "spi_recovery.zip"), None),
    "downloaded_artifacts": downloads,
    "missing_artifacts": missing,
    "selected_bundle_model_skipped_if_over_limit": bool(model_entry and model_entry["bytes"] > MAX_SELECTED_MODEL_BYTES),
}
(ROOT / "artifact_manifest_v20.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "status": status["status"],
    "run_id": RUN_ID,
    "runtime_seconds": summary["runtime_seconds_last_log"],
    "output_files": summary["output_file_count_for_run"],
    "recovery_members": summary["recovery_member_count"],
    "recovery_uncompressed_bytes": summary["recovery_uncompressed_bytes"],
    "downloaded_count": len(downloads),
    "downloaded_bytes": sum(item["bytes"] for item in downloads),
    "missing": missing,
    "selected_model_bytes": None if model_entry is None else model_entry["bytes"],
}, indent=2), flush=True)
