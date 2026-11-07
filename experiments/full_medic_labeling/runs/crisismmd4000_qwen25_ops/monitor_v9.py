import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNEL = "satyapaladugu/crisismmd1000-spi-labeling/9"
SOURCE_SHA256 = "C44C2D075B23900F784C6B1CDD363CA35E57448EA78A1213819C281AFA704C4D"
POLL_SECONDS = 60
CAP_SECONDS = 24_600
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED", "KILLED"}
PROGRESS_PATTERN = r"(^|/)(identity\.json|progress\.json)$"

ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"] = "1"


def run_cli(args, timeout=180):
    return subprocess.run(
        ["kaggle", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=ENV,
        timeout=timeout,
    )


def current_status():
    result = run_cli(["kernels", "status", KERNEL], timeout=120)
    text = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", text)
    return match.group(1) if match else "UNKNOWN", result.returncode


def current_logs():
    result = run_cli(["kernels", "logs", KERNEL], timeout=180)
    lines = [
        line
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if "outdated `kaggle` version" not in line
    ]
    return "\n".join(lines).strip(), result.returncode


def save_log(text, observed):
    stamp = observed.strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / f"kaggle_v9_logs_{stamp}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path.name


started = time.monotonic()
started_utc = datetime.now(timezone.utc).isoformat()
history_path = ROOT / "kaggle_v9_status_history.jsonl"
last_status = None
last_log_hash = None
latest_log = ""
last_log_poll = 0.0
runtime_path = None
status = "UNKNOWN"

while time.monotonic() - started < CAP_SECONDS:
    observed = datetime.now(timezone.utc)
    status, exit_code = current_status()

    status_changed = status != last_status
    if status_changed:
        record = {
            "observed_utc": observed.isoformat(),
            "kernel": KERNEL,
            "status": status,
            "status_exit_code": exit_code,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"[{record['observed_utc']}] status={status}", flush=True)
        last_status = status

    if status_changed or time.monotonic() - last_log_poll >= 120:
        # Status transitions get log reads immediately; otherwise cap to 120 s.
        last_log_poll = time.monotonic()
        try:
            log_text, log_exit = current_logs()
        except Exception as error:
            log_text, log_exit = "", -1
            error_type = type(error).__name__
        else:
            error_type = None
        if log_text:
            digest = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
            if digest != last_log_hash:
                name = save_log(log_text, observed)
                print(
                    f"[{observed.isoformat()}] new_log_bytes={len(log_text.encode('utf-8'))} snapshot={name}",
                    flush=True,
                )
                latest_log = log_text
                last_log_hash = digest
                for line in log_text.splitlines():
                    if "/kaggle/working/experiments/full_medic_labeling/runs/" in line:
                        runtime_path = line.strip()
        elif error_type:
            print(f"[{observed.isoformat()}] log_poll_error_type={error_type}", flush=True)

    if status in TERMINAL:
        break
    time.sleep(POLL_SECONDS)

elapsed = round(time.monotonic() - started, 2)
ended = datetime.now(timezone.utc)
retrieval_root = ROOT / "kaggle_v9_progress_identity"
retrieval_root.mkdir(exist_ok=True)
retrieval = []
terminal_log_exit = None
terminal_log_error = None

if status in TERMINAL:
    try:
        final_log, terminal_log_exit = current_logs()
    except Exception as error:
        final_log, terminal_log_exit = "", -1
        terminal_log_error = type(error).__name__
    if final_log:
        digest = hashlib.sha256(final_log.encode("utf-8")).hexdigest()
        if digest != last_log_hash:
            save_log(final_log, ended)
            latest_log = final_log
        for line in final_log.splitlines():
            if "/kaggle/working/experiments/full_medic_labeling/runs/" in line:
                runtime_path = line.strip()

    # Only fetch minimal identity and progress receipts. Never request labels,
    # SQLite, images, embeddings, checkpoints, or model weights.
    result = run_cli(
        [
            "kernels",
            "output",
            KERNEL,
            "--path",
            str(retrieval_root),
            "--file-pattern",
            PROGRESS_PATTERN,
            "--quiet",
        ],
        timeout=300,
    )
    for path in sorted(retrieval_root.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            record = {
                "path": str(path.relative_to(ROOT)),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
            }
            if path.name == "progress.json":
                try:
                    record["content"] = json.loads(payload)
                except json.JSONDecodeError:
                    record["parse_error"] = True
            retrieval.append(record)
    retrieval_status = result.returncode
else:
    retrieval_status = None

receipt = {
    "kernel": KERNEL,
    "version_number": 9,
    "source_sha256": SOURCE_SHA256,
    "status": status,
    "monitor_started_utc": started_utc,
    "observed_utc": ended.isoformat(),
    "monitor_elapsed_seconds": elapsed,
    "cap_seconds": CAP_SECONDS,
    "runtime_path": runtime_path,
    "log_bytes": len(latest_log.encode("utf-8")),
    "log_sha256": hashlib.sha256(latest_log.encode("utf-8")).hexdigest().upper()
    if latest_log
    else None,
    "terminal_log_exit_code": terminal_log_exit,
    "terminal_log_error_type": terminal_log_error,
    "progress_identity_retrieval_exit_code": retrieval_status,
    "retrieved_progress_identity": retrieval,
}
(ROOT / "kaggle_v9_monitor_receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print("MONITOR END " + json.dumps(receipt, separators=(",", ":")), flush=True)
