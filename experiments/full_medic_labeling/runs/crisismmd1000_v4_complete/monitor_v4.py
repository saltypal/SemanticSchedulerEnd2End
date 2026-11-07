import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNEL = "satyapaladugu/crisismmd1000-spi-labeling/4"
SOURCE_SHA256 = "CDA913B1C1A90E933D244F353EB12C7571FACB3759ECEF6D18DE51D2537F4AEC"
POLL_SECONDS = 60
CAP_SECONDS = 24_600
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "KILLED"}
OUTPUT_PATTERN = (
    r"(^|/)(labels\.json|timings\.json|teacher_prefill_decode_profiles\.json|"
    r"study_status\.json|terminal\.json|status\.json|config\.json|"
    r"split_receipt\.json)$"
)

ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"] = "1"


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def run_cli(args, timeout=120):
    return subprocess.run(
        ["kaggle", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=ENV,
        timeout=timeout,
    )


def visible_logs():
    result = run_cli(["kernels", "logs", KERNEL], timeout=120)
    lines = [
        line
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if "outdated `kaggle` version" not in line
    ]
    return "\n".join(lines).strip(), result.returncode


def save_log_snapshot(text, observed):
    stamp = observed.strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / f"kaggle_v4_logs_{stamp}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path.name


started = time.monotonic()
started_utc = now_utc()
last_state = None
last_log_hash = None
last_logs = ""
last_log_poll = 0.0
last_run_path = None
events = ROOT / "kaggle_v4_status_history.jsonl"

while time.monotonic() - started < CAP_SECONDS:
    observed = datetime.now(timezone.utc)
    status_result = run_cli(["kernels", "status", KERNEL], timeout=120)
    status_text = (status_result.stdout + "\n" + status_result.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status_text)
    state = match.group(1) if match else "UNKNOWN"

    changed = state != last_state
    if changed:
        record = {
            "observed_utc": observed.isoformat(),
            "kernel": KERNEL,
            "status": state,
            "status_exit_code": status_result.returncode,
        }
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"[{record['observed_utc']}] status={state}", flush=True)
        last_state = state

    if changed or time.monotonic() - last_log_poll >= 120:
        last_log_poll = time.monotonic()
        try:
            log_text, log_exit = visible_logs()
        except Exception as error:
            log_text, log_exit = "", -1
            log_error_type = type(error).__name__
        else:
            log_error_type = None
        if log_text:
            digest = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
            if digest != last_log_hash:
                log_name = save_log_snapshot(log_text, observed)
                print(
                    f"[{observed.isoformat()}] new_log_bytes={len(log_text.encode('utf-8'))} "
                    f"snapshot={log_name}",
                    flush=True,
                )
                last_log_hash = digest
                last_logs = log_text
                for line in log_text.splitlines():
                    if "/kaggle/working/experiments/full_medic_labeling/runs/" in line:
                        last_run_path = line.strip()
        elif log_error_type:
            # Do not persist exception text, which can contain transport details.
            print(f"[{observed.isoformat()}] log_poll_error_type={log_error_type}", flush=True)

    if state in TERMINAL:
        break
    time.sleep(POLL_SECONDS)

elapsed = round(time.monotonic() - started, 2)
terminal_observed = datetime.now(timezone.utc)
retrieval_root = ROOT / "kaggle_v4_output"
retrieval_root.mkdir(exist_ok=True)
retrieval_attempts = []

if state in TERMINAL:
    # Final log read can expose cell output that appeared with terminal status.
    try:
        final_logs, final_log_exit = visible_logs()
    except Exception as error:
        final_logs, final_log_exit = "", -1
        final_log_error_type = type(error).__name__
    else:
        final_log_error_type = None
    if final_logs:
        digest = hashlib.sha256(final_logs.encode("utf-8")).hexdigest()
        if digest != last_log_hash:
            save_log_snapshot(final_logs, terminal_observed)
            last_logs = final_logs
            last_log_hash = digest
        for line in final_logs.splitlines():
            if "/kaggle/working/experiments/full_medic_labeling/runs/" in line:
                last_run_path = line.strip()

    for attempt in range(1, 4):
        result = run_cli(
            [
                "kernels",
                "output",
                KERNEL,
                "--path",
                str(retrieval_root),
                "--file-pattern",
                OUTPUT_PATTERN,
                "--quiet",
            ],
            timeout=300,
        )
        files = [
            path
            for path in retrieval_root.rglob("*")
            if path.is_file()
        ]
        retrieval_attempts.append(
            {
                "attempt": attempt,
                "observed_utc": now_utc(),
                "exit_code": result.returncode,
                "retrieved_file_count": len(files),
            }
        )
        if files:
            break
        if attempt < 3:
            time.sleep(60)

inventory_files = []
for path in sorted(retrieval_root.rglob("*")):
    if path.is_file():
        payload = path.read_bytes()
        inventory_files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
            }
        )

receipt = {
    "kernel": KERNEL,
    "version_number": 4,
    "source_sha256": SOURCE_SHA256,
    "status": state,
    "monitor_started_utc": started_utc,
    "observed_utc": terminal_observed.isoformat(),
    "monitor_elapsed_seconds": elapsed,
    "cap_seconds": CAP_SECONDS,
    "runtime_run_path": last_run_path,
    "log_bytes": len(last_logs.encode("utf-8")),
    "log_sha256": hashlib.sha256(last_logs.encode("utf-8")).hexdigest().upper()
    if last_logs
    else None,
    "final_log_exit_code": final_log_exit if state in TERMINAL else None,
    "final_log_error_type": final_log_error_type if state in TERMINAL else None,
    "retrieval_attempts": retrieval_attempts,
    "retrieved_files": inventory_files,
}
(ROOT / "kaggle_v4_monitor_receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print("MONITOR END " + json.dumps(receipt, separators=(",", ":")), flush=True)
