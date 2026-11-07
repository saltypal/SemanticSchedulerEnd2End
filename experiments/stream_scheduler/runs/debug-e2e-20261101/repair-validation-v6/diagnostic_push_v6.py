import hashlib
import json
import pathlib
import requests
import sys
import time

root = pathlib.Path(__file__).resolve().parents[5]
run = pathlib.Path(__file__).resolve().parent
notebook = root / "experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb"
expected = "d74699398f4eb77d8e56ada978958be3df72435f8c7c4583bdc8fc555c32742b"
actual = hashlib.sha256(notebook.read_bytes()).hexdigest()
if actual != expected or notebook.stat().st_size >= 1_000_000:
    raise SystemExit("v6 source changed or exceeds Kaggle's source limit; not submitted")

record = {
    "attempt": "authorized_v6_technical_followup",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notebook_sha256": actual,
    "notebook_bytes": notebook.stat().st_size,
    "timeout_seconds": 900,
    "response": None,
}
original_send = requests.sessions.Session.send


def send(self, request, **kwargs):
    response = original_send(self, request, **kwargs)
    if "kernels.KernelsApiService/SaveKernel" in request.url:
        record["response"] = {
            "status_code": response.status_code,
            "body": response.text[:12000],
        }
        (run / "submission_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return response


requests.sessions.Session.send = send
from kaggle.cli import main

sys.argv = [
    "kaggle",
    "kernels",
    "push",
    "-p",
    "experiments/stream_scheduler/kaggle",
    "-t",
    "900",
    "--accelerator",
    "NvidiaTeslaT4",
]
try:
    code = main()
except BaseException as exc:
    record["exception"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    code = 1
record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
record["cli_exit_code"] = code if isinstance(code, int) else 0
(run / "submission_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
print("V6_SUBMISSION", json.dumps(record, sort_keys=True))
raise SystemExit(code if isinstance(code, int) else 0)
