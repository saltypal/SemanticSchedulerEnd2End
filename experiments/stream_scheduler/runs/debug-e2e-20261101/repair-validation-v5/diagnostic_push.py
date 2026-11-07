import json
import pathlib
import requests
import sys
import time

run = pathlib.Path(__file__).resolve().parent
record = {
    "attempt": "single_authorized_diagnostic_resubmission",
    "purpose": "capture Kaggle SaveKernel validation response",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "max_timeout_seconds": 1200,
    "same_source_and_metadata": True,
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
        (run / "diagnostic_push_response.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
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
    "1200",
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
(run / "diagnostic_push_response.json").write_text(json.dumps(record, indent=2) + "\n")
print("DIAGNOSTIC_PUSH_RESULT", json.dumps(record, sort_keys=True))
raise SystemExit(code if isinstance(code, int) else 0)
