"""Run the complete active local test setup and preserve every stage result."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "experiments/stream_scheduler/runs" / (
        "validation-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update({name: "1" for name in
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    packages = {}
    for name in ("numpy", "torch", "pytest", "lightgbm", "transformers",
                 "lm-format-enforcer", "jsonschema", "nbformat", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    snapshot = {"interpreter": sys.executable, "python": sys.version,
                "platform": platform.platform(), "packages": packages,
                "scope": "all active project test roots; archived copies and actual codec inference excluded"}
    files = [path for owner in ("Scheduler", "Evaluation", "parameters", "Test", "experiments/stream_scheduler")
             for path in (ROOT / owner).rglob("*.py")
             if not any(part in ("runs", "__pycache__", ".venv") for part in path.relative_to(ROOT / owner).parts)]
    snapshot["sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in files}
    (output / "manifest.json").write_text(json.dumps(snapshot, indent=2))
    stages = [
        ("pytest", [sys.executable, "-m", "pytest", "-q", "parameters/tests",
                    "Scheduler/tests", "Test/tests", "experiments/stream_scheduler/tests",
                    "--junitxml=" + str(output / "junit.xml")]),
        ("waiting_reference", [sys.executable, "experiments/stream_scheduler/check_waiting_spec.py", "--output", str(output / "waiting_reference")]),
    ]
    results = {}
    for name, command in stages:
        print("START", name, command, "logs", output, flush=True)
        started = time.monotonic()
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        results[name] = {"exit": result.returncode, "seconds": time.monotonic() - started,
                         "command": command, "log": str(output / (name + ".log"))}
        (output / "validation.json").write_text(json.dumps(results, indent=2))
        print("END", name, results[name], flush=True)
    drift = [name for name, before in snapshot["sources"].items()
             if not (ROOT / name).exists() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != before]
    (output / "source_drift.json").write_text(json.dumps(drift, indent=2))
    print("VALIDATION_FINISHED", output, "source_drift", len(drift), flush=True)
    return int(bool(drift) or any(row["exit"] != 0 for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
