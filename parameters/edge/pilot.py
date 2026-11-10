"""Pilot run: every pipeline stage on a tiny sample, one notebook, one report (no publishing).

Each stage is a separate subprocess so a failure in one stage never hides the others. The report lists
per stage: pass/fail, wall time, the last lines of output, and file checks. It also records the
environment (GPU, free disk, library versions) because most surprises on Kaggle are environmental.

    python -m parameters.edge.pilot --work /kaggle/working/pilot --per-source 20

Besides the four pipeline stages there is one stage per pilot check (Codec/edge/pilot_checks.py): each
UNVERIFIED item from earlier reports is measured, and PILOT_REPORT.md ends with an
"unverified item -> check -> PASS/FAIL/SKIP" table built from ``<work>/checks/*.json``.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from Codec.edge import pilot_checks
from Evaluation.edge.edge_config import repo_root

META_DIR = "experiments/full_medic_labeling/runs/crisismmd4000_selection"
TAIL_LINES = 40


def environment():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "free_disk_gb": round(shutil.disk_usage("/" if Path("/kaggle").exists() else ".").free / 1e9, 1)}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=bool(torch.cuda.is_available()),
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except Exception as error:                              # report, never crash the pilot
        info["torch_error"] = repr(error)
    for lib in ("transformers", "timm", "lpips", "sklearn", "PIL"):
        try:
            module = __import__(lib)
            info[lib] = getattr(module, "__version__", "?")
        except Exception as error:
            info[lib] = "missing: " + repr(error)
    return info


def make_stages(work, per_source, meta_dir=META_DIR):
    a, b, cache = work / "dataset-a", work / "dataset-b", work / "cache"
    py = sys.executable
    stages = [
        {"name": "a_download_images",
         "argv": [py, "-m", "parameters.edge.dataset_a", "--meta-dir", meta_dir, "--out", str(a),
                  "--workers", "8", "--max-images", str(per_source), "--no-publish"],
         "expect": [a / "index.jsonl", a / "manifest.json"], "timeout": 1800},
        {"name": "b_label",
         "argv": [py, "-m", "parameters.edge.build_label_dataset", "--meta-dir", meta_dir, "--out", str(b),
                  "--images-root", str(a), "--no-publish"],
         "expect": [b / "labels_raw.jsonl", b / "counts.json"], "timeout": 600},
        {"name": "c_feature_pass_smoke",
         "argv": [py, "-m", "Codec.edge.feature_pass", "run", "--images-root", str(a), "--labels-root", str(b),
                  "--out", str(cache), "--smoke"],
         "expect": [cache / "summary.json", cache / "compute_profile.json"], "timeout": 3600},
        {"name": "d_grid_export",
         "argv": [py, "-m", "Codec.edge.feature_pass", "export-grid", "--labels-root", str(b),
                  "--out", str(cache)],
         "expect": [], "timeout": 600},
    ]
    return _with_checks(stages, work, py)


CHECK_TIMEOUT = {"embedders": 3600, "finetune_pilot": 2400, "finetune_source": 1200, "publish_roundtrip": 1800,
                 "sqgan_latency": 1800, "compute_profile_real": 2400}


def check_stage(name, work, py):
    return {"name": "check_" + name, "check": name,
            "argv": [py, "-m", "Codec.edge.pilot_checks", name, "--out", str(work)],
            "expect": [work / "checks" / (name + ".json")], "timeout": CHECK_TIMEOUT.get(name, 900)}


def _with_checks(stages, work, py):
    """env_gpu first (environment facts), the data stages, then every other check (they read stage outputs)."""
    order = pilot_checks.CHECK_ORDER
    return [check_stage(order[0], work, py)] + stages + [check_stage(n, work, py) for n in order[1:]]


def child_peak_rss_mb():
    """Largest RSS among finished child processes so far (resource is POSIX-only; None elsewhere)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return round(kb / 1024.0, 1) if sys.platform != "darwin" else round(kb / 1048576.0, 1)
    except Exception:
        return None


def tail(text, n=TAIL_LINES):
    return "\n".join((text or "").splitlines()[-n:])


def run_stage(stage, runner=subprocess.run, cwd=None):
    start = time.time()
    try:
        done = runner(stage["argv"], capture_output=True, text=True, cwd=cwd, timeout=stage.get("timeout"))
        rc, out, err = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as error:
        rc, out, err = -9, error.stdout or "", "TIMEOUT after %ss" % stage.get("timeout")
    except Exception as error:                              # e.g. missing interpreter
        rc, out, err = -1, "", repr(error)
    missing = [str(p) for p in stage.get("expect", []) if not Path(p).exists()]
    return {"name": stage["name"], "argv": " ".join(map(str, stage["argv"])), "returncode": rc,
            "passed": rc == 0 and not missing, "seconds": round(time.time() - start, 1),
            "missing_outputs": missing, "peak_rss_mb": child_peak_rss_mb(), "stdout_tail": tail(out), "stderr_tail": tail(err)}


def read_checks(work):
    out = {}
    for f in sorted((Path(work) / "checks").glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            out[f.stem] = {"name": f.stem, "status": "fail", "error": "unreadable json", "items": {}}
    return out


def item_row(item_id, checks):
    """(status, one-line detail) of one unverified item from the check JSONs."""
    desc, source, name = pilot_checks.UNVERIFIED_ITEMS[item_id]
    res = checks.get(name)
    if res is None:
        return "FAIL", "check %s produced no result (stage crashed or did not run)" % name
    it = (res.get("items") or {}).get(item_id)
    status = (it or {}).get("status", res.get("status", "fail"))
    err = (res.get("error") or "").strip().splitlines()
    detail = (it or {}).get("detail") or (err[-1] if err else "")
    return status.upper(), " ".join(str(detail).replace("|", "/").split())[:160]


def items_table(checks):
    rows = ["", "## Unverified items", "", "| item | check | result | detail |", "|---|---|---|---|"]
    for item_id, (desc, source, name) in pilot_checks.UNVERIFIED_ITEMS.items():
        status, detail = item_row(item_id, checks)
        rows.append("| %s | %s | %s | %s |" % (item_id, name, status, detail))
    counts = {}
    for item_id in pilot_checks.UNVERIFIED_ITEMS:
        st = item_row(item_id, checks)[0]
        counts[st] = counts.get(st, 0) + 1
    return rows + ["", "Totals: " + ", ".join("%s=%d" % kv for kv in sorted(counts.items()))]


def markdown(report):
    rows = ["# Pilot report", "", "| stage | result | seconds | missing outputs |", "|---|---|---:|---|"]
    for s in report["stages"]:
        rows.append("| %s | %s | %s | %s |" % (s["name"], "PASS" if s["passed"] else "FAIL", s["seconds"],
                                              ", ".join(Path(m).name for m in s["missing_outputs"]) or "-"))
    rows += items_table(report.get("checks", {}))
    rows += ["", "## Environment", "```", json.dumps(report["environment"], indent=2), "```"]
    for s in report["stages"]:
        if not s["passed"]:
            rows += ["", "## %s (failed)" % s["name"], "```", s["stderr_tail"] or s["stdout_tail"], "```"]
    return "\n".join(rows) + "\n"


def run_pilot(work, per_source, runner=subprocess.run, stages=None, only=None, cwd=None):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    chosen = stages if stages is not None else make_stages(work, per_source)
    if only:
        chosen = [s for s in chosen if s["name"] in only]
    report = {"per_source": per_source, "environment": environment(), "stages": []}
    for stage in chosen:
        result = run_stage(stage, runner, cwd or str(repo_root()))
        if stage.get("check") and not result["missing_outputs"]:
            status = json.loads(Path(stage["expect"][0]).read_text(encoding="utf-8")).get("status")
            result["passed"] = result["returncode"] == 0 and status != "fail"
            result["check_status"] = status
        report["stages"].append(result)
        (work / "stage_results.json").write_text(json.dumps(report["stages"], indent=2), encoding="utf-8")
        print("[pilot] %-24s %s (%.0fs)" % (result["name"], "PASS" if result["passed"] else "FAIL",
                                             result["seconds"]), flush=True)
    report["checks"] = read_checks(work)
    report["all_passed"] = all(s["passed"] for s in report["stages"])
    (work / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (work / "PILOT_REPORT.md").write_text(markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/kaggle/working/pilot")
    ap.add_argument("--per-source", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None, help="stage names to run")
    args = ap.parse_args(argv)
    report = run_pilot(args.work, args.per_source, only=args.only)
    print(markdown(report))


if __name__ == "__main__":
    main()
