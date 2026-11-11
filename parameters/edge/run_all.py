"""One-notebook orchestrator: dataset A (images), B (labels), C (cache + classifier grid) in a single run.

    python -m parameters.edge.run_all --mode pilot|full --work /kaggle/working/run [--force STAGE ...]

Every stage is a SEPARATE SUBPROCESS. A failed stage is recorded, stages that depend on it are marked
SKIP, independent stages still run. Stages write ``<work>/state/<stage>.done.json`` and are skipped on
re-run unless ``--force <stage>``. Full child output goes to ``<work>/logs/<stage>.log``; the console
(the Kaggle web log) gets a banner per stage, a few filtered lines and a RUN SUMMARY table.

Order: env, A, B, C_features, C_grid, [checks (pilot only)], publish. ``pilot`` uses --max-images 20,
--smoke and the unverified-item checks of ``parameters.edge.pilot`` (PILOT_REPORT.md) and never publishes;
``full`` runs full sizes and enforces the B gates (test/val Critical counts) before C.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from Evaluation.edge.edge_config import repo_root
from parameters.edge import pilot

MAX_CHILD_LINES = 15
TAIL_ON_FAIL = 25
KEYWORDS = ("done", "error", "warning", "shard", "images", "fail", "pass", "wrote", "saved", "gate", "skip",
            "published", "created", "versioned", "traceback", "exception")
_PROGRESS = re.compile(r"(\d+(\.\d+)?\s*%)|(\d+/\d+\s*\[)|(it/s)|(s/it)|(\bETA\b)|(\|[#█▏▎▍▌▋▊▉ ]{3,}\|)", re.I)

# stage name -> (stages whose success it needs)
DEPS = {"env": [], "A": [], "B": ["A"], "C_features": ["A", "B"], "C_grid": ["C_features"], "checks": [],
        "publish": ["A", "B"]}


class LineFilter:
    """Console filter for child output: drops progress spam, repeats and non-key lines; caps the line count."""

    def __init__(self, limit=MAX_CHILD_LINES, keep_all=False, keywords=KEYWORDS):
        self.limit, self.keep_all, self.keywords = limit, keep_all, tuple(k.lower() for k in keywords)
        self.seen, self.printed, self.dropped = set(), 0, 0

    def accept(self, line):
        """Return the line to print, or None."""
        text = line.rstrip("\r\n").split("\r")[-1].strip()
        if not text or _PROGRESS.search(text) or text in self.seen:
            return None
        self.seen.add(text)
        if not self.keep_all and not any(k in text.lower() for k in self.keywords):
            return None
        if self.printed >= self.limit:
            self.dropped += 1
            return None
        self.printed += 1
        return text[:300]


def default_runner(argv, cwd, env, timeout, emit):
    """Run argv, stream merged stdout/stderr lines to ``emit``; return the exit code (-9 on timeout)."""
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", bufsize=1)
    timer = threading.Timer(timeout, proc.kill) if timeout else None
    if timer:
        timer.start()
    try:
        for line in proc.stdout:
            emit(line)
        rc = proc.wait()
    finally:
        if timer:
            timer.cancel()
    if timer and not timer.is_alive() and rc < 0:
        emit("ERROR timeout after %ss\n" % timeout)
        return -9
    return rc


def dir_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def _gate_error(counts_file, enforce):
    """B's gates (test/val Critical counts). Returns a message when they fail and are enforced."""
    try:
        gates = json.loads(Path(counts_file).read_text(encoding="utf-8"))["gates"]
    except (OSError, ValueError, KeyError) as error:
        return "cannot read gates from %s (%r)" % (counts_file, error)
    bad = [k for k in ("test_critical_ok", "val_critical_ok") if not gates.get(k)]
    if not bad:
        return None
    why = "B gate failed: " + ", ".join("%s (test=%s, val=%s)" % (k, gates.get("test_critical"),
                                                                  gates.get("val_critical")) for k in bad)
    return why if enforce else None


ENV_CODE = ("import json,sys;from pathlib import Path;from parameters.edge.pilot import environment;"
            "from Codec.edge.pilot_checks import input_tree;e=environment();"
            "Path(sys.argv[1]).mkdir(parents=True,exist_ok=True);"
            "Path(sys.argv[1],'environment.json').write_text(json.dumps(e,indent=1));print('environment:',json.dumps(e));"
            "print('input tree:');[print(' ',x) for x in input_tree('/kaggle/input',3,60)]")
PUBLISH_CODE = ("import json,sys;from parameters.edge import kaggle_publish as k, dataset_a as a, build_label_dataset as b;"
                "m={'A':(a.SLUG,a.TITLE,'stage A images'),'B':(b.SLUG,b.TITLE,'stage B labels')}[sys.argv[1]];"
                "print('publish',sys.argv[1],json.dumps(k.publish_dataset(sys.argv[2],m[0],m[1],m[2])))")


def build_stages(mode, work, meta_dir=None, py=None):
    """Ordered stage dicts: name, argvs (list of argv lists), timeout, expect, keep_all, keep_going, post."""
    work, py = Path(work), py or sys.executable
    meta_dir = str(meta_dir or (repo_root() / pilot.META_DIR))
    pil = mode == "pilot"
    a, b, cache = work / "dataset-a", work / "dataset-b", work / "cache"
    grid = cache / "grid"
    stages = [
        {"name": "env", "argvs": [[py, "-c", ENV_CODE, str(work)]], "timeout": 300, "keep_all": True},
        {"name": "A", "timeout": 6 * 3600, "expect": [a / "index.jsonl", a / "manifest.json"],
         "argvs": [[py, "-m", "parameters.edge.dataset_a", "--meta-dir", meta_dir, "--out", str(a),
                    "--workers", "8" if pil else "16", "--no-publish"] + (["--max-images", "20"] if pil else [])]},
        {"name": "B", "timeout": 1800, "expect": [b / "labels_raw.jsonl", b / "counts.json"],
         "argvs": [[py, "-m", "parameters.edge.build_label_dataset", "--meta-dir", meta_dir, "--out", str(b),
                    "--images-root", str(a), "--no-publish"]],
         "post": lambda: _gate_error(b / "counts.json", enforce=not pil)},
        {"name": "C_features", "timeout": 8 * 3600, "expect": [cache / "summary.json"],
         "parallel": lambda n_gpu: _feature_parts(py, a, b, cache, pil, n_gpu),
         "argvs": [[py, "-c", MERGE_CODE, str(cache)]]},
        {"name": "C_grid", "timeout": 4 * 3600, "argvs": [[py, "-m", "Codec.edge.feature_pass", "export-grid", "--labels-root", str(b), "--out", str(cache)],
                   [py, "-m", "parameters.edge.embedding_grid", "grid", "--embeddings-dir", str(grid),
                    "--labels", str(grid / "labels_by_uid.jsonl"), "--splits", str(grid / "splits_by_uid.json"),
                    "--out", str(grid)],
                   [py, "-m", "parameters.edge.embedding_grid", "select", "--grid-dir", str(grid)]]},
    ]
    if pil:
        from Codec.edge import pilot_checks
        parts = [pilot.check_stage(n, work, py) for n in pilot_checks.CHECK_ORDER]
        stages.append({"name": "checks", "argvs": [p["argv"] for p in parts], "keep_going": True,
                       "timeout": 3 * 3600, "expect": [p["expect"][0] for p in parts],
                       "post": lambda: _checks_error(work)})
    stages.append({"name": "publish", "timeout": 3600, "skip_reason": "pilot mode never publishes" if pil else None,
                   "argvs": lambda: _publish_argvs(py, a, b, cache)})
    return stages


MERGE_CODE = ("import sys,json;from Codec.edge.feature_pass import summarize;s=summarize(sys.argv[1]);"
              "print('summary merged: shards done',s['counts'].get('shards',0),'images ok',s['counts'].get('ok',0),"
              "'failures',s['n_failures'])")


def _feature_parts(py, a, b, cache, pil, n_gpu):
    """[(label, argv, extra_env)]: one child per GPU (process-level sharding), or a single plain child."""
    base = [py, "-m", "Codec.edge.feature_pass", "run", "--images-root", str(a), "--labels-root", str(b),
            "--out", str(cache)] + (["--smoke"] if pil else [])
    if n_gpu < 2:
        return [("part0", base + ["--part", "0/1"], {})]
    return [("part%d" % i, base + ["--part", "%d/%d" % (i, n_gpu)], {"CUDA_VISIBLE_DEVICES": str(i)})
            for i in range(n_gpu)]


def gpu_count():
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _publish_argvs(py, a, b, cache):
    out = [[py, "-c", PUBLISH_CODE, "A", str(a)], [py, "-c", PUBLISH_CODE, "B", str(b)]]
    if (Path(cache) / "summary.json").exists():
        out.append([py, "-m", "Codec.edge.feature_pass", "publish", "--out", str(cache),
                    "--slug", "scheduler-dataset-v1-cache"])
    return out


def _checks_error(work):
    checks = pilot.read_checks(work)
    bad = [n for n, r in checks.items() if r.get("status") == "fail"]
    return ("checks failed: " + ", ".join(bad)) if bad else None


class Orchestrator:
    def __init__(self, mode, work, force=(), runner=default_runner, out=None, meta_dir=None, py=None, cwd=None,
                 gpus=gpu_count, progress_every=60.0):
        self.gpus, self.progress_every = gpus, progress_every
        self.mode, self.work, self.force = mode, Path(work), set(force or ())
        self.runner, self.cwd = runner, str(cwd or repo_root())
        self.out = out or (lambda text: print(text, flush=True))
        self.stages = build_stages(mode, self.work, meta_dir, py)
        self.results = {}
        for sub in ("state", "logs"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)

    def _env(self):
        env = dict(os.environ)
        env.update(TQDM_DISABLE="1", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                   PYTHONWARNINGS="ignore", TOKENIZERS_PARALLELISM="false")
        return env

    def _done_file(self, name):
        return self.work / "state" / (name + ".done.json")

    def _result(self, name, status, seconds=0.0, note=""):
        res = {"stage": name, "status": status, "seconds": round(seconds, 1), "note": note}
        self.results[name] = res
        return res

    def _run_parts(self, stage, parts, log):
        """Run the children concurrently (own log each); returns an error text or None."""
        name, rcs, lock = stage["name"], {}, threading.Lock()
        shards = self.work / "cache" / "shards"
        stop = threading.Event()

        def child(label, argv, extra):
            env = self._env()
            env.update(extra)
            with open(self.work / "logs" / ("%s_%s.log" % (name, label)), "w", encoding="utf-8") as plog:
                plog.write("$ %s  %s\n" % (extra, " ".join(argv)))
                def emit(line):
                    plog.write(line if line.endswith("\n") else line + "\n")
                try:
                    rc = self.runner(argv, self.cwd, env, stage.get("timeout"), emit)
                except Exception as error:
                    plog.write("ERROR could not run: %r\n" % (error,))
                    rc = -1
            with lock:
                rcs[label] = rc
                self.out("  | %s %s finished rc=%s" % (name, label, rc))

        def progress():
            while not stop.wait(self.progress_every):
                n = len(list(shards.glob("shard_????/done.json"))) if shards.exists() else 0
                self.out("  | %s progress: %d shards finished" % (name, n))

        threads = []
        for label, argv, extra in parts:
            log.write("$ [%s] %s %s\n" % (label, extra, " ".join(argv)))
            self.out("  | %s %s start%s" % (name, label, (" (" + ", ".join("%s=%s" % kv for kv in extra.items()) + ")")
                                         if extra else ""))
            threads.append(threading.Thread(target=child, args=(label, argv, extra)))
        ticker = threading.Thread(target=progress, daemon=True)
        ticker.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        bad = ["%s rc=%s" % (k, v) for k, v in sorted(rcs.items()) if v != 0]
        return ("feature pass failed: " + ", ".join(bad) + " (see logs/%s_<part>.log)" % name) if bad else None

    def run_stage(self, stage, index, total):
        name = stage["name"]
        self.out("==== STAGE %d/%d: %s ====" % (index, total, name))
        if stage.get("skip_reason"):
            self.out("%s: SKIP (%s)" % (name, stage["skip_reason"]))
            return self._result(name, "SKIP", note=stage["skip_reason"])
        blocked = [d for d in DEPS.get(name, []) if d in self.results and self.results[d]["status"] not in ("PASS",)]
        if blocked:
            first = self.results[blocked[0]]
            note = "needs %s (%s: %s)" % (", ".join(blocked), first["status"], first["note"])
            self.out("%s: SKIP (%s)" % (name, note))
            return self._result(name, "SKIP", note=note)
        if self._done_file(name).exists() and name not in self.force:
            self.out("%s: already done, skipped (use --force %s to redo)" % (name, name))
            return self._result(name, "PASS", note="resumed")
        argvs = stage["argvs"]() if callable(stage["argvs"]) else stage["argvs"]
        log_path = self.work / "logs" / (name + ".log")
        flt = LineFilter(keep_all=stage.get("keep_all", False))
        self.out("%s: start" % name)
        start, failed = time.time(), None
        with open(log_path, "w", encoding="utf-8") as log:
            def emit(line):
                log.write(line if line.endswith("\n") else line + "\n")
                shown = flt.accept(line)
                if shown:
                    self.out("  | " + shown)
            if stage.get("parallel"):
                failed = self._run_parts(stage, stage["parallel"](self.gpus()), log)
                if failed:
                    argvs = []                 # finished shards stay valid for resume; no merge on failure
            for argv in argvs:
                log.write("$ %s\n" % " ".join(map(str, argv)))
                try:
                    rc = self.runner([str(x) for x in argv], self.cwd, self._env(), stage.get("timeout"), emit)
                except Exception as error:
                    emit("ERROR could not run: %r" % (error,))
                    rc = -1
                if rc != 0:
                    failed = failed or "exit code %s from %s" % (rc, " ".join(map(str, argv))[:120])
                    if not stage.get("keep_going"):
                        break
        missing = [p.name for p in stage.get("expect", []) if not Path(p).exists()]
        if not failed and missing:
            failed = "missing outputs: " + ", ".join(missing)
        if not failed and stage.get("post"):
            failed = stage["post"]()
        seconds = time.time() - start
        if flt.dropped:
            self.out("  | (%d more lines in logs/%s.log)" % (flt.dropped, name))
        if failed:
            self.out("%s: FAIL after %.0fs (%s)" % (name, seconds, failed))
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-TAIL_ON_FAIL:]
            self.out("--- last %d lines of logs/%s.log ---\n%s\n---" % (len(lines), name, "\n".join(lines)))
            return self._result(name, "FAIL", seconds, failed)
        self._done_file(name).write_text(json.dumps({"stage": name, "seconds": round(seconds, 1),
                                                     "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        self.out("%s: done in %.0fs" % (name, seconds))
        return self._result(name, "PASS", seconds)

    def run(self):
        total = len(self.stages)
        for i, stage in enumerate(self.stages, 1):
            self.run_stage(stage, i, total)
        if self.mode == "pilot":
            self._write_pilot_report()
        self.out(self.summary())
        (self.work / "state" / "summary.json").write_text(json.dumps(list(self.results.values()), indent=2))
        return 1 if any(r["status"] == "FAIL" for r in self.results.values()) else 0

    def summary(self):
        rows = ["RUN SUMMARY (mode=%s)" % self.mode, "stage        | result | seconds | note",
                "-------------|--------|---------|-----"]
        for r in self.results.values():
            rows.append("%-12s | %-6s | %7.0f | %s" % (r["stage"], r["status"], r["seconds"], r["note"]))
        sizes = ", ".join("%s=%s MB" % (k, dir_size_mb(self.work / d))
                          for k, d in (("A", "dataset-a"), ("B", "dataset-b"), ("C", "cache")))
        text = "\n".join(rows + ["sizes: " + sizes])
        (self.work / "RUN_SUMMARY.md").write_text(text + "\n", encoding="utf-8")
        return text

    def _write_pilot_report(self):
        env_file = self.work / "environment.json"
        environment = json.loads(env_file.read_text()) if env_file.exists() else {}
        stages = [{"name": n, "passed": r["status"] == "PASS", "seconds": r["seconds"], "missing_outputs": [],
                   "stderr_tail": r["note"], "stdout_tail": ""} for n, r in self.results.items()]
        report = {"mode": "pilot", "environment": environment, "stages": stages, "checks": pilot.read_checks(self.work)}
        report["all_passed"] = all(s["passed"] or self.results[s["name"]]["status"] == "SKIP" for s in stages)
        (self.work / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.work / "PILOT_REPORT.md").write_text(pilot.markdown(report), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--work", default="/kaggle/working/run")
    ap.add_argument("--force", action="append", default=[], metavar="STAGE", help="re-run a finished stage (repeatable)")
    ap.add_argument("--meta-dir", default=None)
    a = ap.parse_args(argv)
    return Orchestrator(a.mode, a.work, a.force, meta_dir=a.meta_dir).run()


if __name__ == "__main__":
    sys.exit(main())
