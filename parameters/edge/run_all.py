"""Two-notebook orchestrator (``--which``): the working parts are not blocked by SQ-GAN problems.

    python -m parameters.edge.run_all --which ab-embed|sqgan --mode pilot|full --work /kaggle/working/run [--force STAGE ...]

``ab-embed`` (notebook ``dataset-semschedule``): env, A (images), B (labels), C_features (embeddings of all 5
encoders + SimHash + pHash, ``--no-sqgan``, 2-way GPU sharded), C_grid, publish. No SQ-GAN anywhere.
``sqgan`` (notebook ``dataset-semschedule-sqgan``): env, S_locate (finds A and B in the output of notebook 1
under /kaggle/input), S_sqgan (``--only-sqgan --require-sqgan``, 2-way GPU sharded), checks (pilot only, the
SQ-GAN related ones), publish (``scheduler-dataset-v1-sqgan``, needs a Kaggle Secret).

Every stage is a SEPARATE SUBPROCESS. A failed stage is recorded, stages that depend on it are marked
SKIP, independent stages still run. Stages write ``<work>/state/<stage>.done.json`` and are skipped on
re-run unless ``--force <stage>``. Full child output goes to ``<work>/logs/<stage>.log``; the console
(the Kaggle web log) gets a banner per stage, a few filtered lines and a RUN SUMMARY table.

``pilot`` uses --max-images 20, --smoke and never publishes; ``full`` runs full sizes and enforces the B
gates (test/val Critical counts) before C. The exit code is non-zero when any stage fails.
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

WHICH = ("ab-embed", "sqgan")
# stage name -> (stages whose success it needs), per notebook
DEPS_BY = {"ab-embed": {"env": [], "A": [], "B": ["A"], "C_features": ["A", "B"], "C_grid": ["C_features"],
                        "publish": ["A", "B"]},
           "sqgan": {"env": [], "S_locate": [], "S_sqgan": ["S_locate"], "checks": [], "publish": ["S_sqgan"]}}
DEPS = DEPS_BY["ab-embed"]
# the pilot checks that belong to the SQ-GAN notebook (env_gpu = environment facts)
SQGAN_CHECKS = ("env_gpu", "bundle_restore", "source_root", "bundle_deep_scan", "payload_semantic_map",
                "masker_format", "feature_pass_codec_semantics", "sqgan_latency", "compute_profile_real",
                "finetune_source", "finetune_pilot", "feature_pass_smoke_verify")
SQGAN_SLUG = "scheduler-dataset-v1-sqgan"
LOCATE_DEPTH = 6
PRUNE_DIRS = ("images", "shards", "candidates", ".git", "__pycache__")
NOTEBOOK1 = "dataset-semschedule"


class LocateError(RuntimeError):
    """Dataset A (index.jsonl) or B (labels_raw.jsonl) was not found exactly once under the input root."""


def _find_files(root, name, depth=LOCATE_DEPTH):
    root = Path(root)
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) >= depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        if name in filenames:
            hits.append(Path(dirpath))
    return sorted(hits)


def locate_inputs(root="/kaggle/input", depth=LOCATE_DEPTH):
    """{'A': folder with index.jsonl, 'B': folder with labels_raw.jsonl}; LocateError unless exactly one each."""
    found, problems = {}, []
    for key, fname, what in (("A", "index.jsonl", "dataset A (images)"), ("B", "labels_raw.jsonl", "dataset B (labels)")):
        hits = _find_files(root, fname, depth)
        if len(hits) == 1:
            found[key] = hits[0]
        else:
            problems.append("%s: found %d folders with %s under %s (need exactly 1)%s" % (
                what, len(hits), fname, root, "".join("\n    " + str(h) for h in hits)))
    if problems:
        raise LocateError(
            "cannot locate the notebook-1 output. Run notebook '%s' (pilot or full) first and keep it attached as a "
            "kernel source of this notebook.\n  %s" % (NOTEBOOK1, "\n  ".join(problems)))
    return found


def locate_main(work, root="/kaggle/input"):
    """Subprocess entry for stage S_locate: writes <work>/located.json; exit 1 with a clear message on failure."""
    try:
        found = locate_inputs(root)
    except LocateError as error:
        print("ERROR", error)
        return 1
    Path(work).mkdir(parents=True, exist_ok=True)
    Path(work, "located.json").write_text(json.dumps({k: str(v) for k, v in found.items()}), encoding="utf-8")
    print("located dataset A images at", found["A"], "and dataset B labels at", found["B"])
    return 0


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


LOCATE_CODE = "import sys;from parameters.edge.run_all import locate_main;sys.exit(locate_main(sys.argv[1], sys.argv[2]))"


def has_secret():
    """True when Kaggle Secrets provide credentials (dataset publishing possible)."""
    try:
        from parameters.edge import kaggle_publish
        return kaggle_publish.get_credentials() is not None
    except Exception:
        return False


def _grid_note(grid):
    """Pass note of C_grid: pilot data too thin for a classifier (GridDataError allowed) is plumbing only."""
    f = Path(grid) / "grid_degenerate.json"
    if f.exists():
        try:
            return "plumbing only: " + json.loads(f.read_text(encoding="utf-8"))["message"]
        except (OSError, ValueError, KeyError):
            return "plumbing only: degenerate grid data"
    return ""


def build_stages(mode, work, meta_dir=None, py=None, which="ab-embed", input_root="/kaggle/input"):
    """Ordered stage dicts: name, argvs (list of argv lists), timeout, expect, keep_all, keep_going, post."""
    if which not in WHICH:
        raise ValueError("which must be one of %s, got %r" % (WHICH, which))
    work, py = Path(work), py or sys.executable
    pil = mode == "pilot"
    env_stage = {"name": "env", "argvs": [[py, "-c", ENV_CODE, str(work)]], "timeout": 300, "keep_all": True}
    if which == "sqgan":
        return _sqgan_stages(pil, work, py, input_root, env_stage)
    meta_dir = str(meta_dir or (repo_root() / pilot.META_DIR))
    a, b, cache = work / "dataset-a", work / "dataset-b", work / "cache"
    grid = cache / "grid"
    degenerate = ["--allow-degenerate"] if pil else []
    return [
        env_stage,
        {"name": "A", "timeout": 6 * 3600, "expect": [a / "index.jsonl", a / "manifest.json"],
         "argvs": [[py, "-m", "parameters.edge.dataset_a", "--meta-dir", meta_dir, "--out", str(a),
                    "--workers", "8" if pil else "16", "--no-publish"] + (["--max-images", "20"] if pil else [])]},
        {"name": "B", "timeout": 1800, "expect": [b / "labels_raw.jsonl", b / "counts.json"],
         "argvs": [[py, "-m", "parameters.edge.build_label_dataset", "--meta-dir", meta_dir, "--out", str(b),
                    "--images-root", str(a), "--no-publish"]],
         "post": lambda: _gate_error(b / "counts.json", enforce=not pil)},
        {"name": "C_features", "timeout": 8 * 3600, "expect": [cache / "summary.json"], "cache_dir": cache,
         "parallel": lambda n_gpu: _feature_parts(py, a, b, cache, pil, n_gpu, ["--no-sqgan"]),
         "argvs": [[py, "-c", MERGE_CODE, str(cache)]]},
        {"name": "C_grid", "timeout": 4 * 3600, "pass_note": lambda: _grid_note(grid),
         "argvs": [[py, "-m", "Codec.edge.feature_pass", "export-grid", "--labels-root", str(b), "--out", str(cache)],
                   [py, "-m", "parameters.edge.embedding_grid", "grid", "--embeddings-dir", str(grid),
                    "--labels", str(grid / "labels_by_uid.jsonl"), "--splits", str(grid / "splits_by_uid.json"),
                    "--out", str(grid)] + degenerate,
                   [py, "-m", "parameters.edge.embedding_grid", "select", "--grid-dir", str(grid)] + degenerate]},
        {"name": "publish", "timeout": 3600, "skip_reason": "pilot mode never publishes" if pil else None,
         "argvs": lambda: _publish_argvs(py, a, b, cache)},
    ]


def _sqgan_stages(pil, work, py, input_root, env_stage):
    cache = work / "cache_sqgan"

    def located():
        info = json.loads((work / "located.json").read_text(encoding="utf-8"))
        return Path(info["A"]), Path(info["B"])

    def publish_skip():
        if pil:
            return "pilot mode never publishes"
        return None if has_secret() else "no Kaggle Secret (add KAGGLE_API_TOKEN) so nothing can be published"

    stages = [env_stage,
              {"name": "S_locate", "timeout": 600, "keep_all": True, "expect": [work / "located.json"],
               "argvs": [[py, "-c", LOCATE_CODE, str(work), str(input_root)]]},
              {"name": "S_sqgan", "timeout": 8 * 3600, "expect": [cache / "summary.json"], "cache_dir": cache,
               "parallel": lambda n_gpu: _feature_parts(py, *located(), cache, pil, n_gpu,
                                                        ["--only-sqgan", "--require-sqgan"]),
               "argvs": [[py, "-c", MERGE_CODE, str(cache)]]}]
    if pil:
        parts = pilot.check_stages(work, py, SQGAN_CHECKS)
        stages.append({"name": "checks", "argvs": [p["argv"] for p in parts], "keep_going": True,
                       "timeout": 3 * 3600, "expect": [p["expect"][0] for p in parts],
                       "post": lambda: _checks_error(work)})
    stages.append({"name": "publish", "timeout": 3600, "skip_reason": publish_skip,
                   "argvs": [[py, "-m", "Codec.edge.feature_pass", "publish", "--out", str(cache),
                              "--slug", SQGAN_SLUG]]})
    return stages


MERGE_CODE = ("import sys,json;from Codec.edge.feature_pass import summarize;s=summarize(sys.argv[1]);"
              "print('summary merged: shards done',s['counts'].get('shards',0),'images ok',s['counts'].get('ok',0),"
              "'failures',s['n_failures'])")


def _feature_parts(py, a, b, cache, pil, n_gpu, extra_flags=()):
    """[(label, argv, extra_env)]: one child per GPU (process-level sharding), or a single plain child."""
    base = [py, "-m", "Codec.edge.feature_pass", "run", "--images-root", str(a), "--labels-root", str(b),
            "--out", str(cache)] + (["--smoke"] if pil else []) + list(extra_flags)
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
                 gpus=gpu_count, progress_every=60.0, which="ab-embed", input_root="/kaggle/input"):
        self.gpus, self.progress_every, self.which = gpus, progress_every, which
        self.deps = DEPS_BY[which]
        self.mode, self.work, self.force = mode, Path(work), set(force or ())
        self.runner, self.cwd = runner, str(cwd or repo_root())
        self.out = out or (lambda text: print(text, flush=True))
        self.stages = build_stages(mode, self.work, meta_dir, py, which, input_root)
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
        shards = Path(stage.get("cache_dir") or self.work / "cache") / "shards"
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
        reason = stage.get("skip_reason")
        reason = reason() if callable(reason) else reason
        if reason:
            self.out("%s: SKIP (%s)" % (name, reason))
            return self._result(name, "SKIP", note=reason)
        blocked = [d for d in self.deps.get(name, []) if d in self.results and self.results[d]["status"] not in ("PASS",)]
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
        note = stage["pass_note"]() if stage.get("pass_note") else ""
        self.out("%s: done in %.0fs%s" % (name, seconds, (" (%s)" % note) if note else ""))
        return self._result(name, "PASS", seconds, note)

    def run(self):
        if self.which == "sqgan":
            self.out("NOTE: this notebook has no dataset A/B of its own. It reads A (images) and B (labels) from the "
                     "output of notebook '%s' (kernel source; run that notebook once, pilot or full, first)." % NOTEBOOK1)
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
        dirs = ((("A", "dataset-a"), ("B", "dataset-b"), ("C", "cache")) if self.which == "ab-embed"
                else (("SQGAN", "cache_sqgan"),))
        sizes = ", ".join("%s=%s MB" % (k, dir_size_mb(self.work / d)) for k, d in dirs)
        text = "\n".join(rows + ["sizes: " + sizes])
        (self.work / "RUN_SUMMARY.md").write_text(text + "\n", encoding="utf-8")
        return text

    def _write_pilot_report(self):
        env_file = self.work / "environment.json"
        environment = json.loads(env_file.read_text()) if env_file.exists() else {}
        stages = [{"name": n, "status": r["status"], "passed": r["status"] == "PASS", "seconds": r["seconds"],
                   "missing_outputs": [], "stderr_tail": r["note"], "stdout_tail": ""}
                  for n, r in self.results.items()]
        report = {"mode": "pilot", "which": self.which, "environment": environment, "stages": stages,
                  "checks": pilot.read_checks(self.work),
                  "only_checks": list(SQGAN_CHECKS) if self.which == "sqgan" else []}
        report["all_passed"] = all(s["status"] in ("PASS", "SKIP") for s in stages)
        # pilot-check compatible record (workers16_memory reads stage_results.json by the old pilot stage names)
        alias = {"A": "a_download_images", "B": "b_label", "C_features": "c_feature_pass_smoke", "C_grid": "d_grid_export"}
        legacy = [{"name": alias.get(s["name"], s["name"]), "passed": s["passed"], "seconds": s["seconds"],
                   "status": s["status"], "peak_rss_mb": pilot.child_peak_rss_mb()} for s in stages]
        (self.work / "stage_results.json").write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        (self.work / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.work / "PILOT_REPORT.md").write_text(pilot.markdown(report), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=WHICH, default="ab-embed")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--work", default="/kaggle/working/run")
    ap.add_argument("--force", action="append", default=[], metavar="STAGE", help="re-run a finished stage (repeatable)")
    ap.add_argument("--meta-dir", default=None)
    a = ap.parse_args(argv)
    return Orchestrator(a.mode, a.work, a.force, meta_dir=a.meta_dir, which=a.which, input_root=a.input_root).run()


if __name__ == "__main__":
    sys.exit(main())
