import json
import re
from pathlib import Path

from parameters.edge import run_all


class FakeRunner:
    """Records argv; fails stages by substring; writes the files a real stage would produce."""

    def __init__(self, work, fail=(), gates_ok=True, lines=("done",)):
        self.work, self.fail, self.gates_ok, self.lines, self.calls = Path(work), list(fail), gates_ok, lines, []
        self.envs = []

    def __call__(self, argv, cwd, env, timeout, emit):
        text = " ".join(argv)
        self.calls.append(text)
        self.envs.append(env.get("CUDA_VISIBLE_DEVICES"))
        for line in self.lines:
            emit(line + "\n")
        w = self.work
        if any(f in text for f in self.fail):
            emit("ERROR boom\n")
            return 1
        if "dataset_a" in text:
            (w / "dataset-a").mkdir(parents=True, exist_ok=True)
            (w / "dataset-a" / "index.jsonl").write_text("x")
            (w / "dataset-a" / "manifest.json").write_text("{}")
        if "build_label_dataset" in text:
            (w / "dataset-b").mkdir(parents=True, exist_ok=True)
            (w / "dataset-b" / "labels_raw.jsonl").write_text("x")
            (w / "dataset-b" / "counts.json").write_text(json.dumps(
                {"gates": {"test_critical": 1, "test_critical_ok": self.gates_ok, "val_critical": 1,
                           "val_critical_ok": True}}))
        if "feature_pass run" in text:
            (w / "cache").mkdir(parents=True, exist_ok=True)
            (w / "cache" / "summary.json").write_text("{}")
        if "pilot_checks" in text:
            name = argv[3]
            (w / "checks").mkdir(parents=True, exist_ok=True)
            (w / "checks" / (name + ".json")).write_text(json.dumps({"name": name, "status": "pass", "items": {}}))
        return 0


def run(tmp_path, mode="full", fail=(), force=(), gates_ok=True, gpus=1, **kw):
    out = []
    runner = FakeRunner(tmp_path, fail, gates_ok, **kw)
    orch = run_all.Orchestrator(mode, tmp_path, force, runner=runner, out=out.append, meta_dir="meta", py="py", cwd=".",
                                  gpus=lambda: gpus, progress_every=3600)
    rc = orch.run()
    return orch, runner, out, rc


def status(orch):
    return {k: v["status"] for k, v in orch.results.items()}


def test_full_order_banners_and_no_checks(tmp_path):
    orch, runner, out, rc = run(tmp_path)
    assert rc == 0
    assert list(orch.results) == ["env", "A", "B", "C_features", "C_grid", "publish"]
    banners = [o for o in out if o.startswith("====")]
    assert banners[0] == "==== STAGE 1/6: env ====" and banners[-1] == "==== STAGE 6/6: publish ===="
    assert not any("-m Codec.edge.pilot_checks" in c for c in runner.calls)
    assert not any("--max-images" in c or "--smoke" in c for c in runner.calls)
    cmds = [c for c in runner.calls]
    assert any("export-grid" in c for c in cmds) and any("embedding_grid grid" in c for c in cmds)
    assert any("embedding_grid select --grid-dir" in c for c in cmds)
    assert sum("kaggle_publish" in c for c in cmds) == 2 and any("feature_pass publish" in c for c in cmds)
    assert "scheduler-dataset-v1-cache" in " ".join(cmds)


def test_pilot_adds_checks_flags_and_report(tmp_path):
    orch, runner, out, rc = run(tmp_path, mode="pilot")
    joined = "\n".join(runner.calls)
    assert "--max-images 20" in joined and "--smoke" in joined and "-m Codec.edge.pilot_checks" in joined
    assert status(orch)["checks"] == "PASS" and status(orch)["publish"] == "SKIP"
    assert (tmp_path / "PILOT_REPORT.md").exists()
    assert "## Unverified items" in (tmp_path / "PILOT_REPORT.md").read_text()


def test_failure_skips_dependents_but_not_independent(tmp_path):
    orch, runner, out, rc = run(tmp_path, fail=["dataset_a"])
    s = status(orch)
    assert rc == 1 and s["A"] == "FAIL" and s["env"] == "PASS"
    assert s["B"] == s["C_features"] == s["C_grid"] == s["publish"] == "SKIP"
    assert any("last" in o and "A.log" in o and "boom" in o for o in out)
    assert not (tmp_path / "state" / "A.done.json").exists()


def test_gate_failure_stops_c_and_says_why(tmp_path):
    orch, runner, out, rc = run(tmp_path, gates_ok=False)
    s = status(orch)
    assert s["B"] == "FAIL" and s["C_features"] == "SKIP" and s["C_grid"] == "SKIP"
    assert "test_critical_ok" in orch.results["B"]["note"] and "gate" in orch.results["C_features"]["note"]
    assert not any("feature_pass run" in c for c in runner.calls)
    orch2, _, _, rc2 = run(tmp_path / "pilot", mode="pilot", gates_ok=False)   # tiny pilot sample: gate is not enforced
    assert status(orch2)["C_features"] == "PASS"


def test_resume_and_force(tmp_path):
    run(tmp_path)
    orch, runner, out, rc = run(tmp_path)
    assert runner.calls == [] and all(r["status"] == "PASS" for r in orch.results.values())
    assert any("already done" in o for o in out)
    orch, runner, out, rc = run(tmp_path, force=["B"])
    assert len(runner.calls) == 1 and "build_label_dataset" in runner.calls[0]


def test_failed_stage_is_retried_on_rerun(tmp_path):
    run(tmp_path, fail=["build_label_dataset"])
    orch, runner, out, rc = run(tmp_path)
    assert rc == 0 and len(runner.calls) > 0 and not any("-m parameters.edge.dataset_a" in c for c in runner.calls)


def test_line_filter_drops_progress_and_repeats_keeps_key_lines():
    f = run_all.LineFilter(limit=3)
    lines = ["  45%|####      | 9/20 [00:01<00:02]", "Downloading 12.3it/s", "random chatter", "shard 1 done",
             "shard 1 done", "WARNING low disk", "ERROR failed x", "7 images saved", "\r50% 3/6"]
    kept = [k for k in map(f.accept, lines) if k]
    assert kept == ["shard 1 done", "WARNING low disk", "ERROR failed x"]
    assert f.dropped == 1


def test_console_output_is_capped(tmp_path):
    lines = ["shard %d done" % i for i in range(100)]
    orch, runner, out, rc = run(tmp_path, lines=lines)
    a_lines = [o for o in out if o.startswith("  | shard")]
    assert 0 < len(a_lines) <= run_all.MAX_CHILD_LINES * 6
    assert "shard 99 done" in (tmp_path / "logs" / "A.log").read_text()


def test_summary_table_and_sizes(tmp_path):
    orch, runner, out, rc = run(tmp_path, fail=["embedding_grid grid"])
    text = out[-1]
    assert text.startswith("RUN SUMMARY") and "stage" in text and "FAIL" in text and "PASS" in text
    assert re.search(r"sizes: A=[\d.]+ MB, B=[\d.]+ MB, C=[\d.]+ MB", text)
    assert (tmp_path / "RUN_SUMMARY.md").exists()
    assert json.loads((tmp_path / "state" / "summary.json").read_text())[0]["stage"] == "env"


def _feature_calls(runner):
    return [(c, e) for c, e in zip(runner.calls, runner.envs) if "feature_pass run" in c]


def test_two_gpus_two_children_with_env_and_part(tmp_path):
    orch, runner, out, rc = run(tmp_path, gpus=2)
    calls = sorted(_feature_calls(runner), key=lambda x: x[0])
    assert len(calls) == 2 and rc == 0
    assert "--part 0/2" in calls[0][0] and calls[0][1] == "0"
    assert "--part 1/2" in calls[1][0] and calls[1][1] == "1"
    assert (tmp_path / "logs" / "C_features_part0.log").exists() and (tmp_path / "logs" / "C_features_part1.log").exists()
    assert any("summarize" in c for c in runner.calls)          # shard summary merge after both finish
    assert any("part1 finished rc=0" in o for o in out)


def test_one_gpu_single_child_and_pilot_still_splits(tmp_path):
    orch, runner, out, rc = run(tmp_path, gpus=1)
    calls = _feature_calls(runner)
    assert len(calls) == 1 and "--part 0/1" in calls[0][0] and calls[0][1] is None
    orch, runner, out, rc = run(tmp_path / "p", mode="pilot", gpus=2)
    calls = _feature_calls(runner)
    assert len(calls) == 2 and all("--smoke" in c for c, _ in calls)


def test_one_part_failing_fails_stage_without_merge(tmp_path):
    orch, runner, out, rc = run(tmp_path, gpus=2, fail=["--part 1/2"])
    assert status(orch)["C_features"] == "FAIL" and status(orch)["C_grid"] == "SKIP"
    assert "part1" in orch.results["C_features"]["note"] and "part0" not in orch.results["C_features"]["note"]
    assert not any("summarize" in c for c in runner.calls)


def test_summary_merge_aggregates_shards(tmp_path):
    from Codec.edge import feature_pass as fp
    for i, n in enumerate((3, 5)):
        d = tmp_path / "shards" / ("shard_%04d" % i)
        d.mkdir(parents=True)
        (d / "done.json").write_text(json.dumps({"n_requested": n, "n_ok": n - 1, "skipped": {}}))
        (d / "failures.json").write_text("[]")
        (d / "timings.jsonl").write_text("")
    s = fp.summarize(tmp_path)
    assert s["counts"]["shards"] == 2 and s["counts"]["requested"] == 8 and s["counts"]["ok"] == 6
    assert json.loads((tmp_path / "summary.json").read_text())["counts"]["ok"] == 6
