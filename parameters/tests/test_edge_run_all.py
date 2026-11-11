import json
import re
from pathlib import Path

from parameters.edge import run_all


class FakeRunner:
    """Records argv; fails stages by substring; writes the files a real stage would produce."""

    def __init__(self, work, fail=(), gates_ok=True, lines=("done",), degenerate=False):
        self.work, self.fail, self.gates_ok, self.lines, self.calls = Path(work), list(fail), gates_ok, lines, []
        self.degenerate = degenerate
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
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text("{}")
        if "locate_main" in text:
            (w / "located.json").write_text(json.dumps({"A": str(w / "fa"), "B": str(w / "fb")}))
        if "embedding_grid grid" in text and self.degenerate and "--allow-degenerate" in text:
            (w / "cache" / "grid").mkdir(parents=True, exist_ok=True)
            (w / "cache" / "grid" / "grid_degenerate.json").write_text(json.dumps({"message": "train n=3 only Low"}))
        if "embedding_grid grid" in text and self.degenerate and "--allow-degenerate" not in text:
            emit("GridDataError: only class 3\n")
            return 1
        if "pilot_checks" in text:
            name = argv[3]
            (w / "checks").mkdir(parents=True, exist_ok=True)
            (w / "checks" / (name + ".json")).write_text(json.dumps({"name": name, "status": "pass", "items": {}}))
        return 0


def run(tmp_path, mode="full", fail=(), force=(), gates_ok=True, gpus=1, which="ab-embed", **kw):
    out = []
    runner = FakeRunner(tmp_path, fail, gates_ok, **kw)
    orch = run_all.Orchestrator(mode, tmp_path, force, runner=runner, out=out.append, meta_dir="meta", py="py", cwd=".",
                                  gpus=lambda: gpus, progress_every=3600, which=which)
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
    assert all("--no-sqgan" in c and "--only-sqgan" not in c for c in runner.calls if "feature_pass run" in c)
    cmds = [c for c in runner.calls]
    assert any("export-grid" in c for c in cmds) and any("embedding_grid grid" in c for c in cmds)
    assert any("embedding_grid select --grid-dir" in c for c in cmds)
    assert sum("kaggle_publish" in c for c in cmds) == 2 and any("feature_pass publish" in c for c in cmds)
    assert "scheduler-dataset-v1-cache" in " ".join(cmds) and "sqgan" not in " ".join(c for c in cmds if "publish" in c)


def test_pilot_flags_report_and_skip_matches_state(tmp_path):
    orch, runner, out, rc = run(tmp_path, mode="pilot")
    joined = "\n".join(runner.calls)
    assert "--max-images 20" in joined and "--smoke" in joined and "-m Codec.edge.pilot_checks" not in joined
    assert list(orch.results) == ["env", "A", "B", "C_features", "C_grid", "publish"]
    assert status(orch)["publish"] == "SKIP" and rc == 0
    md = (tmp_path / "PILOT_REPORT.md").read_text()
    assert "| publish | SKIP |" in md and "(failed)" not in md          # report == run_all state
    rep = json.loads((tmp_path / "pilot_report.json").read_text())
    assert rep["all_passed"] is True
    legacy = {s["name"]: s for s in json.loads((tmp_path / "stage_results.json").read_text())}
    assert "a_download_images" in legacy                                 # workers16_memory finds stage A


def test_grid_data_error_pilot_passes_plumbing_only_full_fails(tmp_path):
    orch, runner, out, rc = run(tmp_path / "p", mode="pilot", degenerate=True)
    assert status(orch)["C_grid"] == "PASS" and rc == 0
    assert orch.results["C_grid"]["note"].startswith("plumbing only: train n=3")
    assert all("--allow-degenerate" in c for c in runner.calls if "embedding_grid" in c)
    orch, runner, out, rc = run(tmp_path / "f", mode="full", degenerate=True)
    assert status(orch)["C_grid"] == "FAIL" and rc == 1
    assert not any("--allow-degenerate" in c for c in runner.calls)


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


# ------------------------------------------------------------------ notebook 2 (sqgan)

def test_sqgan_full_stage_list_and_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "has_secret", lambda: True)
    orch, runner, out, rc = run(tmp_path, which="sqgan", gpus=2)
    assert rc == 0 and list(orch.results) == ["env", "S_locate", "S_sqgan", "publish"]
    calls = _feature_calls(runner)
    assert len(calls) == 2
    for c, _ in calls:
        assert "--only-sqgan" in c and "--require-sqgan" in c and "--no-sqgan" not in c and "--smoke" not in c
        assert "--images-root %s" % (tmp_path / "fa") in c and "--labels-root %s" % (tmp_path / "fb") in c
        assert "--out %s" % (tmp_path / "cache_sqgan") in c
    assert sorted(e for _, e in calls) == ["0", "1"]
    assert any("feature_pass publish" in c and "scheduler-dataset-v1-sqgan" in c for c in runner.calls)
    assert not any("dataset_a" in c or "build_label_dataset" in c for c in runner.calls)
    assert any(o.startswith("NOTE: this notebook has no dataset A/B") for o in out)
    assert "SQGAN=" in out[-1]


def test_sqgan_pilot_checks_subset_smoke_and_no_publish(tmp_path):
    orch, runner, out, rc = run(tmp_path, mode="pilot", which="sqgan")
    assert list(orch.results) == ["env", "S_locate", "S_sqgan", "checks", "publish"]
    assert status(orch)["publish"] == "SKIP" and rc == 0
    names = [c.split("-m Codec.edge.pilot_checks ")[1].split()[0] for c in runner.calls if "-m Codec.edge.pilot_checks" in c]
    from Codec.edge import pilot_checks
    assert names == [n for n in pilot_checks.CHECK_ORDER if n in run_all.SQGAN_CHECKS]
    assert {"env_gpu", "bundle_restore", "feature_pass_smoke_verify"} <= set(names) and "embedders" not in names
    assert all("--smoke" in c for c, _ in _feature_calls(runner))
    md = (tmp_path / "PILOT_REPORT.md").read_text()
    assert "embedders" not in md and "| publish | SKIP |" in md


def test_sqgan_publish_skipped_without_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "has_secret", lambda: False)
    orch, runner, out, rc = run(tmp_path, which="sqgan")
    assert status(orch)["publish"] == "SKIP" and "Secret" in orch.results["publish"]["note"] and rc == 0
    assert not any("feature_pass publish" in c for c in runner.calls)


def test_sqgan_dependencies_and_failure_propagation(tmp_path):
    assert run_all.DEPS_BY["sqgan"]["S_sqgan"] == ["S_locate"] and run_all.DEPS_BY["sqgan"]["publish"] == ["S_sqgan"]
    assert run_all.DEPS_BY["ab-embed"]["C_features"] == ["A", "B"] and run_all.DEPS_BY["ab-embed"]["C_grid"] == ["C_features"]
    orch, runner, out, rc = run(tmp_path, which="sqgan", fail=["locate_main"])
    s = status(orch)
    assert rc == 1 and s["S_locate"] == "FAIL" and s["S_sqgan"] == "SKIP" and s["publish"] == "SKIP"
    assert not any("feature_pass run" in c for c in runner.calls)


def test_sqgan_part_failure_fails_core_stage(tmp_path):
    orch, runner, out, rc = run(tmp_path, which="sqgan", gpus=2, fail=["--part 0/2"])
    assert rc == 1 and status(orch)["S_sqgan"] == "FAIL" and not any("summarize" in c for c in runner.calls)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_locate_inputs_finds_exactly_one_each(tmp_path):
    root = tmp_path / "input"
    _touch(root / "dataset-semschedule" / "run" / "dataset-a" / "index.jsonl")
    _touch(root / "dataset-semschedule" / "run" / "dataset-b" / "labels_raw.jsonl")
    _touch(root / "dataset-semschedule" / "run" / "dataset-a" / "images" / "crisismmd" / "index.jsonl")  # pruned
    found = run_all.locate_inputs(root)
    assert found["A"].name == "dataset-a" and found["B"].name == "dataset-b"
    assert run_all.locate_main(tmp_path / "w", root) == 0
    assert json.loads((tmp_path / "w" / "located.json").read_text())["A"].endswith("dataset-a")


def test_locate_errors_are_clear(tmp_path, capsys):
    import pytest
    root = tmp_path / "input"
    root.mkdir()
    with pytest.raises(run_all.LocateError) as e:                       # nothing there
        run_all.locate_inputs(root)
    msg = str(e.value)
    assert "dataset-semschedule" in msg and "index.jsonl" in msg and "labels_raw.jsonl" in msg and "found 0" in msg
    _touch(root / "k1" / "dataset-a" / "index.jsonl")
    _touch(root / "k2" / "dataset-a" / "index.jsonl")
    _touch(root / "k1" / "dataset-b" / "labels_raw.jsonl")
    with pytest.raises(run_all.LocateError, match="found 2"):           # ambiguous A
        run_all.locate_inputs(root)
    assert run_all.locate_main(tmp_path / "w2", root) == 1
    assert "ERROR" in capsys.readouterr().out and not (tmp_path / "w2" / "located.json").exists()
    deep = tmp_path / "deep"                                            # beyond depth 6 is not searched
    _touch(deep / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "index.jsonl")
    with pytest.raises(run_all.LocateError):
        run_all.locate_inputs(deep)
