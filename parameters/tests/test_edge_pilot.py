"""Pilot orchestrator: isolation of failures, report contents, expected-output checks."""
import json
import subprocess
from types import SimpleNamespace

from Codec.edge import pilot_checks
from parameters.edge import pilot


def fake_runner(plan):
    """plan: stage-name fragment -> (returncode, stdout, stderr, files_to_create)."""
    def run(argv, **kwargs):
        text = " ".join(argv)
        for key, (rc, out, err, files) in plan.items():
            if key in text:
                for f in files:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text("{}")
                return SimpleNamespace(returncode=rc, stdout=out, stderr=err)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def test_failure_in_one_stage_does_not_stop_the_others(tmp_path):
    stages = [{"name": "one", "argv": ["x", "stage_one"], "expect": []},
              {"name": "two", "argv": ["x", "stage_two"], "expect": []},
              {"name": "three", "argv": ["x", "stage_three"], "expect": []}]
    runner = fake_runner({"stage_two": (1, "", "Traceback boom", [])})
    report = pilot.run_pilot(tmp_path, 5, runner=runner, stages=stages, cwd=str(tmp_path))
    assert [s["passed"] for s in report["stages"]] == [True, False, True]
    assert report["all_passed"] is False
    assert "boom" in report["stages"][1]["stderr_tail"]


def test_missing_expected_output_fails_even_with_returncode_zero(tmp_path):
    target = tmp_path / "out" / "index.jsonl"
    stages = [{"name": "a", "argv": ["x", "stage_a"], "expect": [target]}]
    report = pilot.run_pilot(tmp_path, 5, runner=fake_runner({"stage_a": (0, "ok", "", [])}),
                             stages=stages, cwd=str(tmp_path))
    assert report["stages"][0]["passed"] is False and report["stages"][0]["missing_outputs"] == [str(target)]
    report = pilot.run_pilot(tmp_path, 5, runner=fake_runner({"stage_a": (0, "ok", "", [target])}),
                             stages=stages, cwd=str(tmp_path))
    assert report["all_passed"] is True


def test_timeout_and_crash_are_reported_not_raised(tmp_path):
    def boom(argv, **kw):
        if "slow" in argv:
            raise subprocess.TimeoutExpired(argv, 1)
        raise FileNotFoundError("no python")
    stages = [{"name": "slow", "argv": ["slow"], "expect": [], "timeout": 1},
              {"name": "crash", "argv": ["x"], "expect": []}]
    report = pilot.run_pilot(tmp_path, 5, runner=boom, stages=stages, cwd=str(tmp_path))
    assert [s["passed"] for s in report["stages"]] == [False, False]
    assert "TIMEOUT" in report["stages"][0]["stderr_tail"]


def test_report_files_written_and_default_stages_are_ordered(tmp_path):
    report = pilot.run_pilot(tmp_path, 3, runner=fake_runner({}), stages=[], cwd=str(tmp_path))
    assert (tmp_path / "pilot_report.json").exists() and (tmp_path / "PILOT_REPORT.md").exists()
    assert json.loads((tmp_path / "pilot_report.json").read_text())["per_source"] == 3
    names = [s["name"] for s in pilot.make_stages(tmp_path, 3)]
    data = ["a_download_images", "b_label", "c_feature_pass_smoke", "d_grid_export"]
    assert [n for n in names if not n.startswith("check_")] == data
    assert names[0] == "check_env_gpu" and names[1:5] == data
    assert names[5:] == ["check_" + n for n in pilot_checks.CHECK_ORDER[1:]]
    assert names.index("check_dataset_scale") > names.index("a_download_images")
    assert names[-1] == "check_feature_pass_smoke_verify"
    assert "--max-images" in pilot.make_stages(tmp_path, 3)[1]["argv"]
    assert "--no-publish" in pilot.make_stages(tmp_path, 3)[1]["argv"]
