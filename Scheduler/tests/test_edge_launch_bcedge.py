"""BCEdge policy/trainer (tiny fake table, CPU), SemCover in the phase-2 notebooks, and the launcher in dry-run."""
import json
import os
import re

import numpy as np
import pytest

from Evaluation.edge import run_sweep, sweep_stages, tables
from Scheduler.edge import bcedge, registry
from Scheduler.tests.test_edge_train_cli_v3 import LAT, _jobs, _rows  # noqa: F401  (fixtures reused)
from Scheduler.tests.test_edge_dqn_ltr import make_env
from Scheduler.training.edge import learned_policy, train_cli
from scripts.edge import launch_phase2 as lp
from scripts.edge import make_scheduler_sweeps_notebook as sw
from scripts.edge import make_scheduler_train_notebook as tr


def _src(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


@pytest.fixture
def fake_tables(monkeypatch):
    monkeypatch.setattr(tables, "load_rows", lambda path: _rows())
    monkeypatch.setattr(tables, "latency_from_profile_json", lambda path: LAT)
    monkeypatch.setattr(tables, "rows_to_jobs", _jobs)


def test_bcedge_registered_grouped_and_header_only():
    from Evaluation.edge.edge_config import load_config
    assert "bcedge" in registry.all_policies() and "bcedge" in load_config("policies")["ext_batching"]
    assert registry.build("bcedge").uses_truth is False


def test_bcedge_trains_loads_and_runs_episode(fake_tables, tmp_path):
    out = str(tmp_path / "bc.pt")
    hist = train_cli.main(["--algo", "bcedge", "--table", "t", "--profile", "p", "--out", out, "--iters", "3",
                           "--episodes", "2", "--n-jobs", "30", "--rate", "1.5", "--hidden", "8", "--max-batch", "8"])
    assert os.path.getsize(out) > 0 and len(hist) == 3 and np.isfinite(hist[-1]["loss"])
    pol = learned_policy.build_learned("learned:bcedge:" + out)
    assert pol.uses_truth is False and run_sweep.learned_policy.is_learned("learned:bcedge:" + out)
    res = make_env(0).run(pol)
    assert res.outcomes and any(r["status"] == "decoded" for r in res.outcomes)


def test_bcedge_batch_order_is_tier_then_edf():
    env = make_env(1)
    pol = bcedge.BCEdge(record=True)          # untrained: largest feasible size
    env.run(pol)
    assert pol.records and all(len(r[3]) in (1, 2, 4) for r in pol.records)
    assert all(bcedge.ACTIONS[r[2]] >= len(r[3]) for r in pol.records)


def test_semcover_algo_and_semantic_group_in_notebooks():
    nb, meta = tr.build("semcover", step="edge-train-semcover")
    assert "'semcover'" in _src(nb) and meta["id"].endswith("/edge-train-semcover")
    assert "semcover" in sweep_stages.compare_policies()
    assert "bcedge" in tr.ALGOS


def test_sweep_learned_spec_and_sts_fusion_passthrough():
    nb, meta = sw.build("compare", policies=["semantic", "learned:semcover:semcover_seed0.pt"], sts_fusion=True,
                        kernel_sources=["satyapaladuguplus/edge-train-semcover"], step="edge-sweep-compare-x")
    src = _src(nb)
    assert "learned:semcover:semcover_seed0.pt" in src and 'find_input("sts_fusion.json")' in src
    assert meta["kernel_sources"] == ["satyapaladuguplus/edge-train-semcover"]
    compile("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"), "nb", "exec")


def test_launch_plan_split_and_limits():
    entries = lp.plan()
    assert [e.step for e in entries if e.kind == "train"] == [
        "edge-train-ppo", "edge-train-scope-ppo", "edge-train-bc", "edge-train-dagger", "edge-train-dqn",
        "edge-train-scope-dqn", "edge-train-ltr-lambdamart", "edge-train-ltr-ranknet", "edge-train-scope-ltr",
        "edge-train-semcover", "edge-train-bcedge"]
    assert [e.step for e in entries if e.kind == "sweep"] == ["edge-sweep-tune-0of2", "edge-sweep-tune-1of2"]
    per = {a: sum(e.account == a for e in entries) for a in ("wsl", "win")}
    assert abs(per["wsl"] - per["win"]) <= 1 and not any(e.gpu for e in entries)
    lp.check_limits(entries)


def test_launch_dry_run_builds_and_never_pushes(tmp_path):
    cmds = []

    def runner(cmd, env=None):
        cmds.append(cmd)
        return 0, ""

    lines = []
    cfg = lp.bn.load_kaggle_config()
    res = lp.main(["--out-dir", str(tmp_path)], runner=runner, cfg=cfg, out=lines.append)
    assert cmds == [] and len(res) == 13                      # dry run: nothing executed
    pushes = [l for l in lines if "kernels push" in l]
    assert len(pushes) == 13 and sum(l.startswith("$ wsl ") for l in pushes) == 7
    assert sum(l.startswith("$ kaggle ") for l in pushes) == 6
    for e in lp.plan():
        meta = json.loads((tmp_path / (e.step + ".kernel-metadata.json")).read_text())
        user = cfg["accounts"][e.account]["username"]
        assert meta["id"] == user + "/" + e.step and meta["enable_gpu"] is False
        assert re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") == e.step
        nb = json.loads((tmp_path / (e.step + ".ipynb")).read_text())
        assert ("train_cli" in _src(nb)) == (e.kind == "train") and not any(c.get("outputs") for c in nb["cells"])
    assert "'bcedge'" in _src(json.loads((tmp_path / "edge-train-bcedge.ipynb").read_text()))


def test_status_and_collect_dry_and_429(tmp_path):
    lines, sleeps = [], []
    cfg = lp.bn.load_kaggle_config()
    st = lp.main(["--status", "--dry-run", "--out-dir", str(tmp_path)], cfg=cfg, out=lines.append)
    assert len(st) == 13 and sum("kernels status" in l for l in lines) == 13
    calls = []

    def runner(cmd, env=None):
        calls.append(cmd)
        return (1, "429 Too Many Requests") if len(calls) == 1 else (0, "ok")

    la = lp.Launcher(cfg, runner, dry_run=False, out=lambda s: None, out_dir=tmp_path, sleep=sleeps.append)
    one = [e for e in lp.plan() if e.step == "edge-train-ppo"]
    folder = tmp_path / "got" / "edge-train-ppo"
    folder.mkdir(parents=True)
    (folder / "a.json").write_text("{}")
    (folder / "big.log").write_text("x")
    assert la.collect_all(one, tmp_path / "got") == {"edge-train-ppo": 0}
    assert sleeps == [60] and len(calls) == 2 and "--file-pattern" in calls[0]
    assert (folder / "a.json").exists() and not (folder / "big.log").exists()
