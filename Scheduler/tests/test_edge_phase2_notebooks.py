import json
import re

import pytest

from Evaluation.edge import sweep_stages
from scripts.edge import make_live_t4_notebook as live
from scripts.edge import make_scheduler_sweeps_notebook as sw
from scripts.edge import make_scheduler_train_notebook as tr


def _src(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _slug_ok(meta):
    return re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") == meta["id"].split("/")[1]


@pytest.mark.parametrize("mod,step", [(tr, "edge-scheduler-train"), (sw, "edge-scheduler-sweeps"), (live, "edge-live-t4")])
def test_ids_titles_and_accounts(mod, step):
    kw = {"stage": "compare"} if mod is sw else ({"algo": "ppo"} if mod is tr else {})
    nb, meta = mod.build(**kw)
    assert meta["id"] == "satyapaladuguplus/" + step and _slug_ok(meta) and meta["code_file"] == step + ".ipynb"
    assert mod.build(account="win", **kw)[1]["id"] == "satyapaladugu/" + step
    src = _src(nb)
    assert "rev-parse" in src and "run_manifest.json" in src and "find_input" in src
    assert "PRIVATE" in src and not any(c["outputs"] for c in nb["cells"] if c["cell_type"] == "code")
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))


def test_dataset_sources():
    assert tr.build("ppo")[1]["dataset_sources"] == ["satyapaladuguplus/scheduler-dataset-v1-simtable"]
    assert sw.build("tune")[1]["dataset_sources"] == ["satyapaladuguplus/scheduler-dataset-v1-simtable"]
    ds = live.build()[1]
    assert "satyapaladugu/scheduler-sqgan-v1" in ds["dataset_sources"]
    assert ds["enable_gpu"] and ds["machine_shape"] == "NvidiaTeslaT4"


@pytest.mark.parametrize("algo", tr.ALGOS)
def test_train_algo_parameter(algo):
    nb, meta = tr.build(algo, seeds=5)
    src = _src(nb)
    assert "ALGO, SEEDS" in src and repr(algo) in src and "[0, 1, 2, 3, 4]" in src
    assert "Scheduler.training.edge.train_cli" in src and meta["enable_gpu"] is False
    json.dumps(nb)


def test_train_rejects_unknown_algo_and_ltr_ranker():
    with pytest.raises(ValueError):
        tr.build("nope")
    assert "ranknet" in _src(tr.build("ltr", ranker="ranknet")[0])


def test_sweeps_stage_rules():
    assert "1/2" in _src(sw.build("compare", shard="1/2")[0])
    with pytest.raises(ValueError):
        sw.build("test")
    with pytest.raises(ValueError):
        sw.build("test", policies=["fifo"])
    with pytest.raises(ValueError):
        sw.build("compare", final_test=True)
    assert "--final-test" in _src(sw.build("test", policies=["fifo", "sv_dpp"], final_test=True)[0])
    with pytest.raises(ValueError):
        sw.build("bogus")


def test_cli_writes_files(tmp_path):
    tr.main(["--algo", "dqn", "--account", "win", "--out-dir", str(tmp_path)])
    meta = json.loads((tmp_path / "edge-scheduler-train.kernel-metadata.json").read_text())
    assert meta["id"] == "satyapaladugu/edge-scheduler-train"


def test_shard_helpers(tmp_path):
    assert sweep_stages.parse_shard("") == (0, 1) and sweep_stages.parse_shard("1/3") == (1, 3)
    with pytest.raises(ValueError):
        sweep_stages.parse_shard("3/3")
    assert sweep_stages.take_shard(range(7), (1, 3)) == [1, 4]
    assert sweep_stages.out_path(tmp_path, "compare", (1, 2)).name == "compare.shard1of2.json"
    assert sweep_stages.out_path(tmp_path, "tune", (0, 1)).name == "tune.json"


def test_test_stage_needs_flag():
    args = type("A", (), {"final_test": False, "policies": ["fifo"]})()
    with pytest.raises(SystemExit):
        sweep_stages.stage_test([], None, args, (0, 1), None)


def test_tune_and_compare_stages_on_synthetic_table():
    from types import SimpleNamespace
    from Scheduler.tests.test_edge_sweep import FAKE, LAT, make_rows
    a = SimpleNamespace(max_batch=2, realisations=1, seed=0, n_boot=10, learner_grid=None, out_dir=".",
                        policies=["fifo", "edf"], workers=[1], rhos=[1.0], tuned=None)
    rows = make_rows(80)
    tune = sweep_stages.stage_tune(rows, LAT, a, (0, 4), FAKE)
    assert len(tune["sv_dpp_grid"]) == 5 and tune["best_kwargs"]["V"] in sweep_stages.V_GRID
    cmp0 = sweep_stages.stage_compare(rows, LAT, a, (0, 2), FAKE)
    cmp1 = sweep_stages.stage_compare(rows, LAT, a, (1, 2), FAKE)
    assert [c["policy"] for c in cmp0["cells"]] + [c["policy"] for c in cmp1["cells"]] == ["fifo", "edf"]


def test_ablations_fail_loudly_when_module_missing():
    import importlib.util
    if importlib.util.find_spec("Evaluation.edge.header_proofs") is not None:
        pytest.skip("header_proofs present")
    with pytest.raises(SystemExit, match="ABLATIONS NOT AVAILABLE"):
        sweep_stages.stage_ablations(type("A", (), {"out_dir": "."})(), (0, 1))
