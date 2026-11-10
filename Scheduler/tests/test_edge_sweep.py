"""Sweep runner on synthetic in-memory tables (a tiny fake ``tables`` module is injected)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from Evaluation.edge import arrivals, run_sweep
from Evaluation.edge.contracts import Header, LatencyProfile, Truth

LAT = LatencyProfile({1: [0.9, 1.0, 1.1], 2: [1.5, 1.6, 1.7]})
HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def _header(row, arrival, capture):
    return Header(capture, arrival, 500, row["mission"], 1, 2, HOT[row["tier"]], 0.5,
                  row["image_id"], row["image_id"] * 7, ())


def _truth(row):
    return Truth(row["tier"], row["fidelity"], row["dup_group"], row["event"], "s")


FAKE = SimpleNamespace(
    rows_to_jobs=lambda rows, times, rng, network_delay=0.0:
        arrivals.build_jobs(rows, times, _header, _truth, rng, network_delay))


def make_rows(n=60):
    return [dict(image_id=i, tier=i % 4, mission=i % 3, fidelity=int(i % 5 != 0), dup_group=None,
                 event="e", split="test" if i < n - 10 else "val") for i in range(n)]


def sweep(**kw):
    args = dict(rows=make_rows(), latency=LAT, split="test", policies=["fifo", "edf"],
                workers=[1, 2], rhos=[0.8, 1.2], realisations=2, base_seed=5, max_batch=2,
                n_boot=20, tables=FAKE)
    args.update(kw)
    return run_sweep.run_sweep(**args)


def test_one_entry_per_cell_with_finite_metrics():
    res = sweep()
    assert len(res["cells"]) == 2 * 2 * 2
    keys = {(c["policy"], c["workers"], c["rho"]) for c in res["cells"]}
    assert len(keys) == 8
    for c in res["cells"]:
        assert math.isfinite(c["ctr_gated"]) and math.isfinite(c["value"])
        lo, hi = c["ci95"]["ctr_gated"]
        assert math.isfinite(lo) and lo <= hi
    assert res["config"]["n_images"] == 50


def test_same_seed_identical_and_different_seed_differs():
    strip = lambda r: [{k: v for k, v in c.items() if not k.startswith("decision") and k != "seconds"}
                       for c in r["cells"]]
    assert strip(sweep()) == strip(sweep())
    assert strip(sweep()) != strip(sweep(base_seed=6))


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        sweep(policies=["no_such_policy"])


def test_group_names_and_genie_resolve():
    names = run_sweep.resolve_policies(["references", "equivalence_appendix"])
    assert "genie_sv_dpp" in names and "llf" in names
    res = sweep(policies=["genie_sv_dpp"], workers=[1], rhos=[1.0], realisations=1)
    assert len(res["cells"]) == 1


def test_val_tuning_records_grid(tmp_path):
    out = run_sweep.val_tuning(make_rows(), LAT, [0.5, 2.0], workers=1, rho=1.0, realisations=1,
                               max_batch=2, n_boot=10, split="val", tables=FAKE)
    assert [p["kwargs"]["V"] for p in out["grid"]] == [0.5, 2.0]
    assert out["best_V"] in (0.5, 2.0)
    run_sweep.save_results(sweep(), tmp_path / "r.json")
    assert (tmp_path / "r.json").stat().st_size > 0
