"""train_cli trains every v3 learner on a tiny synthetic table (CPU, seconds) and saves weights."""
import os

import pytest

from Evaluation.edge import arrivals, tables
from Evaluation.edge.contracts import Header, LatencyProfile, Truth
from Scheduler.training.edge import ltr, train_cli
from Scheduler.training.edge.policy_net import load_model

LAT = LatencyProfile({1: [0.9, 1.0, 1.1], 2: [1.5, 1.6, 1.7], 4: [2.0, 2.2, 2.4]})
HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def _rows():
    return [dict(image_id=i, tier=i % 4, mission=i % 3, fidelity=int(i % 5 != 0), dup_group=None,
                 event="e", split="train") for i in range(40)]


def _jobs(rows, times, rng, network_delay=0.0):
    hdr = lambda r, a, c: Header(c, a, 500, r["mission"], 1, 2, HOT[r["tier"]], 0.5, r["image_id"],
                                 r["image_id"] * 7, (0.1,) * 13)
    tr = lambda r: Truth(r["tier"], r["fidelity"], r["dup_group"], r["event"], "s")
    return arrivals.build_jobs(rows, times, hdr, tr, rng, network_delay)


@pytest.fixture
def fake_tables(monkeypatch):
    monkeypatch.setattr(tables, "load_rows", lambda path: _rows())
    monkeypatch.setattr(tables, "latency_from_profile_json", lambda path: LAT)
    monkeypatch.setattr(tables, "rows_to_jobs", _jobs)


def cli(tmp_path, algo, *extra):
    out = str(tmp_path / (algo + ".bin"))
    hist = train_cli.main(["--algo", algo, "--table", "t", "--profile", "p", "--out", out, "--teacher", "edf",
                           "--iters", "2", "--episodes", "2", "--n-jobs", "30", "--rate", "1.5",
                           "--hidden", "8", *extra])
    assert os.path.getsize(out) > 0 and hist
    return out, hist


@pytest.mark.parametrize("algo", ["dqn", "scope_dqn"])
def test_dqn_family_trains_and_saves(fake_tables, tmp_path, algo):
    import torch
    out, hist = cli(tmp_path, algo)
    blob = torch.load(out)
    assert blob["hidden"] == 8 and blob["state"]


@pytest.mark.parametrize("ranker", ["lambdamart", "ranknet"])
def test_ltr_trains_and_saves(fake_tables, tmp_path, ranker):
    out, hist = cli(tmp_path, "ltr", "--ranker", ranker, "--no-lightgbm")
    cls = ltr.RankNet if ranker == "ranknet" else ltr.TreeRanker
    assert isinstance(cls.load(out), cls) and hist[0]["n_rows"] > 0


def test_scope_ltr_trains_and_saves(fake_tables, tmp_path):
    out, hist = cli(tmp_path, "scope_ltr")
    assert ltr.RankNet.load(out).in_dim > 0 and len(hist[0]["lambdas"]) == 4


def test_existing_algos_still_work(fake_tables, tmp_path):
    out, _ = cli(tmp_path, "bc")
    assert load_model(out) is not None
