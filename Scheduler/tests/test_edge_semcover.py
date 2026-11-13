"""SemCover: kernel, conditional-DPP greedy MAP, facility objective, analytic policy, training and loading."""
import itertools
import math

import numpy as np
import pytest
import torch

from Evaluation.edge import tables
from Evaluation.edge.contracts import Header, QueuedJob, State, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry, semcover as sc
from Scheduler.tests.test_edge_train_cli_v3 import LAT, _jobs, _rows
from Scheduler.training.edge import learned_policy as lp
from Scheduler.training.edge import semcover_train as st
from Scheduler.training.edge import train_cli

SPEC = ValueSpec.from_config(load_config("sim"))
STS = (1.0, 0, 0, 0, 0, 0, 0) + (0.0,) * 6
STS2 = (0, 1.0, 0, 0, 0, 0, 0) + (0.0,) * 6
CRIT, MED = (0.9, 0.1, 0, 0), (0, 0, 1.0, 0)


def hdr(sem, perc, p, sts=STS, cap=0.0):
    return Header(cap, cap, 500, 0, 1, 2, p, 0.5, sem, perc, sts)


def comps_for(headers):
    return sc.pair_components(headers)


def test_kernel_is_psd_and_floor_reduces_critical_pairs():
    hs = [hdr(0, 0, CRIT), hdr(1, 1, CRIT), hdr(2 ** 40, 2 ** 50, MED, STS2), hdr(2 ** 63, 7, MED)]
    comps, crit = comps_for(hs)
    w = torch.ones(3, dtype=sc.DT)
    S = sc.build_kernel(comps, crit, w, 0.25, 1e-3)
    assert torch.linalg.eigvalsh(S).min() > 0
    raw = sc.build_kernel(comps, crit, w, 0.0, 1e-3)
    assert S[0, 1] < raw[0, 1] * 0.85 and abs(S[2, 3] - raw[2, 3]) < 0.05   # only Critical pairs reduced


def test_greedy_map_diverse_and_memory_suppresses():
    a, a2, b = hdr(0, 0, CRIT, cap=0), hdr(0, 0, MED), hdr(2 ** 32 - 1, 2 ** 60, MED, STS2)
    hs = [hdr(0, 0, MED), hdr(0, 1, MED), hdr(2 ** 32 - 1, 2 ** 60, MED, STS2)]
    comps, crit = comps_for(hs)
    S = sc.build_kernel(comps, crit, torch.ones(3, dtype=sc.DT), 0.25, 1e-3)
    q = torch.tensor([1.0, 0.99, 0.6], dtype=sc.DT)
    L = q[:, None] * S * q[None, :]
    assert sc.greedy_map(L, 2, torch.ones(3, dtype=torch.bool)) == [0, 2]   # near-copy 1 has tiny marginal gain
    # memory holding a copy of job 2's scene suppresses it
    mem = hdr(2 ** 32 - 1, 2 ** 60, MED, STS2)
    comps, crit = comps_for([mem] + hs)
    Sc = sc.build_cond_kernel(comps, crit, torch.tensor([1.0], dtype=sc.DT), torch.ones(3, dtype=sc.DT), 0.25, 1e-3)
    assert Sc[2, 2] < 0.05 and Sc[0, 0] > 0.3
    old = sc.build_cond_kernel(comps, crit, torch.tensor([0.0], dtype=sc.DT), torch.ones(3, dtype=sc.DT), 0.25, 1e-3)
    assert old[2, 2] > 0.9                                                   # fully decayed memory: no suppression


def test_facility_monotone_and_within_1_minus_1_over_e():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 7
        X = torch.as_tensor(np.abs(rng.normal(size=(n, 5))), dtype=sc.DT)
        Sn = (X / X.norm(dim=1, keepdim=True)) @ (X / X.norm(dim=1, keepdim=True)).T
        q, base = torch.as_tensor(rng.uniform(0.2, 1, n), dtype=sc.DT), torch.zeros(n, dtype=sc.DT)
        val = lambda b: sc.facility_value(Sn, q, base, list(b))
        assert all(val(b + (j,)) >= val(b) - 1e-12 for b in [(0,), (1, 2)] for j in range(n) if j not in b)
        g = sc.greedy_facility(Sn, q, base, 3, torch.ones(n, dtype=torch.bool))
        best = max(val(c) for c in itertools.combinations(range(n), 3))
        assert val(g) >= (1 - 1 / math.e) * best - 1e-9


def test_critical_floor_keeps_corroborating_copy_selectable():
    hs = [hdr(0, 0, CRIT), hdr(0, 0, CRIT), hdr(0, 0, MED)]
    comps, crit = comps_for(hs)
    w = torch.ones(3, dtype=sc.DT)
    q = torch.ones(3, dtype=sc.DT)
    gain = lambda z: float((lambda L: L[1, 1] - L[0, 1] ** 2 / L[0, 0])(q[:, None] * sc.build_kernel(comps, crit, w, z, 1e-3) * q[None]))
    assert gain(0.25) > 0.3 and gain(0.25) > 10 * gain(0.0)                    # Critical copy keeps real gain


def _qjobs(hs, ages):
    return tuple(QueuedJob(i, h, a) for i, (h, a) in enumerate(zip(hs, ages)))


def test_analytic_policy_state_decision_and_no_truth():
    hs = [hdr(i * 3, i * 5, CRIT if i < 2 else MED, cap=0) for i in range(6)]
    st_ = State(10.0, _qjobs(hs, [1.0] * 6), 1, 1, LAT, SPEC, 4)
    for obj in ("logdet", "facility"):
        pol = registry.build("semcover", objective=obj)
        pol.reset(SPEC, np.random.default_rng(0))
        d = pol.select(st_)
        assert len(set(d.batch)) == len(d.batch) and len(d.batch) in LAT.batch_sizes
    assert registry.build("semcover").uses_truth is False
    assert "semcover" in registry.all_policies()
    import inspect
    assert "truth" not in inspect.signature(sc.SemCover.__init__).parameters


def test_analytic_runs_episode_and_memory_fills():
    rng = np.random.default_rng(0)
    jobs = _jobs(_rows()[:40], np.cumsum(rng.exponential(1 / 1.5, size=40)), rng)
    pol = sc.SemCover()
    res = EdgeEnv(jobs, LAT, SPEC, 1, 4, Receiver(), 0).run(pol)
    assert any(o["status"] == "decoded" for o in res.outcomes) and len(pol.memory) > 0


def test_mono_urgency_non_decreasing():
    net = st.SemCoverNet(8)
    x = torch.rand(20, 3)
    base = net.mono(x)
    for c in range(3):
        y = x.clone()
        y[:, c] += 0.3
        assert (net.mono(y) >= base - 1e-6).all()


@pytest.fixture
def fake_tables(monkeypatch):
    monkeypatch.setattr(tables, "load_rows", lambda path: _rows())
    monkeypatch.setattr(tables, "latency_from_profile_json", lambda path: LAT)
    monkeypatch.setattr(tables, "rows_to_jobs", _jobs)


def test_train_cli_semcover_checkpoint_loads_and_runs(fake_tables, tmp_path):
    out = str(tmp_path / "sc.bin")
    hist = train_cli.main(["--algo", "semcover", "--table", "t", "--profile", "p", "--out", out, "--teacher", "edf",
                           "--iters", "1", "--episodes", "1", "--n-jobs", "20", "--rate", "1.5", "--hidden", "8"])
    assert hist["stage1"] and hist["stage2"]
    pol = lp.build_learned("learned:semcover:" + out)
    assert pol.uses_truth is False
    rng = np.random.default_rng(1)
    jobs = _jobs(_rows()[:30], np.cumsum(rng.exponential(1 / 1.5, size=30)), rng)
    res = EdgeEnv(jobs, LAT, SPEC, 1, 4, Receiver(), 0).run(pol)
    assert any(o["status"] == "decoded" for o in res.outcomes)
