"""Trained learners as sweep policies: ``learned:<algo>:<checkpoint>`` (Scheduler.training.edge.learned_policy).

Tiny checkpoints come from train_cli on the same synthetic table as test_edge_train_cli_v3.py.
"""
import numpy as np
import pytest

from Evaluation.edge import run_sweep, tables
from Evaluation.edge.contracts import Job, Policy, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.tests.test_edge_sweep import FAKE, LAT as SWEEP_LAT, make_rows
from Scheduler.tests.test_edge_train_cli_v3 import LAT, _jobs, _rows
from Scheduler.training.edge import learned_policy as lp
from Scheduler.training.edge import train_cli


def _spec():
    return ValueSpec.from_config(load_config("sim"))


@pytest.fixture
def fake_tables(monkeypatch):
    monkeypatch.setattr(tables, "load_rows", lambda path: _rows())
    monkeypatch.setattr(tables, "latency_from_profile_json", lambda path: LAT)
    monkeypatch.setattr(tables, "rows_to_jobs", _jobs)


def train(tmp_path, algo, *extra):
    out = str(tmp_path / ("%s_%s.bin" % (algo, "_".join(extra).replace("-", ""))))
    train_cli.main(["--algo", algo, "--table", "t", "--profile", "p", "--out", out, "--teacher", "edf",
                    "--iters", "2", "--episodes", "2", "--n-jobs", "30", "--rate", "1.5", "--hidden", "8",
                    *extra])
    return out


def jobs_for(seed=0, n=40):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.exponential(1.0 / 1.5, size=n))
    return _jobs(_rows()[:n], times, rng)


def episode(policy, jobs, seed=0):
    return EdgeEnv(jobs, LAT, _spec(), 1, 4, Receiver(), seed).run(policy)


# ---- spec parsing -------------------------------------------------------------------------------

def test_parse_spec_accepts_known_algo_and_existing_file(tmp_path):
    ckpt = tmp_path / "m.bin"
    ckpt.write_bytes(b"x")
    assert lp.parse_spec("learned:dqn:" + str(ckpt)) == ("dqn", str(ckpt))
    assert lp.is_learned("learned:ppo:x") and not lp.is_learned("fifo")


@pytest.mark.parametrize("bad", ["learned:nope:{p}", "learned:dqn:", "learned:dqn", "learned:dqn:{missing}"])
def test_parse_spec_rejects_bad_specs(tmp_path, bad):
    with pytest.raises(ValueError):
        lp.parse_spec(bad.format(p=tmp_path, missing=tmp_path / "absent.bin"))


def test_resolve_policies_accepts_learned_names(tmp_path):
    ckpt = tmp_path / "m.bin"
    ckpt.write_bytes(b"x")
    name = "learned:dqn:" + str(ckpt)
    assert run_sweep.resolve_policies(["fifo", name]) == ["fifo", name]
    with pytest.raises(ValueError):
        run_sweep.resolve_policies(["learned:dqn:" + str(tmp_path / "absent.bin")])


# ---- training -> loading -> episode -------------------------------------------------------------

@pytest.mark.parametrize("algo,extra", [
    ("ppo", ()),
    ("dqn", ()),
    ("ltr", ("--ranker", "ranknet")),
    ("ltr", ("--ranker", "lambdamart", "--no-lightgbm")),
])
def test_trained_checkpoint_runs_a_header_only_episode(fake_tables, tmp_path, algo, extra):
    ckpt = train(tmp_path, algo, *extra)
    spec = "learned:%s:%s" % (algo, ckpt)
    policy = lp.build_learned(spec)
    assert policy.uses_truth is False
    jobs = jobs_for()
    res = episode(policy, jobs)
    assert len(res.outcomes) == len(jobs)
    assert any(o["status"] == "decoded" for o in res.outcomes)
    assert not any(o["status"] == "purged" for o in res.outcomes)   # learners order, they never purge


def test_scope_dqn_checkpoint_loads_as_dqn(fake_tables, tmp_path):
    ckpt = train(tmp_path, "scope_dqn")
    res = episode(lp.build_learned("learned:scope_dqn:" + ckpt), jobs_for())
    assert any(o["status"] == "decoded" for o in res.outcomes)


def test_wrong_family_fails_loudly(fake_tables, tmp_path):
    ckpt = train(tmp_path, "dqn")
    with pytest.raises(KeyError):
        lp.build_learned("learned:ppo:" + ckpt)   # a DQN blob has no SetPolicy "cfg"


# ---- truth-leak checks --------------------------------------------------------------------------

class _Poison:
    """Any access to the truth table is a test failure."""

    def _boom(self, *a, **k):
        raise AssertionError("learned policy must not read the truth table")

    __getitem__ = __iter__ = get = items = keys = values = __len__ = _boom


def test_build_never_touches_truth_table(fake_tables, tmp_path):
    ckpt = train(tmp_path, "dqn")
    pol = run_sweep._build_policy("learned:dqn:" + ckpt, {}, _Poison())
    assert pol.uses_truth is False
    with pytest.raises(ValueError):
        run_sweep._build_policy("learned:dqn:" + ckpt, {"V": 1.0}, _Poison())


class _Recorder(Policy):
    """Wraps a policy and logs what it saw and decided at every call."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def reset(self, spec, rng):
        self.inner.reset(spec, rng)

    def observe(self, event, payload):
        self.inner.observe(event, payload)

    def select(self, state):
        dec = self.inner.select(state)
        self.log.append((state.now, tuple(q.id for q in state.queue),
                         tuple(round(q.age, 9) for q in state.queue), dec.batch))
        return dec


def _scrambled(jobs):
    """Same headers and arrivals, different Truth (tier, fidelity, event, source)."""
    return [Job(j.id, j.header, Truth((j.truth.tier + 1) % 4, 1 - j.truth.fidelity, None, "x", "y"))
            for j in jobs]


@pytest.mark.parametrize("algo", ["dqn", "ltr"])
def test_decisions_do_not_depend_on_truth(fake_tables, tmp_path, algo):
    ckpt = train(tmp_path, algo, *(("--ranker", "ranknet") if algo == "ltr" else ()))
    spec = "learned:%s:%s" % (algo, ckpt)
    jobs = jobs_for()
    logs = []
    for variant in (jobs, _scrambled(jobs)):
        rec = _Recorder(run_sweep._build_policy(spec, {}, None))
        episode(rec, variant)
        logs.append(rec.log)
    assert logs[0] and logs[0] == logs[1]


# ---- sweep integration --------------------------------------------------------------------------

def test_run_sweep_and_stage_compare_accept_learned_policy(fake_tables, tmp_path):
    from types import SimpleNamespace
    from Evaluation.edge import sweep_stages
    ckpt = train(tmp_path, "dqn")
    spec = "learned:dqn:" + ckpt
    res = run_sweep.run_sweep(make_rows(), SWEEP_LAT, "test", [spec, "fifo"], [1], [1.0], 1, base_seed=3,
                              max_batch=2, n_boot=10, tables=FAKE)
    cells = {c["policy"]: c for c in res["cells"]}
    assert set(cells) == {spec, "fifo"} and np.isfinite(cells[spec]["ctr_ungated"])
    a = SimpleNamespace(max_batch=2, realisations=1, seed=0, n_boot=10, policies=[spec], workers=[1],
                        rhos=[1.0], tuned=None)
    out = sweep_stages.stage_compare(make_rows(80), SWEEP_LAT, a, (0, 1), FAKE)
    assert [c["policy"] for c in out["cells"]] == [spec]
