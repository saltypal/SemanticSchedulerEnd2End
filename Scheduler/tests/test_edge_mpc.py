"""MPC rollout baseline: valid decisions, determinism, and a crafted overload where it beats FIFO."""
import numpy as np

from Evaluation.edge import metrics
from Evaluation.edge.contracts import Header, Job, LatencyProfile, QueuedJob, State, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry
from Scheduler.edge.mpc import POLICIES, MPCRollout

SPEC = ValueSpec.from_config(load_config("sim"))
LAT = LatencyProfile({1: [0.9, 1.0, 1.1], 2: [1.5, 1.6, 1.7]})
HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def header(t, tier):
    return Header(t, t, 500, 0, 1, 2, HOT[tier], 0.5, 1, 1, ())


def overload_jobs(n=80, rate=1.5):
    jobs = []
    for i in range(n):
        t = i / rate
        tier = 0 if i % 5 == 4 else 3
        jobs.append(Job(i, header(t, tier), Truth(tier, 1, None, "e", "s")))
    return jobs


def run(policy, workers=1):
    env = EdgeEnv(overload_jobs(), LAT, SPEC, workers, 1, Receiver(), seed=1)
    return env.run(policy)


def make_state(n=40):
    q = tuple(QueuedJob(i, header(0.0, i % 4), float(i % 7), 0.0, 1.0) for i in range(n))
    return State(10.0, q, 1, 2, LAT, SPEC, 2, arrival_rate=0.5)


def test_registered():
    assert POLICIES["mpc_rollout"] is MPCRollout
    assert "mpc_rollout" in registry.all_policies()
    assert MPCRollout.uses_truth is False


def test_valid_decision_and_queue_cap():
    p = MPCRollout(K=4, horizon=3, max_queue_considered=8)
    p.reset(SPEC, np.random.default_rng(0))
    st = make_state()
    d = p.select(st)
    ids = {j.id for j in st.queue}
    assert d.batch and len(set(d.batch)) == len(d.batch) and set(d.batch) <= ids
    assert len(d.batch) in LAT.batch_sizes and len(d.batch) <= st.max_batch
    assert p.select(st) == d


def test_empty_queue_is_idle():
    p = MPCRollout()
    p.reset(SPEC, np.random.default_rng(0))
    st = make_state(0)
    assert p.select(st).batch == ()


def test_beats_fifo_on_critical_ctr():
    mpc = metrics.ctr(run(MPCRollout()), False)
    fifo = metrics.ctr(run(registry.build("fifo")), False)
    assert mpc > fifo


def test_bad_args():
    import pytest
    with pytest.raises(ValueError):
        MPCRollout(K=0)
