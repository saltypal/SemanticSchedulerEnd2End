"""Classic edge policies: ordering, batch clipping, determinism, id validity."""
import numpy as np
import pytest

from Evaluation.edge.contracts import Header, LatencyProfile, QueuedJob, State, ValueSpec
from Scheduler.edge.classic import POLICIES

SPEC = ValueSpec((10.0, 20.0, 40.0, 80.0), 2.0, (8.0, 4.0, 2.0, 1.0), (0.9, 0.8, 0.7, 0.6))
LAT = LatencyProfile({1: [1.0], 2: [1.5], 4: [2.0]})


def probs(k):
    p = [0.01] * 4
    p[k] = 0.97
    return tuple(p)


def job(i, tier=2, age=0.0, arrival=None, mission=0):
    h = Header(0.0, float(i) if arrival is None else arrival, 100, mission, 0, 0, probs(tier), 0.5, 0, 0)
    return QueuedJob(i, h, age)


def state(jobs, max_batch=1, served=()):
    return State(100.0, tuple(jobs), 1, 1, LAT, SPEC, max_batch, mission_served=tuple(served))


def run(name, st, seed=0):
    p = POLICIES[name]()
    p.reset(SPEC, np.random.default_rng(seed))
    return p, p.select(st)


CASES = [
    ("fifo", [job(0, arrival=5), job(1, arrival=1)], 1),
    ("edf", [job(0, tier=3, age=10), job(1, tier=0, age=5)], 1),
    ("tuf_dasa", [job(0, tier=3), job(1, tier=0)], 1),
    ("spi_first_edf", [job(0, tier=2, age=39), job(1, tier=0, age=0)], 1),
    ("priority_aging", [job(0, tier=3, age=0), job(1, tier=0, age=0)], 1),
    ("mlwdf", [job(0, tier=2, age=1), job(1, tier=2, age=30)], 1),
    ("llf", [job(0, tier=3, age=10), job(1, tier=0, age=9)], 1),
    ("lelf", [job(0, tier=3, age=10), job(1, tier=0, age=15)], 1),
    ("exp_rule", [job(0, tier=2, age=1), job(1, tier=2, age=30)], 1),
    ("dapf", [job(0, tier=3, age=1), job(1, tier=0, age=1)], 1),
    ("pf_mission", [job(0, mission=0), job(1, mission=1)], 1),
    ("gen_cmu", [job(0, tier=3, age=1), job(1, tier=0, age=1)], 1),
]


@pytest.mark.parametrize("name,jobs,first", CASES)
def test_first_job(name, jobs, first):
    served = [(0, 50.0), (1, 0.0)] if name == "pf_mission" else ()
    _, d = run(name, state(jobs, served=served))
    assert d.batch == (first,) and d.purge == ()


def test_registry_covers_all():
    assert len(POLICIES) == 14
    for f in POLICIES.values():
        assert f.uses_truth is False


def test_batch_clipping():
    jobs = [job(i) for i in range(3)]
    assert len(run("fifo", state(jobs, max_batch=4))[1].batch) == 2
    assert len(run("fifo", state(jobs[:1], max_batch=4))[1].batch) == 1
    assert len(run("fifo", state([job(i) for i in range(9)], max_batch=4))[1].batch) == 4
    assert len(run("fifo", state([job(i) for i in range(9)], max_batch=3))[1].batch) == 2


def test_random_deterministic():
    jobs = [job(i) for i in range(8)]
    a = run("random", state(jobs, max_batch=4), seed=3)[1]
    b = run("random", state(jobs, max_batch=4), seed=3)[1]
    assert a == b


def test_rr_mission_alternates():
    q = [job(0, mission=0), job(1, mission=0), job(2, mission=1), job(3, mission=1)]
    p = POLICIES["rr_mission"]()
    p.reset(SPEC, np.random.default_rng(0))
    seen = []
    for _ in range(4):
        d = p.select(state(q))
        seen.append(next(j.header.mission for j in q if j.id == d.batch[0]))
        q = [j for j in q if j.id not in d.batch]
    assert seen == [0, 1, 0, 1]


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_ids_valid_no_duplicates(name):
    jobs = [job(i, tier=i % 4, age=float(i), mission=i % 3) for i in range(10)]
    ids = {j.id for j in jobs}
    for mb in (1, 2, 4):
        _, d = run(name, state(jobs, max_batch=mb, served=[(0, 1.0)]))
        assert d.batch and set(d.batch) <= ids and len(set(d.batch)) == len(d.batch)
        assert d.purge == ()


def test_empty_queue_idle():
    assert run("fifo", state([]))[1].batch == ()
