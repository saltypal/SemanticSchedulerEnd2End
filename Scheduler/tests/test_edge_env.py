import dataclasses
import numpy as np
import pytest

from Evaluation.edge.arrivals import (build_jobs, compress_trace, mmpp_times, poisson_times,
                                      rate_for_rho)
from Evaluation.edge.contracts import (Decision, Header, Job, LatencyProfile, Policy, Truth,
                                       ValueSpec)
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.metrics import bootstrap_ci, ctr, false_purges, summarize, timely_by_tier
from Evaluation.edge.receiver import Receiver

SPEC = ValueSpec((5.0, 10.0, 20.0, 40.0), 2.0, (8.0, 4.0, 2.0, 1.0), (0.95, 0.85, 0.6, 0.2), 0.25)
LAT1 = LatencyProfile({1: [1.0], 2: [1.0]})


def header(t, sem=0, perc=0):
    return Header(t, t, 1000, 0, 0, 0, (0.25,) * 4, 0.5, sem, perc)


def job(i, t, tier=0, u=1, grp=None, sem=0, perc=0):
    return Job(i, header(t, sem, perc), Truth(tier, u, grp, "e%d" % (i % 2), "src"))


class Fifo(Policy):
    def __init__(self, b=1):
        self.b = b

    def select(self, state):
        ids = [q.id for q in state.queue]
        return Decision(tuple(ids[:self.b]) if len(ids) >= self.b else ())


class Bad(Policy):
    def __init__(self, dec):
        self.dec = dec

    def select(self, state):
        return self.dec


def env(jobs, lat=LAT1, workers=1, max_batch=2, seed=0, warm=0.0):
    return EdgeEnv(jobs, lat, SPEC, workers, max_batch, Receiver(), seed, warmup_fraction=warm)


def random_jobs(n, seed):
    rng = np.random.default_rng(seed)
    ts = poisson_times(n, 1.5, rng)
    return [job(i, float(t), int(rng.integers(4)), int(rng.integers(2)),
                int(rng.integers(8)) if rng.random() < 0.3 else None,
                int(rng.integers(1 << 62)), int(rng.integers(1 << 62)))
            for i, t in enumerate(ts)]


def test_conservation_every_job_ends_once():
    lat = LatencyProfile({1: [0.5, 1.0, 2.0], 2: [1.0, 3.0]})
    res = env(random_jobs(200, 1), lat, warm=0.1).run(Fifo(1))
    ids = [o["id"] for o in res.outcomes]
    assert len(ids) == len(set(ids)) == 200
    assert {o["status"] for o in res.outcomes} <= {"decoded", "purged", "expired", "queued_at_end"}
    assert len(res.measured()) == 180


def test_determinism_by_seed():
    lat = LatencyProfile({1: [0.5, 1.0, 2.0], 2: [1.0, 3.0]})
    a = env(random_jobs(100, 2), lat, seed=5).run(Fifo(1))
    b = env(random_jobs(100, 2), lat, seed=5).run(Fifo(1))
    c = env(random_jobs(100, 2), lat, seed=6).run(Fifo(1))
    assert a.outcomes == b.outcomes
    assert a.outcomes != c.outcomes


def test_hand_computed_toy_trace():
    jobs = [job(0, 0.0), job(1, 0.5, tier=1), job(2, 0.6, tier=3)]
    res = env(jobs).run(Fifo(1))
    o = {r["id"]: r for r in res.outcomes}
    assert [o[i]["finish_time"] for i in range(3)] == [1.0, 2.0, 3.0]
    assert o[1]["delay"] == pytest.approx(1.5) and o[1]["wait_start"] == 1.0
    assert o[0]["value"] == 8.0 and o[1]["value"] == 4.0 and o[2]["value"] == 1.0
    assert res.busy_time == 3.0


def test_batch_of_two_finishes_together():
    jobs = [job(0, 0.0, tier=1), job(1, 0.0, tier=2)]
    res = env(jobs).run(Fifo(2))
    assert [r["finish_time"] for r in res.outcomes] == [1.0, 1.0]
    assert all(r["batch_size"] == 2 for r in res.outcomes)
    assert res.busy_time == 1.0


@pytest.mark.parametrize("dec", [
    Decision((99,)), Decision((0, 0)), Decision((0,), (0,)), Decision((), (42,)),
])
def test_invalid_decision_raises(dec):
    with pytest.raises(ValueError):
        env([job(0, 0.0), job(1, 0.0)]).run(Bad(dec))


def test_invalid_batch_size_raises():
    jobs = [job(i, 0.0) for i in range(3)]
    with pytest.raises(ValueError):   # 3 not profiled and > max_batch
        env(jobs, max_batch=3).run(Bad(Decision((0, 1, 2))))


def test_without_replacement_arrivals():
    rows = list(range(5))
    mk = lambda r, t, c: dataclasses.replace(header(t), capture_time=c)
    tr = lambda r: Truth(0, 1, None)
    with pytest.raises(ValueError):
        build_jobs(rows, np.arange(6.0), mk, tr, np.random.default_rng(0))
    jobs = build_jobs(rows, [3.0, 1.0, 2.0], mk, tr, np.random.default_rng(0), network_delay=0.5)
    assert len({j.id for j in jobs}) == 3
    assert [j.header.arrival_time for j in jobs] == [1.0, 2.0, 3.0]
    assert jobs[0].header.capture_time == 0.5


def test_duplicate_value_rule():
    jobs = [job(0, 0.0, tier=1, grp=7), job(1, 0.1, tier=1, grp=7),
            job(2, 0.2, tier=0, grp=9), job(3, 0.3, tier=0, grp=9)]
    res = env(jobs).run(Fifo(1))
    o = {r["id"]: r for r in res.outcomes}
    assert o[0]["value"] == 4.0 and not o[0]["dup_flag"]
    assert o[1]["value"] == 0.0 and o[1]["dup_flag"]
    assert o[2]["value"] == 8.0
    assert o[3]["value"] == pytest.approx(8.0 * 0.25) and o[3]["dup_flag"]


def test_rate_for_rho_round_trip():
    lat = LatencyProfile({4: [1.0, 3.0]})
    lam = rate_for_rho(0.8, lat, 4, 2)
    assert lam * lat.mean(4) / (4 * 2) == pytest.approx(0.8)
    rng = np.random.default_rng(0)
    t = poisson_times(20000, lam, rng)
    assert len(t) / t[-1] == pytest.approx(lam, rel=0.05)
    m = mmpp_times(500, (1.0, 5.0), 0.5, rng)
    assert len(m) == 500 and np.all(np.diff(m) > 0)
    assert compress_trace([10.0, 30.0], 2.0).tolist() == [0.0, 10.0]


def test_bootstrap_ci_contains_mean():
    rows = [{"x": 1.0 + 0.01 * (i % 5), "g": i % 4} for i in range(60)]
    stat = lambda rs: float(np.mean([r["x"] for r in rs]))
    mean = stat(rows)
    for grp in (None, "g"):
        lo, hi = bootstrap_ci(rows, stat, 200, np.random.default_rng(0), group=grp)
        assert lo <= mean <= hi


def test_receiver_memory_only_delivered():
    rec = Receiver(size=2)
    h = header(0.0, sem=0b1111, perc=0b1111)
    assert rec.dup_prob(h) == 0.0
    e = EdgeEnv([job(0, 0.0, sem=5, perc=5), job(1, 0.1, sem=6, perc=6)], LAT1, SPEC, 1, 1, rec, 0,
                warmup_fraction=0.0)
    e.run(_PurgeSecond())
    assert list(rec.memory) == [(5, 5)]            # purged job 1 never enters memory
    assert rec.dup_prob(header(0.0, 5, 5)) > 0.98
    assert rec.dup_prob(header(0.0, 5 ^ ((1 << 64) - 1), 5)) < rec.dup_prob(header(0.0, 5, 5))
    rec.deliver(header(0.0, 1, 1))
    rec.deliver(header(0.0, 2, 2))
    assert len(rec.memory) == 2                    # ring size respected


class _PurgeSecond(Policy):
    def select(self, state):
        ids = [q.id for q in state.queue]
        if 1 in ids:
            return Decision(purge=(1,))
        return Decision(tuple(ids[:1]))


def test_metrics_basic():
    jobs = [job(0, 0.0, tier=0, u=0), job(1, 0.0, tier=0, u=1), job(2, 0.0, tier=0)]
    res = env(jobs).run(_PurgeLast())
    assert ctr(res, False) == pytest.approx(2 / 3) and ctr(res, True) == pytest.approx(1 / 3)
    assert false_purges(res)["count"] == 1
    assert timely_by_tier(res)[0]["n"] == 3
    s = summarize(res)
    assert s["decision_time"]["n"] > 0 and s["by_source"]["src"]["n"] == 3


class _PurgeLast(Policy):
    def select(self, state):
        ids = [q.id for q in state.queue]
        if 2 in ids:
            return Decision(purge=(2,))
        return Decision(tuple(ids[:1]))
