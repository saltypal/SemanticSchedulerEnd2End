"""Every registered policy runs end to end on the simulator with a synthetic trace."""
import numpy as np
import pytest

from Evaluation.edge import metrics
from Evaluation.edge.contracts import Header, Job, LatencyProfile, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry

SPEC = ValueSpec.from_config(load_config("sim"))
LATENCY = LatencyProfile({1: [0.8, 1.0, 1.2], 2: [1.2, 1.4, 1.6], 4: [2.0, 2.2, 2.4]})
ONE_HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def make_jobs(n=120, seed=0, rate=1.2):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1.0 / rate, size=n))
    jobs = []
    for i, arrival in enumerate(t):
        tier = int(rng.choice(4, p=[0.1, 0.2, 0.3, 0.4]))
        header = Header(capture_time=float(arrival), arrival_time=float(arrival), payload_bytes=500,
                        mission=int(rng.integers(3)), codec_id=1, m_index=2, tier_probs=ONE_HOT[tier],
                        relevance=0.5, sem_key=int(rng.integers(0, 2**62)), perc_key=int(rng.integers(0, 2**62)),
                        sts=(0.1,) * 13)
        group = i // 2 if i % 7 == 0 else None
        jobs.append(Job(i, header, Truth(tier=tier, fidelity=int(rng.random() < 0.8), dup_group=group,
                                         event="e", source="crisismmd")))
    return jobs


def run(policy, workers=2, seed=3):
    env = EdgeEnv(make_jobs(), LATENCY, SPEC, workers=workers, max_batch=4, receiver=Receiver(), seed=seed)
    return env.run(policy)


def policy_names():
    names = [n for n in registry.all_policies() if n not in ("genie_sv_dpp", "mpc_rollout")]
    assert names, "no policies registered"
    return names


@pytest.mark.parametrize("name", policy_names())
def test_policy_runs_and_conserves_jobs(name):
    result = run(registry.build(name))
    rows = result.measured()
    assert rows, "no measured jobs"
    assert {r["status"] for r in rows} <= {"decoded", "purged", "expired", "queued_at_end"}
    assert len({r["id"] for r in rows}) == len(rows)
    summary = metrics.summarize(result)
    assert summary is not None


def test_same_seed_is_reproducible():
    a = run(registry.build("fifo"))
    b = run(registry.build("fifo"))
    assert [(r["id"], r["status"], r["finish_time"]) for r in a.measured()] == \
           [(r["id"], r["status"], r["finish_time"]) for r in b.measured()]


def test_priority_beats_fifo_on_critical_when_overloaded():
    jobs_rate = 3.0
    def critical_ctr(name):
        env = EdgeEnv(make_jobs(300, 1, jobs_rate), LATENCY, SPEC, workers=1, max_batch=1,
                      receiver=Receiver(), seed=1)
        return metrics.ctr(env.run(registry.build(name)), gated=False)
    assert critical_ctr("spi_first_edf") >= critical_ctr("fifo")


def test_all_configured_policies_exist_or_are_known_pending():
    pending = set(registry.missing_configured())
    assert pending <= {"mpc_rollout"}, "configured but unimplemented: %s" % sorted(pending)


@pytest.mark.parametrize("name", policy_names())
def test_every_policy_returns_profiled_batch_size_for_odd_queue(name):
    from Evaluation.edge.contracts import QueuedJob, State
    jobs = make_jobs(3)
    queue = tuple(QueuedJob(j.id, j.header, 0.5) for j in jobs)
    pol = registry.build(name)
    pol.reset(SPEC, np.random.default_rng(0))
    st = State(1.0, queue, 1, 2, LATENCY, SPEC, 4)
    d = pol.select(st)
    assert not d.batch or len(d.batch) in LATENCY.batch_sizes
