"""SV-DPP header-relevance extension (decision D18): factor = 1 + beta * (r_tilde - 0.5).

beta = 0 must reproduce the plan v3 policy exactly. Only header fields are read (never Truth).
"""
import inspect
import re

import numpy as np

from Evaluation.edge.contracts import Header, Job, LatencyProfile, QueuedJob, State, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import sv_dpp as sv
from Scheduler.edge.sv_dpp import GenieSVDPP, SVDPP

SPEC = ValueSpec.from_config(load_config("sim"))
ONE = {k: tuple(1.0 if i == k else 0.0 for i in range(4)) for k in range(4)}
ONE_HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]
LAT = LatencyProfile({1: [0.8, 1.0, 1.2], 2: [1.2, 1.4, 1.6], 4: [2.0, 2.2, 2.4]})
RANGE = {0: (0.52, 0.68), 1: (0.52, 0.68), 2: (0.52, 0.68), 3: (0.52, 0.68)}


def hdr(p, mission=0, relevance=0.5, arrival=0.0):
    return Header(0.0, arrival, 1000, mission, 0, 0, tuple(p), relevance, 0, 0)


def qj(i, p, age=0.0, relevance=0.5, mission=0, arrival=0.0):
    return QueuedJob(i, hdr(p, mission, relevance, arrival), age, 0.0, 1.0)


def state(queue, lat=LAT, max_batch=1, workers=1, rate=0.0):
    return State(0.0, tuple(queue), 1, workers, lat, SPEC, max_batch, arrival_rate=rate)


def make(cls=SVDPP, **kw):
    pol = cls(**kw)
    pol.reset(SPEC, np.random.default_rng(0))
    return pol


def trace(n=60, seed=0, rate=1.2):
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=n))
    jobs = []
    for i, t in enumerate(arrivals):
        tier = int(rng.choice(4, p=[0.1, 0.2, 0.3, 0.4]))
        header = Header(capture_time=float(t), arrival_time=float(t), payload_bytes=500,
                        mission=int(rng.integers(4)), codec_id=1, m_index=2, tier_probs=ONE_HOT[tier],
                        relevance=float(rng.uniform(0.45, 0.75)), sem_key=int(rng.integers(0, 2**62)),
                        perc_key=int(rng.integers(0, 2**62)), sts=(0.1,) * 13)
        jobs.append(Job(i, header, Truth(tier=tier, fidelity=1, dup_group=None, event="e", source="test")))
    return jobs


def episode(policy):
    env = EdgeEnv(trace(), LAT, SPEC, workers=2, max_batch=4, receiver=Receiver(), seed=3)
    return [(r["id"], r["status"], r["finish_time"]) for r in env.run(policy).measured()]


class _FactorDisabled(SVDPP):
    """Pre-change behaviour: the relevance factor is never applied."""

    def _relevance_factor(self, job):
        return 1.0


def test_beta_zero_factor_is_exactly_one_for_any_header_relevance():
    pol = make(relevance_beta=0.0, relevance_range=RANGE)
    for mission in range(4):
        for r in (-1.0, 0.0, 0.3, 0.52, 0.6, 0.68, 1.0, 2.0):
            assert pol._relevance_factor(qj(0, ONE[0], relevance=r, mission=mission)) == 1.0


def test_beta_zero_decisions_match_factor_free_policy_on_episode():
    assert episode(make(SVDPP, relevance_beta=0.0)) == episode(make(_FactorDisabled, relevance_beta=0.0))


def test_beta_positive_changes_ranking_between_identical_jobs_by_relevance():
    # Identical except relevance; the low-relevance job is queued first, so only the factor can pick job 1.
    q = [qj(0, ONE[0], relevance=0.54), qj(1, ONE[0], relevance=0.66)]
    pol = make(relevance_beta=0.5, relevance_range=RANGE)
    assert pol.select(state(q)).batch == (1,)
    assert make(relevance_beta=0.0, relevance_range=RANGE).select(state(q)).batch == (0,)


def test_r_tilde_is_clipped_at_both_ends():
    pol = make(relevance_beta=0.5, relevance_range=RANGE)
    low = pol._relevance_factor(qj(0, ONE[0], relevance=0.52))
    high = pol._relevance_factor(qj(0, ONE[0], relevance=0.68))
    assert low == 1.0 - 0.5 * 0.5 and high == 1.0 + 0.5 * 0.5
    assert pol._relevance_factor(qj(0, ONE[0], relevance=-5.0)) == low
    assert pol._relevance_factor(qj(0, ONE[0], relevance=9.0)) == high
    assert abs(pol._relevance_factor(qj(0, ONE[0], relevance=0.60)) - 1.0) < 1e-12   # midpoint: r_tilde = 0.5


def test_unknown_mission_uses_mission_zero_range():
    rng_map = {0: (0.4, 0.6), 1: (0.1, 0.9)}
    pol = make(relevance_beta=0.5, relevance_range=rng_map)
    job = qj(0, ONE[0], relevance=0.6, mission=9)
    assert pol._relevance_factor(job) == pol._relevance_factor(qj(0, ONE[0], relevance=0.6, mission=0))
    assert pol._relevance_factor(job) == 1.0 + 0.5 * 0.5                    # r_tilde = 1 under range 0
    assert pol._relevance_factor(qj(0, ONE[0], relevance=0.6, mission=1)) == 1.0 + 0.5 * (0.5 / 0.8 - 0.5)


def test_relevance_factor_reads_header_only_and_genie_inherits_it():
    truth = {0: Truth(3, 1, None), 1: Truth(0, 1, None)}
    job = qj(0, ONE[3], relevance=0.66, mission=2)
    n = make(SVDPP, relevance_beta=0.5, relevance_range=RANGE)
    g = make(GenieSVDPP, truth=truth, relevance_beta=0.5, relevance_range=RANGE)
    assert g._relevance_factor(job) == n._relevance_factor(job)
    assert not hasattr(n, "truth") and not SVDPP.uses_truth
    assert not re.search(r"\btruth\b", inspect.getsource(SVDPP), re.IGNORECASE)   # no Truth access in source
