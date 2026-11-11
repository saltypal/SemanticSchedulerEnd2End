"""P0 extended policies: valid in a short env episode, header-only, and one behaviour check each."""
import numpy as np
import pytest

from Evaluation.edge.contracts import Header, Job, LatencyProfile, QueuedJob, State, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry
from Scheduler.edge.batching import ClipperAimd

SPEC = ValueSpec((10.0, 20.0, 40.0, 80.0), 2.0, (8.0, 4.0, 2.0, 1.0), (0.9, 0.8, 0.7, 0.6))
LAT = LatencyProfile({1: [1.0], 2: [1.5], 4: [2.0]})
NAMES = load_config("policies")["extended"]
ONE_HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def probs(k):
    p = [0.01] * 4
    p[k] = 0.97
    return tuple(p)


def job(i, tier=2, age=0.0, arrival=None, payload=100, p=None):
    h = Header(0.0, float(i) if arrival is None else arrival, payload, 0, 0, 0, p or probs(tier), 0.5, 0, 0)
    return QueuedJob(i, h, age)


def state(jobs, max_batch=1, lat=LAT):
    return State(100.0, tuple(jobs), 1, 1, lat, SPEC, max_batch)


def make(name, **kw):
    p = registry.build(name, **kw)
    p.reset(SPEC, np.random.default_rng(0))
    return p


def make_jobs(n=60, seed=0, rate=1.5, flip=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1.0 / rate, size=n))
    jobs = []
    for i, a in enumerate(t):
        tier = int(rng.choice(4, p=[0.1, 0.2, 0.3, 0.4]))
        h = Header(float(a), float(a), 500, 0, 1, 2, ONE_HOT[tier], 0.5, int(rng.integers(0, 2**62)),
                   int(rng.integers(0, 2**62)), (0.1,) * 13)
        jobs.append(Job(i, h, Truth((tier + flip) % 4, int(i % 2 == flip), None, "e", "s")))
    return jobs


def run_env(name, flip=0, max_batch=4):
    env = EdgeEnv(make_jobs(flip=flip), LAT, SPEC, 1, max_batch, Receiver(), seed=3)
    return env.run(registry.build(name))


def test_registered_and_grouped():
    assert len(NAMES) == 13 and set(NAMES) <= set(registry.all_policies())
    groups = load_config("policies")
    fam = [n for k, v in groups.items() if k.startswith("ext_") for n in v]
    assert sorted(fam) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_episode_valid_and_truth_blind(name):
    a, b = run_env(name), run_env(name, flip=1)       # same headers, different truth tier / fidelity
    assert any(o["status"] == "decoded" for o in a.outcomes)
    assert registry.all_policies()[name].uses_truth is False
    key = lambda r: [(o["id"], o["status"], o["batch_size"], o["finish_time"]) for o in r.outcomes]
    assert key(a) == key(b)


@pytest.mark.parametrize("name", NAMES)
def test_empty_queue_is_idle(name):
    d = make(name).select(state([]))
    assert d.batch == () and d.purge == ()


def test_d_over_abandons_low_value_urgent_job_and_purges_hopeless():
    dec = make("d_over").select(state([job(0, 3, age=159.5), job(1, 3, age=158.5), job(2, 0)]))
    assert dec.batch == (2,) and set(dec.purge) == {0, 1}
    assert make("d_over").select(state([job(1, 3, age=158.5)])).batch == (1,)   # free slot: admitted


def test_hvdf_prefers_higher_value_density():
    assert make("hvdf").select(state([job(0, 2, payload=100), job(1, 0, payload=100)])).batch == (1,)
    assert make("hvdf").select(state([job(0, 2, payload=100), job(1, 2, payload=1000)])).batch == (0,)


def test_edf_triage_drops_past_hard_deadline_and_runs_edf():
    dec = make("edf_triage").select(state([job(0, 3, age=159.5), job(1, 2, age=10), job(2, 0, age=1)]))
    assert dec.purge == (0,) and dec.batch == (2,)


def test_lcfs_priority_newest_within_top_tier():
    dec = make("lcfs_priority").select(state([job(0, 1, arrival=1), job(1, 1, arrival=5), job(2, 3, arrival=9)]))
    assert dec.batch == (1,)


def test_drr_respects_weights_over_long_run():
    p, counts, nxt = make("drr_tiers"), [0] * 4, 0
    for _ in range(300):
        jobs = [job(nxt + 10 * k + i, k) for k in range(4) for i in range(10)]   # always backlogged
        nxt += 100
        for i in p.select(state(jobs, max_batch=1)).batch:
            counts[(i % 100) // 10] += 1
    assert counts[0] / counts[3] == pytest.approx(8.0, rel=0.15)
    assert counts[1] / counts[3] == pytest.approx(4.0, rel=0.15)


def test_max_weight_serves_longest_weighted_queue():
    low = [job(i, 3) for i in range(9)]
    assert make("max_weight").select(state(low + [job(10, 0)])).batch == (0,)             # 9 x 1 > 1 x 8
    assert make("max_weight").select(state(low[:7] + [job(10, 0)])).batch == (10,)        # 7 < 8


def test_max_age_first():
    assert make("max_age").select(state([job(0, 0, age=1), job(1, 3, age=50)])).batch == (1,)


def test_whittle_aoi_trades_age_and_tier():
    assert make("whittle_aoi").select(state([job(0, 3, age=5), job(1, 0, age=5)])).batch == (1,)
    assert make("whittle_aoi").select(state([job(0, 2, age=1), job(1, 2, age=30)])).batch == (1,)


def test_cvar_spi_is_risk_averse():
    uncertain = job(0, p=(0.5, 0.0, 0.0, 0.5))     # mean weight 4.5, CVaR_0.5 = 1
    sure = job(1, 1)                                # weight ~4
    assert make("cvar_spi").select(state([uncertain, sure])).batch == (1,)
    assert make("cvar_spi", alpha=1.0).select(state([uncertain, sure])).batch == (0,)


def test_clipper_aimd_grows_then_halves():
    p = make("clipper_aimd")
    assert isinstance(p, ClipperAimd)
    jobs = [job(i, 2) for i in range(4)]
    done = lambda t0, t1, ids: dict(ids=ids, tier_probs=[probs(2)] * len(ids), delays=[0.0] * len(ids),
                                    time=t1, batch_size=len(ids))
    assert len(p.select(state(jobs, max_batch=4)).batch) == 1
    for r in range(3):                                         # fast batches (service 1.0 <= 10)
        b = p.select(state(jobs, max_batch=4)).batch
        p.observe("complete", done(100.0, 101.0, list(b)))
    assert p.cap == 4.0 and len(p.select(state(jobs, max_batch=4)).batch) == 4
    p.observe("complete", done(100.0, 150.0, [0, 1, 2, 3]))    # slow batch -> multiplicative decrease
    assert p.cap == 2.0


def test_clockwork_shrinks_batch_for_tight_deadline_and_rejects_hopeless():
    loose = [job(i, 2) for i in range(4)]
    assert len(make("clockwork").select(state(loose, max_batch=4)).batch) == 4
    tight = make("clockwork").select(state([job(9, 0, age=18.8)] + loose[:3], max_batch=4))
    assert tight.batch == (9,)
    dec = make("clockwork").select(state([job(0, 3, age=159.5), job(1, 2)], max_batch=4))
    assert dec.purge == (0,) and dec.batch == (1,)


def test_shepherd_packs_only_jobs_tolerating_batch_latency():
    loose = [job(i, 2) for i in range(4)]
    assert len(make("shepherd_nexus").select(state(loose, max_batch=4)).batch) == 4
    # job 9 has slack 1.2: tolerates b=1 (1.0) but not b=2 (1.5) / b=4 (2.0)
    mix = [job(9, 0, age=8.8)] + [job(i, 2) for i in range(3)]
    dec = make("shepherd_nexus").select(state(mix, max_batch=4))
    assert dec.batch == (0, 1)
    assert make("shepherd_nexus").select(state([job(0, 0, age=50)], max_batch=4)).batch == (0,)


def test_linucb_learns_to_prefer_valuable_header_and_is_seeded():
    p = make("linucb", alpha=0.0)
    hi, lo = job(0, 0), job(1, 3)
    for _ in range(30):                       # complete high-tier jobs fast, low-tier jobs likewise
        for jb in (hi, lo):
            p.pending[jb.id] = p.features(jb, state([jb]))
            p.observe("complete", dict(ids=[jb.id], tier_probs=[jb.header.tier_probs], delays=[1.0],
                                       time=1.0, batch_size=1))
    assert p.select(state([lo, hi])).batch == (0,)
    t1, t2 = make("linucb", thompson=True), make("linucb", thompson=True)
    js = [job(i, i % 4) for i in range(6)]
    assert t1.select(state(js, max_batch=2)).batch == t2.select(state(js, max_batch=2)).batch
