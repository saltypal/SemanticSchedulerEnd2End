import numpy as np

from Evaluation.edge.contracts import Header, LatencyProfile, QueuedJob, State, ValueSpec
from Scheduler.edge import edgeserving, theory

SPEC = ValueSpec((5, 15, 60, 300), 2.0, (8, 4, 2, 1), (0.95, 0.85, 0.6, 0.2))
LAT = LatencyProfile({1: [1.0] * 5, 2: [1.5] * 5, 4: [2.0] * 5})


def job(i, tier, age):
    p = [0.01] * 4
    p[tier] = 0.97
    h = Header(0, 0, 1, 0, 0, 0, tuple(p), 0.5, 0, 0)
    return QueuedJob(i, h, age)


def state(jobs, max_batch=4):
    return State(10.0, tuple(jobs), 1, 1, LAT, SPEC, max_batch)


def mixed():
    return [job(1, 3, 100), job(2, 0, 1), job(3, 1, 8), job(4, 2, 30), job(5, 0, 4)]


def make_all():
    out = []
    for mod in (theory, edgeserving):
        for factory in mod.POLICIES.values():
            p = factory()
            p.reset(SPEC, np.random.default_rng(0))
            out.append(p)
    return out


def test_all_policies_valid_decisions():
    st = state(mixed())
    ids = {j.id for j in st.queue}
    for p in make_all():
        d = p.select(st)
        assert set(d.batch) <= ids and set(d.purge) <= ids
        assert len(set(d.batch)) == len(d.batch) and not set(d.batch) & set(d.purge)
        assert len(d.batch) in LAT.batch_sizes and not p.uses_truth


def test_whittle_hand_checked():
    # deterministic 1-slot service, linear cost gamma(x)=x: W = gamma(x+2)/theta + d
    w = theory.whittle_index([1.0], 0.0, lambda x: x, 0.5, 1.0)
    assert abs(w - (2 / 0.5 + 1.0)) < 1e-12
    assert theory.service_pmf([0.4, 0.9, 1.0], 1.0).tolist() == [1.0]
    # pmf (0.5, 0.5), theta 0.5, gamma=1, d=0: D=1 -> 0.5*2/0.5... N=1, T=1 ; D=2 -> N=1.5, T=1.25 -> 1.2
    assert abs(theory.whittle_index([0.5, 0.5], 0, lambda x: 1.0, 0.5, 0.0) - 1.2) < 1e-12


def test_urgent_critical_ranked_before_low():
    st = state(mixed(), max_batch=1)
    assert theory.GenCMuH().select(st).batch[0] in (2, 5)


def test_edgeserving_prefers_lower_score():
    st = state([job(1, 0, 9.0), job(2, 0, 9.5), job(3, 0, 0.0), job(4, 0, 0.0)])
    assert len(edgeserving.EdgeServing().select(st).batch) == 4   # empties the queue
    st2 = state([job(1, 0, 9.0), job(2, 3, 0.0)])
    assert edgeserving.EdgeServing().select(st2).batch == (1, 2)


def test_debt_rises_and_falls():
    p = edgeserving.CompletionDPP()
    p.reset(SPEC, np.random.default_rng(0))
    probs = (0.97, 0.01, 0.01, 0.01)
    for i in range(3):
        p.observe("arrival", {"id": i, "tier_probs": probs})
    up = p.Z[0]
    assert up > 0
    p.observe("complete", {"ids": [0, 1], "tier_probs": [probs, probs], "delays": [3.0, 4.0]})
    assert p.Z[0] < up
    p.observe("complete", {"ids": [2], "tier_probs": [probs], "delays": [99.0]})
    assert p.Z[0] >= 0


def test_dpp_purges_hopeless_and_serves_critical():
    for cls in (edgeserving.CompletionDPP, edgeserving.TierDPPNoGain):
        p = cls()
        p.reset(SPEC, np.random.default_rng(0))
        p.observe("arrival", {"id": 9, "tier_probs": (0.97, 0.01, 0.01, 0.01)})
        d = p.select(state([job(1, 0, 50.0), job(2, 0, 2.0), job(3, 3, 1.0)], max_batch=1))
        assert 1 in d.purge and d.batch == (2,)


def test_dpp_batch_size_stays_profiled_after_purge():
    # 4 jobs queued, 1 purged -> 3 scored; batch must be a profiled size (2), not 3.
    p = edgeserving.CompletionDPP()
    p.reset(SPEC, np.random.default_rng(0))
    d = p.select(state([job(1, 0, 50.0), job(2, 0, 1.0), job(3, 1, 2.0), job(4, 2, 3.0)]))
    assert d.purge == (1,) and len(d.batch) in LAT.batch_sizes and len(d.batch) == 2
