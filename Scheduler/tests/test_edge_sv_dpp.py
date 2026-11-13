import numpy as np

from Evaluation.edge.contracts import Header, LatencyProfile, QueuedJob, State, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Scheduler.edge.sv_dpp import POLICIES, GenieSVDPP, SVDPP

SPEC = ValueSpec.from_config(load_config("sim"))
ONE = {k: tuple(1.0 if i == k else 0.0 for i in range(4)) for k in range(4)}


def hdr(p, arrival=0.0):
    return Header(0.0, arrival, 1000, 0, 0, 0, tuple(p), 0.5, 0, 0)


def qj(i, p, age=0.0, dup=0.0, u=1.0, arrival=None):
    return QueuedJob(i, hdr(p, i * 0.01 if arrival is None else arrival), age, dup, u)


def state(queue, lat=None, max_batch=1, workers=1, rate=0.0):
    lat = lat or LatencyProfile({1: [0.1, 0.1], 2: [0.15, 0.15]})
    return State(0.0, tuple(queue), 1, workers, lat, SPEC, max_batch, arrival_rate=rate)


def make(cls=SVDPP, **kw):
    pol = cls(**kw)
    pol.reset(SPEC, np.random.default_rng(0))
    return pol


def test_degenerate_orders_by_expected_weight():
    q = [qj(0, ONE[3]), qj(1, ONE[0]), qj(2, ONE[2]), qj(3, ONE[1])]
    pol, order, st = make(), [], state(q)
    for _ in range(4):
        d = pol.select(st)
        order.append(d.batch[0])
        st = state([x for x in st.queue if x.id != d.batch[0]])
    assert order == [1, 3, 2, 0]


def test_debt_grows_then_falls():
    pol = make()
    for i in range(3):
        pol.observe("arrival", {"id": i, "tier_probs": ONE[0]})
    pol.observe("complete", {"ids": [], "tier_probs": [], "delays": []})
    assert pol.debts[0] == 3 * 0.95
    pol.observe("complete", {"ids": [0, 1, 2], "tier_probs": [ONE[0]] * 3, "delays": [1.0, 2.0, 3.0]})
    assert pol.debts[0] == 3 * 0.95 - 3 or pol.debts[0] == 0.0
    # late completions (delay > H_0 = 10) do not pay the debt
    pol.observe("arrival", {"id": 9, "tier_probs": ONE[0]})
    pol.observe("complete", {"ids": [9], "tier_probs": [ONE[0]], "delays": [50.0]})
    assert pol.debts[0] == 0.95
    pol.observe("purge", {"ids": [1]})
    pol.observe("expire", {"ids": [1]})


def test_past_hard_deadline_is_purged():
    d = make().select(state([qj(0, ONE[0], age=100.0), qj(1, ONE[1])]))
    assert d.purge == (0,) and d.batch == (1,)


def test_likely_duplicate_is_purged_but_soft_kept_below_threshold():
    d = make().select(state([qj(0, ONE[1], dup=0.97), qj(1, ONE[1], dup=0.5)]))
    assert d.purge == (0,) and d.batch == (1,)


def test_batch_ratio_prefers_larger_batch_when_scores_high_and_sublinear():
    lat = LatencyProfile({1: [1.0], 2: [1.1], 4: [1.3]})
    q = [qj(i, ONE[0], age=3.0 + 0.5 * i) for i in range(4)]
    pol = make(debt_init=(40.0, 0, 0, 0))
    d1 = pol.select(state(q, lat, max_batch=1))
    d4 = pol.select(state(q, lat, max_batch=4))
    assert len(d1.batch) == 1 and len(d4.batch) > 1


def test_genie_reads_truth_normal_does_not():
    q = [qj(0, ONE[3]), qj(1, ONE[3])]
    truth = {0: Truth(3, 1, None), 1: Truth(0, 1, None)}
    g = make(GenieSVDPP, truth=truth)
    assert g.uses_truth and g.name == "genie_sv_dpp"
    assert g.select(state(q)).batch == (1,)          # true tier of job 1 is Critical
    n = make()
    assert not n.uses_truth and not hasattr(n, "truth")
    assert n.select(state(q)).batch == (0,)         # tie -> earlier arrival
    assert set(POLICIES) == {"sv_dpp", "genie_sv_dpp"}


def test_genie_dup_only_after_delivery():
    truth = {0: Truth(1, 1, 7), 1: Truth(1, 1, 7), 2: Truth(1, 1, None)}
    g = make(GenieSVDPP, truth=truth)
    q = [qj(1, ONE[1]), qj(2, ONE[1])]
    assert g.select(state(q)).purge == ()
    g.observe("complete", {"ids": [0], "tier_probs": [ONE[1]], "delays": [1.0]})
    assert g.select(state(q)).purge == (1,)


def test_never_returns_ids_outside_queue():
    rng = np.random.default_rng(1)
    lat = LatencyProfile({1: [0.5, 0.7], 2: [0.8, 0.9], 4: [1.0, 1.2]})
    pol = make(debt_init=(5.0, 5.0, 5.0, 5.0))
    for trial in range(20):
        n = int(rng.integers(1, 9))
        q = [qj(100 + i, rng.dirichlet(np.ones(4)), age=float(rng.uniform(0, 80)),
                dup=float(rng.uniform(0, 1))) for i in range(n)]
        d = pol.select(state(q, lat, max_batch=4, workers=2, rate=0.5))
        ids = {x.id for x in q}
        assert set(d.batch) <= ids and set(d.purge) <= ids and not set(d.batch) & set(d.purge)
        assert len(d.batch) <= 4


def test_no_pressure_tie_still_prefers_throughput_batch():
    lat = LatencyProfile({1: [1.0, 1.0], 8: [3.0, 3.0]})
    q = [qj(i, ONE[2]) for i in range(8)]
    d = make().select(state(q, lat=lat, max_batch=8))
    assert len(d.batch) == 8


def test_at_risk_critical_beats_safe_critical_when_debt_high():
    lat = LatencyProfile({1: [1.0, 1.2]})
    pol = make()
    pol.Z = [5.0, 0.0, 0.0, 0.0]
    d = pol.select(state([qj(0, ONE[0], age=0.0), qj(1, ONE[0], age=8.5)], lat=lat, rate=0.2))
    assert d.batch == (1,)


def test_deferring_top_ranked_job_is_not_free():
    """Deferral costs at least one service round (sec. 6.2): the critical about to cross D_C goes first."""
    aged, fresh = qj(0, ONE[0], age=4.5), qj(1, ONE[0], age=0.0)
    d = make().select(state([fresh, aged], lat=LatencyProfile({1: [1.0, 1.0], 2: [1.5, 1.5]})))
    assert d.batch == (0,)


def test_negligible_debt_credit_does_not_outrank_confident_critical():
    """Sec. 6.3 ties are per job: tiny debt credit on a likely-Low job must not starve a confident Critical."""
    lat = LatencyProfile({1: [0.4, 0.4], 2: [0.55, 0.55], 4: [0.85, 0.85], 8: [1.4, 1.4]})
    low = [qj(i + 1, (0.02, 0.03, 0.05, 0.9), age=0.0) for i in range(20)]
    crit = qj(0, (0.98, 0.01, 0.01, 0.0), age=2.0)
    pol = make()
    pol.Z = [15.0, 2.4, 2.0, 0.0]
    d = pol.select(state(low + [crit], lat=lat, max_batch=8, workers=2, rate=1.8))
    assert 0 in d.batch


def test_tie_tol_frac_default_is_val_tuned():
    # val-tuned on true v1 latency (config/edge/sim.yaml comment); 0.1 trailed EDF at rho 1.2 / 2 workers
    pol = make()
    assert pol.tie_tol_frac == 0.3
    assert make(tie_tol_frac=0.1).tie_tol_frac == 0.1
