"""Trace arrivals (I8), regimes (I10) and sensitivity rows (I13) on synthetic simtable-format rows."""
import numpy as np
import pytest

from Evaluation.edge import arrivals, regimes, run_sweep, tables, trace
from Evaluation.edge.contracts import LatencyProfile, ValueSpec
from Evaluation.edge.edge_config import load_config

LAT = LatencyProfile({1: [0.9, 1.0, 1.1], 2: [1.5, 1.6, 1.7]})


def make_rows(n=80, n_crisis=50, split="val"):
    rows = []
    for i in range(n):
        crisis = i < n_crisis
        m = 1 + i % 3 if crisis else 0
        rbm = [0.5 + 0.01 * (i % 7) + 0.1 * k for k in range(4)]
        p = [0.1, 0.1, 0.1, 0.1]
        p[i % 4] = 0.7
        rows.append(dict(image_id="im%d" % i, event="ev%d" % (i % 3) if crisis else None,
                         source="crisismmd" if crisis else "coco_people", split=split,
                         tier=i % 4, tier_alt=(i + 1) % 4, tier_probs=p, mission=m,
                         relevance=rbm[m], relevance_by_mission=rbm, sem_key=i * 977 + 1,
                         perc_key=i * 131 + 2, sts=[], payload_bytes=464, codec="v1", fidelity=1,
                         dup_group=(i // 10) if i % 10 < 2 and crisis else None,
                         capture_time_s=1.5e9 + 37.0 * i ** 1.3 if crisis else None))
    return rows


def sweep(**kw):
    args = dict(rows=make_rows(), latency=LAT, split="val", policies=["fifo", "edf"], workers=[1],
                rhos=[1.0], realisations=2, base_seed=3, max_batch=2, n_boot=10, tables=tables)
    args.update(kw)
    return run_sweep.run_sweep(**args)


def test_trace_order_rho_and_mission_relevance():
    rows = make_rows()
    rate = arrivals.rate_for_rho(1.0, LAT, 2, 1)
    out, times, info = trace.trace_arrivals(rows, rate, np.random.default_rng(0))
    assert len(out) == len(rows) and len({r["image_id"] for r in out}) == len(rows)  # once each
    assert np.all(np.diff(times) >= 0)
    crisis = [r["capture_time_s"] for r in out if r["capture_time_s"] is not None]
    assert crisis == sorted(crisis)                                                  # tweet-time order
    nc = info["n_crisis"]
    idx = [i for i, r in enumerate(out) if r["capture_time_s"] is not None]
    span = times[idx]
    assert (nc - 1) / (span[-1] - span[0]) == pytest.approx(nc / len(rows) * rate, rel=0.05)  # rho met
    for r in out:
        assert r["relevance"] == r["relevance_by_mission"][r["mission"]]
    crisis_missions = {r["image_id"]: r["mission"] for r in rows if r["capture_time_s"] is not None}
    assert all(r["mission"] == crisis_missions[r["image_id"]] for r in out if r["image_id"] in crisis_missions)


def test_trace_realisations_differ_in_everyday_only():
    rows = make_rows()
    a, ta, _ = trace.trace_arrivals(rows, 1.0, np.random.default_rng(1))
    b, tb, _ = trace.trace_arrivals(rows, 1.0, np.random.default_rng(2))

    def ev(o):
        return [r["image_id"] for r in o if r["capture_time_s"] is None]

    def cr(o):
        return [r["image_id"] for r in o if r["capture_time_s"] is not None]
    assert cr(a) == cr(b) and ev(a) != ev(b)


def test_trace_sweep_records_kappa_and_default_unchanged():
    res = sweep(arrival="trace")
    assert all(c["kappa"] and c["kappa"] > 0 and len(c["kappas"]) == 2 for c in res["cells"])
    assert res["cells"][0]["kappa"] == res["cells"][1]["kappa"]            # paired across policies
    assert all(c["kappa"] is None for c in sweep()["cells"])               # poisson stays default


def test_named_scenario_selectable_and_unknown_rejected():
    res = sweep(scenario_names=["noisy_spi_10", "loss_5pct"])
    assert res["config"]["scenario"] == {"spi_noise": 0.10, "loss": 0.05}
    with pytest.raises(ValueError):
        regimes.resolve_scenarios(["nope"])
    names = load_config("sim")["scenarios"]
    for n in ("mmpp_bursts", "wrong_sigma_w_low", "noisy_spi_10", "tier_alt", "critical_share_x2",
              "weights_x2", "loss_5pct", "capture_jitter_30s"):
        assert n in names
        regimes.resolve_scenarios([n])


def test_noisy_spi_changes_header_not_truth():
    rows = make_rows()
    out = regimes.inject_spi_noise(rows, 0.3, np.random.default_rng(0))
    changed = [i for i, (a, b) in enumerate(zip(rows, out)) if a["tier_probs"] != b["tier_probs"]]
    assert 5 < len(changed) < 50
    assert all(a["tier"] == b["tier"] and a["tier_alt"] == b["tier_alt"] for a, b in zip(rows, out))
    assert all(abs(sum(b["tier_probs"]) - 1) < 1e-9 for b in out)
    assert out == regimes.inject_spi_noise(rows, 0.3, np.random.default_rng(0))   # seeded
    assert regimes.inject_spi_noise(rows, 0.0, np.random.default_rng(0)) == rows


def test_noisy_spi_in_sweep_jobs_keeps_truth():
    rows = make_rows()
    base, _ = run_sweep._make_jobs(rows, LAT, tables, 1, 1.0, 0, 3, 2, None, 0.0, {})
    noisy, _ = run_sweep._make_jobs(rows, LAT, tables, 1, 1.0, 0, 3, 2, None, 0.0, {"spi_noise": 0.5})
    assert [j.truth for j in base] == [j.truth for j in noisy]
    assert [j.header.tier_probs for j in base] != [j.header.tier_probs for j in noisy]


def test_wrong_sigma_w_changes_policy_view_only():
    lat = LatencyProfile({1: [0.5, 1.0, 1.5], 2: [1.0, 1.5, 2.0]})
    wide = regimes.scale_latency_variance(lat, 4.0)
    assert wide.mean(1) == pytest.approx(lat.mean(1))
    assert wide.samples[1].std() > 2 * lat.samples[1].std()
    assert regimes.scale_latency_variance(lat, 0.25).samples[1].std() < 0.5 * lat.samples[1].std()
    assert regimes.scale_latency_variance(lat, 1.0) is lat
    from Evaluation.edge.env import EdgeEnv
    from Evaluation.edge.receiver import Receiver
    from Scheduler.edge import registry
    jobs, _ = run_sweep._make_jobs(make_rows(20, 10), lat, tables, 1, 1.0, 0, 0, 2, None, 0.0, {})
    pol = registry.build("fifo")
    seen, orig = [], pol.select
    pol.select = lambda st: (seen.append(st.latency), orig(st))[1]
    spec = ValueSpec.from_config(load_config("sim"))
    EdgeEnv(jobs, lat, spec, 1, 2, Receiver(), 0, policy_latency=wide).run(pol)
    assert seen and all(s is wide for s in seen)


def test_sensitivity_transforms():
    rows = make_rows(200, 150)
    assert [r["tier"] for r in regimes.use_tier_key(rows, "tier_alt")] == [r["tier_alt"] for r in rows]
    assert [r["tier"] for r in rows] == [i % 4 for i in range(200)]       # original rows untouched

    def share(rs):
        return np.mean([r["tier"] == 0 for r in rs if regimes.is_crisis(r)])
    up = regimes.resample_critical(rows, 2.0, np.random.default_rng(0))
    dn = regimes.resample_critical(rows, 0.5, np.random.default_rng(0))
    assert share(up) == pytest.approx(2 * share(rows), rel=0.1) and share(dn) < 0.7 * share(rows)
    assert len({r["image_id"] for r in up}) == len(up)                    # still without replacement
    assert sum(not regimes.is_crisis(r) for r in up) == 50
    spec = ValueSpec.from_config(load_config("sim"))
    assert regimes.scale_weights(spec, 2.0).weights == tuple(2 * w for w in spec.weights)
    jobs, _ = run_sweep._make_jobs(rows, LAT, tables, 1, 1.0, 0, 0, 2, None, 0.0, {})
    lost = regimes.drop_arrivals(jobs, 0.05, np.random.default_rng(0))
    assert 0.8 * len(jobs) < len(lost) < len(jobs)
    jit = regimes.jitter_capture(jobs, 30.0, np.random.default_rng(0))
    assert all(a.header.capture_time >= b.header.capture_time for a, b in zip(jobs, jit))
    assert any(a.header.capture_time > b.header.capture_time for a, b in zip(jobs, jit))
    assert [j.truth for j in jobs] == [j.truth for j in jit]


def test_loss_and_weights_scenarios_run():
    full = sweep(policies=["fifo"])["cells"][0]
    lost = sweep(policies=["fifo"], scenario_names=["loss_5pct"])["cells"][0]
    heavy = sweep(policies=["fifo"], scenario_names=["weights_x2"])["cells"][0]
    assert heavy["value"] == pytest.approx(2 * full["value"], rel=1e-6)
    assert lost["n_measured"] <= full["n_measured"]
