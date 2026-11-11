"""live_check logic with a fake decoder (no GPU, no model)."""
import threading
import time

import numpy as np
import pytest

from Evaluation.edge import arrivals, live_check
from Evaluation.edge.contracts import Header, LatencyProfile, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Scheduler.edge import registry

LAT = LatencyProfile({1: [0.004, 0.005, 0.006], 2: [0.007, 0.008, 0.009]})
SPEC = ValueSpec.from_config(load_config("sim"))
HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def _header(row, arrival, capture):
    return Header(capture, arrival, 500, row["mission"], 1, 0, HOT[row["tier"]], 0.5, row["image_id"],
                  row["image_id"] * 7, (0.0,) * 13)


def _truth(row):
    return Truth(row["tier"], 1, None, "e", "s")


def make_jobs(n=24, gap=0.006):
    rows = [dict(image_id=i, tier=i % 4, mission=i % 3) for i in range(n)]
    times = np.arange(1, n + 1) * gap
    return arrivals.build_jobs(rows, times, _header, _truth, np.random.default_rng(0))


class FakeDecoder:
    def __init__(self, delay=0.005):
        self.delay, self.calls, self.lock = delay, [], threading.Lock()

    def __call__(self, worker, batch):
        with self.lock:
            self.calls.append((worker, len(batch)))
        time.sleep(self.delay * len(batch))


def test_live_run_decodes_every_job_once_on_two_workers():
    jobs, dec = make_jobs(), FakeDecoder()
    rows, log = live_check.live_run(jobs, registry.build("fifo"), dec, LAT, SPEC, workers=2, max_batch=2)
    assert len(rows) == len(jobs) and all(r["status"] == "decoded" for r in rows)
    assert sorted(i for e in log for i in e["ids"]) == sorted(j.id for j in jobs)
    assert {w for w, _ in dec.calls} <= {0, 1}
    assert all(e["latency_s"] >= 0.004 * e["batch_size"] * 0.9 for e in log)
    assert all(r["delay"] is not None and r["delay"] > 0 for r in rows)


def test_decoder_error_surfaces():
    def boom(worker, batch):
        raise RuntimeError("gpu on fire")
    with pytest.raises(RuntimeError, match="gpu on fire"):
        live_check.live_run(make_jobs(6), registry.build("fifo"), boom, LAT, SPEC, workers=2, max_batch=2)


def test_ctr_delta_known_values():
    live = [dict(id=i, tier=0, status="decoded", delay=1.0 if i < 8 else 99.0) for i in range(10)]
    sim = [[dict(id=i, tier=0, status="decoded", delay=1.0) for i in range(10)]]
    out = live_check.ctr_delta(live, sim, 5.0, 200, np.random.default_rng(0))
    assert out["ctr_live"] == pytest.approx(0.8) and out["ctr_sim"] == pytest.approx(1.0)
    assert out["delta"] == pytest.approx(-0.2) and out["delta_ci95"][0] <= -0.2 <= out["delta_ci95"][1]
    none = live_check.ctr_delta([dict(id=1, tier=2, status="decoded", delay=1.0)], sim, 5.0, 10,
                                np.random.default_rng(0))
    assert none["delta"] is None


def test_ks_per_batch_matching_vs_shifted():
    rng = np.random.default_rng(0)
    prof = LatencyProfile({1: rng.normal(1.0, 0.05, 300).clip(0.5), 2: rng.normal(2.0, 0.05, 300).clip(0.5)})
    same = [dict(batch_size=1, latency_s=float(x)) for x in rng.normal(1.0, 0.05, 200)]
    slow = [dict(batch_size=2, latency_s=float(x)) for x in rng.normal(2.5, 0.05, 200)]
    ks = live_check.ks_per_batch(same + slow, prof)
    assert ks["1"]["p_value"] > 0.01 and ks["2"]["p_value"] < 1e-6 and ks["2"]["statistic"] > 0.9
    assert live_check.ks_per_batch(same[:2], prof)["1"]["p_value"] is None


def test_run_live_check_report_shape():
    jobs = make_jobs(20)
    facs = {p: (lambda p=p: registry.build(p)) for p in ("fifo", "edf", "sv_dpp")}
    rep = live_check.run_live_check(jobs, LAT, SPEC, FakeDecoder(), facs, workers=2, max_batch=2,
                                    sim_seeds=range(2), n_boot=50)
    assert set(rep["policies"]) == {"fifo", "edf", "sv_dpp"} and rep["claim"] == "valid for c = 2 only"
    assert set(rep["ks_by_batch"]) == {"1", "2"} and rep["n_jobs"] == 20
    for p in rep["policies"].values():
        assert p["n_jobs"] == 20 and p["ctr_at_d"]["n_critical"] == 5
        assert len(p["ctr_at_d"]["delta_ci95"]) == 2
