"""CPU tests for Evaluation.edge.decision_trace on a tiny synthetic stream."""
import numpy as np

from Evaluation.edge import decision_trace as dt
from Evaluation.edge.contracts import Header, Job, LatencyProfile, Truth, ValueSpec, Policy, Decision
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry

SPEC = ValueSpec((10.0, 20.0, 40.0, 80.0), 2.0, (8.0, 4.0, 2.0, 1.0), (0.9, 0.8, 0.5, 0.3))
LAT = LatencyProfile({1: [1.0, 1.0]})


def _job(i, arrival, tier):
    p = [0.05] * 4
    p[tier] = 0.85
    h = Header(arrival, arrival, 100, 0, 0, 0, tuple(p), 0.5, i * 7919, i * 104729)
    return Job(i, h, Truth(tier, 1, None))


# four Low jobs arrive first, a Critical arrives while the first is decoding
JOBS = [_job(0, 0.0, 3), _job(1, 0.1, 3), _job(2, 0.2, 3), _job(3, 0.3, 0)]


def _run(policy):
    tp = dt.TracingPolicy(policy, {j.id: j.truth for j in JOBS}, {j.id: "img%d.jpg" % j.id for j in JOBS})
    EdgeEnv(JOBS, LAT, SPEC, 1, 1, Receiver(), 0).run(tp)
    return tp


class _Fifo(Policy):
    name = "fifo_t"

    def select(self, state):
        return Decision((min(state.queue, key=lambda j: j.header.arrival_time).id,))


class _Genie(Policy):
    name = "genie_t"

    def __init__(self, truth):
        self.truth = truth

    def select(self, state):
        return Decision((min(state.queue, key=lambda j: (self.truth[j.id].tier, j.id)).id,))


class _Spy(Policy):
    name = "spy"
    seen = None

    def select(self, state):
        self.seen = state
        return Decision((state.queue[0].id,))


def test_records_every_decision_and_no_truth_leak():
    spy = _Spy()
    tp = _run(spy)
    decs = [r for r in tp.records if r["kind"] == "decision"]
    assert len(decs) == 4 and all(len(r["queue"]) >= 1 for r in decs)
    assert {r["true_tier"] for r in decs[0]["queue"]} <= {0, 1, 2, 3}
    assert tp.uses_truth is False
    assert not any(hasattr(j, "truth") for j in spy.seen.queue)   # QueuedJob has header only
    assert sum(r["kind"] == "complete" for r in tp.records) == 4


def test_semantic_precision_genie_vs_fifo():
    g = dt.semantic_metrics(_run(_Genie({j.id: j.truth for j in JOBS})).records)
    f = dt.semantic_metrics(_run(_Fifo()).records)
    assert g["semantic_precision"] == 1.0 and g["inversions_per_100"] == 0.0
    assert f["semantic_precision"] < 1.0 and f["inversions_per_100"] > 0.0
    assert g["wait_critical"] < f["wait_critical"]


def test_inversion_and_kendall_by_hand():
    q = [dict(id=0, true_tier=3, slack_hard=50.0, arrival=0.0), dict(id=1, true_tier=0, slack_hard=50.0, arrival=0.0)]
    rec = [dict(kind="decision", t=1.0, queue=q, batch=[0], min_service=1.0)]
    m = dt.semantic_metrics(rec)
    assert m["inversions_per_100"] == 100.0 and m["semantic_precision"] == 0.0 and m["kendall_tau"] == -1.0
    q[1]["slack_hard"] = 0.5   # Critical can no longer meet its deadline -> not an inversion
    assert dt.semantic_metrics(rec)["inversions_per_100"] == 0.0


def test_registry_policy_scores_and_html(tmp_path):
    sv = registry.build("sv_dpp")
    tp = _run(sv)
    assert any(r["has_scores"] for r in tp.records if r["kind"] == "decision")
    dt.write_outputs(tp.records, dt.semantic_metrics(tp.records), tmp_path, "sv_dpp")
    for ext in ("_trace.jsonl", "_metrics.json", "_replay.txt", "_replay.html"):
        assert (tmp_path / ("sv_dpp" + ext)).stat().st_size > 0
    assert "TRUE" in (tmp_path / "sv_dpp_replay.txt").read_text()
    assert "<script>" in (tmp_path / "sv_dpp_replay.html").read_text()
