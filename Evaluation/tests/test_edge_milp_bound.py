import pytest

from Evaluation.edge.contracts import Header, Job, LatencyProfile, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.milp_bound import clairvoyant_bound, greedy_upper_bound

SPEC = ValueSpec.from_config(load_config("sim"))


def job(i, tier, arrival=0.0, u=1, dup=None):
    h = Header(arrival, arrival, 1000, 0, 0, 0, (0.25,) * 4, 0.5, 0, 0)
    return Job(i, h, Truth(tier, u, dup))


def test_tiny_four_job_known_optimum():
    # service 1 s (2 grid steps), 1 worker; Critical H=10 s. All arrive at 0.
    jobs = [job(0, 0), job(1, 1), job(2, 2), job(3, 3)]
    lat = LatencyProfile({1: [1.0]})
    val, sched = clairvoyant_bound(jobs, lat, SPEC, workers=1, dt=0.5)
    # all four fit before their soft deadlines of 5/15/60/300 only if each finishes <= D:
    # Critical must finish <= 5 s: order C,H,M,L finishes at 1,2,3,4 s -> all g=1 => 8+4+2+1.
    assert val == pytest.approx(15.0) and len(sched) == 4


def test_contention_drops_lowest_value_job():
    # three Critical + one Low, service 4 s, 1 worker: only two 4 s jobs fit before H=10 s.
    jobs = [job(0, 0), job(1, 0), job(2, 0), job(3, 3)]
    val, sched = clairvoyant_bound(jobs, LatencyProfile({1: [4.0]}), SPEC, 1, dt=0.5)
    # first Critical ends at 4 (g=1)=8, second at 8 (g=(10-8)/5=0.4)=3.2, Low then ends 12 g=1 => +1
    assert val == pytest.approx(8 + 3.2 + 1.0)


def test_greedy_bound_dominates_milp_and_handles_dups():
    jobs = [job(0, 0), job(1, 0, dup=3), job(2, 0, dup=3), job(3, 2, u=0)]
    lat = LatencyProfile({1: [3.0]})
    ub = greedy_upper_bound(jobs, SPEC)
    assert ub == pytest.approx(8 + 8 + 0.25 * 8 + 0)
    val, _ = clairvoyant_bound(jobs, lat, SPEC, 1)
    assert val <= ub + 1e-9


def test_rejects_large_batch_and_long_trace():
    lat = LatencyProfile({1: [1.0]})
    with pytest.raises(ValueError):
        clairvoyant_bound([job(0, 0)], lat, SPEC, 1, max_batch=2)
    with pytest.raises(ValueError):
        clairvoyant_bound([job(i, 3) for i in range(201)], lat, SPEC, 1)
