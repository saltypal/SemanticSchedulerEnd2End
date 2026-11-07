"""Bounded numerical checks of proposed objectives, independent of production.

These tiny synthetic queues test mathematical scope, not semantic model quality.
All jobs are complete on arrival, nonpreemptive, and attempted exactly once.
No images, teacher, classifier, GPU, or codec are used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import sys
import time


@dataclass(frozen=True)
class Job:
    identity: str
    arrival: float
    weight: float


def greedy(jobs: tuple[Job, ...]) -> tuple[tuple[Job, float], ...]:
    remaining = list(jobs)
    now = 0.0
    selected = []
    while remaining:
        ready = [j for j in remaining if j.arrival <= now]
        if not ready:
            now = min(j.arrival for j in remaining)
            ready = [j for j in remaining if j.arrival <= now]
        job = min(ready, key=lambda j: (-j.weight, j.arrival, j.identity))
        selected.append((job, now))
        remaining.remove(job)
        now += 1.0
    return tuple(selected)


def exact_cost(jobs: tuple[Job, ...]) -> float:
    """Enumerate legal ready actions; no intentional idling is allowed."""
    @lru_cache(None)
    def recurse(remaining: tuple[int, ...], now: float) -> float:
        if not remaining:
            return 0.0
        ready = [i for i in remaining if jobs[i].arrival <= now]
        if not ready:
            now = min(jobs[i].arrival for i in remaining)
            ready = [i for i in remaining if jobs[i].arrival <= now]
        return min(
            jobs[i].weight * (now - jobs[i].arrival)
            + recurse(tuple(j for j in remaining if j != i), now + 1.0)
            for i in ready
        )

    return recurse(tuple(range(len(jobs))), 0.0)


def cost(trace: tuple[tuple[Job, float], ...]) -> float:
    return sum(j.weight * (start - j.arrival) for j, start in trace)


def integrated_cost(jobs: tuple[Job, ...], starts: dict[str, float], horizon: float) -> float:
    """Compute cost by independent interval integration, including unselected jobs."""
    times = {0.0, horizon}
    times.update(j.arrival for j in jobs if 0 <= j.arrival <= horizon)
    times.update(s for s in starts.values() if 0 <= s <= horizon)
    area = 0.0
    boundaries = sorted(times)
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left + right) / 2
        waiting_weight = sum(
            j.weight for j in jobs
            if j.arrival <= midpoint < starts.get(j.identity, float("inf"))
        )
        area += waiting_weight * (right - left)
    return area


def run() -> dict:
    checked = 0
    max_gap = 0.0
    # Exhaustive four-job integer arrivals and positive static weights.
    for arrivals in itertools.product((0.0, 1.0, 2.0), repeat=4):
        for weights in itertools.product((1.0, 2.0, 4.0), repeat=4):
            jobs = tuple(Job(str(i), a, w) for i, (a, w) in enumerate(zip(arrivals, weights)))
            gap = cost(greedy(jobs)) - exact_cost(jobs)
            max_gap = max(max_gap, abs(gap))
            assert abs(gap) < 1e-10, (jobs, gap)
            checked += 1
        if checked % 2187 == 0:
            print(f"[exhaustive] {checked}/6561 arrival/weight cases", flush=True)

    # Fractional arrivals prevent reliance on integer-slot admission semantics.
    rng = random.Random(20261030)
    for case in range(100):
        jobs = tuple(Job(str(i), rng.randrange(17) / 4, float(rng.randrange(1, 10))) for i in range(5))
        gap = cost(greedy(jobs)) - exact_cost(jobs)
        max_gap = max(max_gap, abs(gap))
        assert abs(gap) < 1e-10, (case, jobs, gap)

    all_ready = tuple(Job(str(i), 0.0, w) for i, w in enumerate((8.0, 4.0, 2.0, 1.0)))
    all_orders = list(itertools.permutations(all_ready))
    weighted = [sum(j.weight * position for position, j in enumerate(order)) for order in all_orders]
    unweighted = [sum(float(position) for position, _ in enumerate(order)) for order in all_orders]
    assert len(set(unweighted)) == 1 and unweighted[0] == 6.0
    assert cost(greedy(all_ready)) == min(weighted)

    # Same-class swapping is invisible to class-only waiting and class-only nDCG.
    a, b = Job("repeat", 0.0, 8.0), Job("new", 0.0, 8.0)
    repeated_first, new_first = ((a, 0.0), (b, 1.0)), ((b, 0.0), (a, 1.0))
    assert cost(repeated_first) == cost(new_first) == 8.0
    # Declared controlled preference, not inferred natural emergency correctness.
    useful_new_waits = {"repeat_first": 1.0, "new_first": 0.0}

    # A fractional arrival inside service and an unselected job at the endpoint.
    jobs = (Job("selected", 0.0, 8.0), Job("during", 0.25, 4.0), Job("never", 0.5, 2.0))
    starts, horizon = {"selected": 0.0, "during": 1.0}, 1.5
    individual = sum(j.weight * max(0.0, min(starts.get(j.identity, horizon), horizon) - j.arrival) for j in jobs)
    area = integrated_cost(jobs, starts, horizon)
    assert area == individual == 5.0

    # Correct available-priority scheduling need not produce monotone class means.
    # High work starts while the Critical job has not arrived.
    jobs = (Job("high", 0.0, 4.0), Job("critical", 0.25, 8.0))
    trace = greedy(jobs)
    waits = {j.identity: start - j.arrival for j, start in trace}
    assert waits == {"high": 0.0, "critical": 0.75}

    # Independent reference weight can differ from the transmitted predicted SPI.
    jobs = (Job("A", 0.0, 8.0), Job("B", 0.0, 1.0))
    erroneous_spi_order = ((jobs[1], 0.0), (jobs[0], 1.0))
    assert cost(erroneous_spi_order) == 8.0 and cost(greedy(jobs)) == 1.0

    return {
        "status": "passed",
        "scope": "Tiny synthetic mathematical reference checks; not learner or semantic validation",
        "assumptions": ["one nonpreemptive server", "unit service", "no intentional wait", "fixed known positive weights", "one attempt per job", "full drain for greedy-optimum comparisons"],
        "exhaustive_integer_cases": checked,
        "fractional_arrival_cases": 100,
        "maximum_greedy_minus_exact_absolute_gap": max_gap,
        "four_ready_job_orders": len(all_orders),
        "unweighted_total_wait_all_orders": unweighted[0],
        "four_job_weighted_cost_range": [min(weighted), max(weighted)],
        "same_class_cost_both_orders": cost(repeated_first),
        "same_class_mean_wait_both_orders": 0.5,
        "controlled_new_information_wait": useful_new_waits,
        "finite_horizon_wait_area": area,
        "correct_policy_nonmonotone_class_wait_example": waits,
        "erroneous_spi_cost": cost(erroneous_spi_order),
        "reference_weight_order_cost": cost(greedy(jobs)),
        "non_claims": ["No universal optimality for variable service, arbitrary idling, changing utilities, or partial information", "No trained model, VLM correctness, real imagery, or measured decoder performance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print("[start] Bounded scheduling-objective reference checks, CPU only", flush=True)
    source = Path(__file__)
    provenance = {
        "script": str(source.resolve()),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    try:
        result = run()
        result["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({"error": repr(error), "elapsed_seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
        raise
    print(f"[complete] {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
