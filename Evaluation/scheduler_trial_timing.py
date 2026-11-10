"""Measure scheduler selection overhead apart from the image-processing stages."""

import argparse
import json
import platform
from pathlib import Path
import random
from statistics import median, quantiles
from time import perf_counter_ns

from Scheduler.trial_evidence_policy import TrialCandidate, select_trial_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = random.Random(1729)
    history = tuple(rng.getrandbits(64) for _ in range(32))
    trials = []
    for size in (8, 32, 128, 512):
        queue = tuple(TrialCandidate(str(i), 0.0, 200.0, rng.random(), rng.getrandbits(64), 10.0)
                      for i in range(size))
        samples = {"static": [], "conditional": [], "hard_near_gate": []}
        for i in range(110):
            for key in samples:
                start = perf_counter_ns()
                select_trial_candidate(queue, history, 1.0, max_wait_ms=100.0, near_bits=4,
                                       receiver_conditioning=key == "conditional",
                                       hard_near_gate=key == "hard_near_gate")
                if i >= 10:
                    samples[key].append((perf_counter_ns() - start) / 1_000_000)
        for key, times in samples.items():
            trials.append(dict(ready=size, history=len(history), policy=key, calls=len(times),
                               p50_ms=median(times), p95_ms=quantiles(times, n=20, method="inclusive")[-1]))
    result = dict(scope="Fixed synthetic ready sets. Local Python wall-clock selector only.",
                  runtime=platform.platform(), python=platform.python_version(), trials=trials)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
