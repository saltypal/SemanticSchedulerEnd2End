"""Run controlled development replays from a measured road-reader probe.

Exact repeats are injected by construction. No field arrival trace is inferred.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median

from Scheduler.trial_evidence_policy import TrialCandidate, select_trial_candidate


@dataclass(frozen=True)
class MeasuredJob:
    candidate: TrialCandidate
    fact: str | None
    positive_report: bool
    service_ms: float
    failed: bool = False


def replay(offers, horizon_ms, policy, max_wait_ms):
    incoming = iter(sorted(offers, key=lambda job: (job.candidate.arrival_ms, job.candidate.token)))
    next_job = next(incoming, None)
    ready = {}
    recognized = set()
    covered = set()
    wait = []
    failures = false_reports = late = unfinished = 0
    order = []
    now = 0.0
    while now < horizon_ms and (ready or next_job is not None):
        if not ready:
            now = max(now, next_job.candidate.arrival_ms)
        if now >= horizon_ms:
            break
        while next_job is not None and next_job.candidate.arrival_ms <= now:
            ready[next_job.candidate.token] = next_job
            next_job = next(incoming, None)
        if policy == "fifo":
            candidate = next(iter(ready.values())).candidate
        else:
            candidate = select_trial_candidate(
                [job.candidate for job in ready.values()], tuple(recognized), now,
                max_wait_ms=max_wait_ms, near_bits=4, receiver_conditioning=policy == "conditional",
            )
        job = ready.pop(candidate.token)
        wait.append(now - candidate.arrival_ms)
        order.append(candidate.token)
        now += job.service_ms
        if now > horizon_ms:
            unfinished += 1
        elif job.failed:
            failures += 1
        else:
            late += now > candidate.deadline_ms
            if job.positive_report:
                recognized.add(candidate.signature)
                false_reports += job.fact is None
                if job.fact is not None and now <= candidate.deadline_ms:
                    covered.add(job.fact)

    offered_facts = {job.fact for job in offers if job.fact is not None}
    return dict(timely_facts=len(covered), offered_facts=len(offered_facts),
                false_reports=false_reports, late=late, failed=failures, unfinished=unfinished,
                unstarted=len(offers) - len(order), maximum_started_wait_ms=max(wait, default=None),
                order=order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    rows = probe["records"]
    service = median(row["decode_ms"] + row["reader_ms"] for row in rows)
    results = []
    for load in (0.5, 0.9, 1.2):
        offers = []
        for index, row in enumerate(rows + rows):
            capture = index * service / load
            arrival = capture + row["sender_ms"] + row["encode_ms"]
            candidate = TrialCandidate(str(index), arrival, capture + 6 * service,
                                       row["header"], int(row["appearance_hash"], 16), service)
            offers.append(MeasuredJob(candidate, row["id"] if row["native_road_damage"] else None,
                                      row["decoded"] >= probe["threshold"],
                                      row["decode_ms"] + row["reader_ms"]))
        horizon = max(job.candidate.deadline_ms for job in offers)
        for policy in ("fifo", "static", "conditional"):
            results.append(dict(load=load, policy=policy, **replay(offers, horizon, policy, 5 * service)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"scope": "Development stress only. Injected exact repeats, synthetic arrivals, "
                             "measured JPEG and frozen-reader service. No independent event estimate.",
                   "service_estimate_ms": service, "results": results}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
