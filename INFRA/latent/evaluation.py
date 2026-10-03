"""Ranking and delivery metrics with separate observable and reference classes."""

from __future__ import annotations

import math
from statistics import mean


def evaluate_latent_schedule(result, *, reference_priority=None):
    references = {} if reference_priority is None else dict(reference_priority)
    facts = {job["job_id"]: job for job in result["jobs"]}
    if references and set(references) != set(facts):
        raise ValueError("Reference priorities must cover the offered jobs exactly")
    if any(type(value) is not int or value not in range(4) for value in references.values()):
        raise ValueError("Reference priority must be 0..3")
    trace = result["decision_trace"]
    outcomes = result["outcomes"]
    served = [row for row in outcomes if "started_ms" in row]
    failed = [row for row in outcomes if row["status"] == "failed_transport"]
    deadlines = [job for job in facts.values() if job["deadline_ms"] is not None]
    on_time = [row for row in served if row["status"] == "completed" and row["deadline_met"] is True]
    matches = mismatches = gate_violations = 0
    for decision in trace:
        candidates = [row for row in decision["candidates"] if row["eligible"]]
        chosen = next(row for row in candidates if row["job_id"] == decision["job_id"])
        predicted = [row["priority_class"] for row in candidates if row["priority_class"] is not None]
        if predicted and chosen["priority_class"] is not None:
            gate_violations += int(chosen["priority_class"] > min(predicted))
        if references:
            for candidate in candidates:
                left = references[chosen["job_id"]]
                right = references[candidate["job_id"]]
                matches += int(left < right)
                mismatches += int(left > right)
    full_static = (result["config"]["mode"] == "static" and len(trace) == len(facts))
    ndcg = ndcg5 = None
    if references and full_static:
        gains = [4 - references[row["job_id"]] for row in trace]
        ideal = sorted(gains, reverse=True)
        def dcg(values):
            return sum(value / math.log2(index + 2) for index, value in enumerate(values))
        ndcg = dcg(gains) / dcg(ideal)
        ndcg5 = dcg(gains[:5]) / dcg(ideal[:5])
    quality_complete = (result["delivery_mode"] in {"cached_evidence", "live_evidence"} and
                        all(row.get("task_quality") is not None
                            for row in served if row["status"] == "completed"))
    utility_auc = task_gain = None
    if quality_complete:
        horizon = result["config"]["horizon_ms"]
        best_by_group = {}
        task_gain = area = 0.0
        for row in sorted(served, key=lambda item: item["completed_ms"]):
            if row["status"] != "completed":
                continue
            group = row.get("evidence_group_id") or row["job_id"]
            quality = row["task_quality"]
            gain = max(0.0, quality - best_by_group.get(group, 0.0))
            best_by_group[group] = max(quality, best_by_group.get(group, 0.0))
            task_gain += gain
            area += gain * max(0.0, horizon - row["completed_ms"])
        # Offered image count is policy independent. This is an explicit
        # evidence-quality coverage proxy, not universal task loss reduction.
        utility_auc = area / (horizon * len(facts))
    per_class = {}
    for label in range(4):
        offered = [job for job in facts.values() if (
            references.get(job["job_id"], job["priority_class"]
                           if job["priority_probabilities"] is None else
                           _median_class(job["priority_probabilities"])) == label)]
        ids = {job["job_id"] for job in offered}
        selected = [row for row in served if row["job_id"] in ids]
        waits = [row["wait_ms"] for row in selected]
        per_class[str(label)] = {"offered": len(offered), "served": len(selected),
                                 "mean_wait_ms": mean(waits) if waits else None,
                                 "max_wait_ms": max(waits) if waits else None}
    known_transport = result["delivery_mode"] in {"cached_evidence", "live_evidence"}
    return {
        "policy": result["policy"], "jobs": len(facts), "served": len(served),
        "confirmed_deliveries": sum(row["status"] == "completed" for row in served) if known_transport else None,
        "failed_transport": len(failed) if known_transport else None,
        "deadline_infeasible": sum(row["status"] == "deadline_infeasible" for row in outcomes),
        "horizon_censored": sum(row["status"] == "horizon_censored" for row in outcomes),
        "confirmed_on_time_fraction": len(on_time) / len(deadlines) if deadlines and known_transport else None,
        "reference_pair_agreement": matches / (matches + mismatches) if matches + mismatches else None,
        "reference_comparable_pairs": matches + mismatches,
        "reference_ndcg": ndcg, "reference_ndcg_at_5": ndcg5,
        "predicted_priority_gate_violations": gate_violations,
        "evidence_quality_gain": task_gain, "evidence_quality_auc": utility_auc,
        "resource_units_served": sum(row["resource_units"] for row in served),
        "resource_unit": result["config"]["resource_unit"],
        "mean_wait_ms": mean(row["wait_ms"] for row in served) if served else None,
        "mean_policy_inference_ms": mean(row["policy_inference_ms"] for row in trace) if trace else None,
        "per_class": per_class,
        "ranking_note": "nDCG requires a complete static schedule; dynamic pairs compare eligible jobs only",
        "utility_note": "Evidence quality coverage proxy; unavailable without confirmed task-quality evidence",
    }


def _median_class(probabilities):
    cumulative = 0.0
    for label, value in enumerate(probabilities):
        cumulative += value
        if cumulative >= 0.5:
            return label
    return 3
