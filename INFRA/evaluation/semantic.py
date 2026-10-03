"""Semantic utility and aggregate metrics independent of policy code."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Iterable, Mapping, Sequence

from INFRA.scheduling.semantic_contracts import BASE_PRIORITY_WEIGHTS, EpisodeJobSpec, SemanticUtilityBreakdown


def normalized_priority_weight(priority: int) -> float:
    return BASE_PRIORITY_WEIGHTS[priority] / BASE_PRIORITY_WEIGHTS[0]


def _feature_value(job: EpisodeJobSpec, name: str, default: float) -> float:
    value = getattr(job.candidate.features, name)
    return float(value.value) if value.available and value.value is not None else default


def delivery_utility(
    job: EpisodeJobSpec,
    *,
    wait_ms: float,
    previous_group_sufficiency: float | None,
) -> SemanticUtilityBreakdown:
    """Compute semantic gain without mutating receiver knowledge state."""
    sufficiency = max(0.0, min(1.0, _feature_value(job, "evidence_sufficiency", 0.0)))
    urgency = max(0.0, min(3.0, _feature_value(job, "urgency", 0.0)))
    if previous_group_sufficiency is None:
        coverage = 1.0
    elif sufficiency > previous_group_sufficiency:
        coverage = 0.25
    else:
        coverage = 0.05
    sufficiency_factor = 1.0 if sufficiency >= 0.5 else 0.25
    delay_factor = math.exp(-(urgency / 3.0) * wait_ms / 200.0)
    utility = normalized_priority_weight(job.base_priority) * coverage * sufficiency_factor * delay_factor
    return SemanticUtilityBreakdown(coverage, sufficiency_factor, delay_factor, utility)


def expired_penalty(jobs: Iterable[EpisodeJobSpec]) -> float:
    return sum(normalized_priority_weight(item.base_priority) for item in jobs)


def transition_reward(
    breakdown: SemanticUtilityBreakdown | None,
    *,
    normalized_service_cost: float,
    newly_expired: Iterable[EpisodeJobSpec] = (),
    failed_job: EpisodeJobSpec | None = None,
    cost_coefficient: float = 0.1,
) -> float:
    if not 0.0 <= normalized_service_cost:
        raise ValueError("normalized_service_cost must be nonnegative")
    utility = breakdown.utility_delta if breakdown is not None else 0.0
    expiry = expired_penalty(newly_expired)
    failure = normalized_priority_weight(failed_job.base_priority) * 0.25 if failed_job is not None else 0.0
    return 4.0 * utility - cost_coefficient * normalized_service_cost - expiry - failure


def normalized_utility_auc(points: Sequence[tuple[float, float]], *, horizon_ms: float, maximum_utility: float) -> float | None:
    if horizon_ms <= 0 or maximum_utility <= 0 or not points:
        return None
    ordered = sorted(points)
    area = 0.0
    previous_time, previous_utility = ordered[0]
    for time, utility in ordered[1:]:
        bounded = min(horizon_ms, max(previous_time, time))
        area += max(0.0, bounded - previous_time) * previous_utility
        previous_time, previous_utility = bounded, utility
        if previous_time >= horizon_ms:
            break
    if previous_time < horizon_ms:
        area += (horizon_ms - previous_time) * previous_utility
    return area / (horizon_ms * maximum_utility)


def priority_evaluation_window_ms(result: Mapping[str, object]) -> float:
    """Fixed per-episode window from arrivals and offered service demand.

    The 2-second simulator horizon is only a safety limit. Using it for the
    primary AUC would make schedules finishing in a few hundred milliseconds
    all score close to one and hide meaningful ordering differences.
    """
    facts = tuple(result.get("job_facts", ()))
    horizon = float(result.get("horizon_ms", 0.0))
    rate = float(result.get("service_rate_units_per_ms", 0.0))
    if not facts or horizon <= 0 or rate <= 0:
        return horizon
    service_times = [max(10.0, 10.0 * math.ceil(
        (float(row["resource_units"]) / rate) / 10.0)) for row in facts]
    window = (max(float(row["arrival_ms"]) for row in facts)
              + sum(service_times) + max(service_times))
    return min(horizon, window)


def priority_delivery_auc(result: Mapping[str, object]) -> float | None:
    """Area under delivered rubric priority, independent of receiver history.

    Each successful job contributes its priority weight from completion until
    a policy-independent workload window. An undelivered job contributes zero.
    The window is based on arrivals and offered service, never a policy's
    completion times. It is bounded by the simulator safety horizon.
    """
    facts = {row["job_id"]: row for row in result.get("job_facts", ())}
    horizon = priority_evaluation_window_ms(result)
    maximum = sum(normalized_priority_weight(int(row["base_priority"]))
                  for row in facts.values())
    if horizon <= 0 or maximum <= 0:
        return None
    area = 0.0
    for outcome in result.get("outcomes", ()):
        if outcome.get("status") not in {"completed_on_time", "completed_late"}:
            continue
        completion = outcome.get("completed_ms")
        if completion is None:
            continue
        weight = normalized_priority_weight(
            int(facts[outcome["job_id"]]["base_priority"]))
        area += weight * max(0.0, horizon - float(completion))
    return area / (horizon * maximum)


def jain_fairness(counts: Iterable[int]) -> float | None:
    values = [float(value) for value in counts]
    if not values or sum(values) == 0:
        return None
    return sum(values) ** 2 / (len(values) * sum(value * value for value in values))


def percentile(values: Sequence[float], percentage: float) -> float | None:
    """Dependency-free linear percentile for small episode traces."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def static_ranking_metrics(result: Mapping[str, object]) -> dict[str, float | None]:
    """Score a deliberately static all-at-once scenario, never an arrival stream."""
    facts = {row["job_id"]: row for row in result.get("job_facts", ())}
    if not facts or any(float(row["arrival_ms"]) != 0.0 for row in facts.values()):
        raise ValueError("Static ranking metrics require all jobs to arrive at time zero")
    decisions = tuple(result.get("decision_trace", ()))
    if len(decisions) != len(facts) or len({row["job_id"] for row in decisions}) != len(facts):
        raise ValueError("Static ranking metrics require one decision for every job")
    gains = [BASE_PRIORITY_WEIGHTS[int(facts[row["job_id"]]["base_priority"])]
             for row in decisions]
    def discounted(values: Sequence[float]) -> float:
        return sum(value / math.log2(index + 2.0)
                   for index, value in enumerate(values))
    ideal = discounted(sorted(gains, reverse=True))
    concordant = discordant = truth_ties = 0
    for left in range(len(gains)):
        for right in range(left + 1, len(gains)):
            if gains[left] == gains[right]:
                truth_ties += 1
            elif gains[left] > gains[right]:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    denominator = math.sqrt((comparable + truth_ties) * comparable)
    top_five = gains[:5]
    critical_total = sum(value >= BASE_PRIORITY_WEIGHTS[1] for value in gains)
    critical_top_five = sum(value >= BASE_PRIORITY_WEIGHTS[1]
                            for value in top_five)
    return {
        "ndcg": discounted(gains) / ideal if ideal else None,
        "ndcg_at_5": discounted(top_five) /
        discounted(sorted(gains, reverse=True)[:5]) if top_five else None,
        "critical_recall_at_5": (critical_top_five / critical_total
                                 if critical_total else None),
        "kendall_tau_b": (concordant - discordant) / denominator if denominator else None,
        "priority_pair_agreement": concordant / comparable if comparable else None,
    }


def aggregate_episode_metrics(result: Mapping[str, object]) -> dict[str, float | int | None]:
    """Report post-hoc metrics without exposing evaluator labels to policies."""
    outcomes = tuple(result.get("outcomes", ()))
    completed = [row for row in outcomes if row.get("status") in {"completed_on_time", "completed_late"}]
    on_time = [row for row in outcomes if row.get("status") == "completed_on_time"]
    waits = [float(row["wait_ms"]) for row in completed if row.get("wait_ms") is not None]
    served = Counter(row.get("ue_id") for row in completed)
    facts = {row["job_id"]: row for row in result.get("job_facts", ())}
    all_ues = sorted({row["ue_id"] for row in facts.values()})
    decisions = tuple(result.get("decision_trace", ()))
    latencies = [float(row["terminal_ms"]) - float(row["arrival_ms"])
                 for row in completed if row.get("terminal_ms") is not None]
    inference = [float(row["inference_latency_ms"]) for row in decisions]
    covered = {facts[row["job_id"]]["evidence_group_id"] for row in completed
               if row["job_id"] in facts}
    critical = [row for row in facts.values() if int(row["base_priority"]) <= 1]
    completed_critical = sum(
        row.get("status") == "completed_on_time" and row.get("job_id") in facts
        and int(facts[row["job_id"]]["base_priority"]) <= 1 for row in outcomes)
    # An inversion is counted only when a higher-priority job was already
    # eligible at the same decision point. Future arrivals cannot be inverted.
    inversion_weight = 0.0
    inversion_count = 0
    for decision in decisions:
        selected = facts.get(decision.get("job_id"))
        if selected is None:
            continue
        for candidate in decision.get("candidate_snapshot", ()):
            other = facts.get(candidate.get("job_id"))
            if candidate.get("eligible") and other is not None:
                gap = int(selected["base_priority"]) - int(other["base_priority"])
                if gap > 0:
                    inversion_count += 1
                    inversion_weight += float(gap)
    started = [row for row in outcomes if row.get("started_ms") is not None]
    resource_units = sum(int(row["resource_units"]) for row in started)
    starvation = [float(row["terminal_ms"]) - float(row["arrival_ms"])
                  for row in outcomes if row.get("terminal_ms") is not None]
    utility_points = tuple(result.get("utility_curve", ()))
    horizon = float(result.get("horizon_ms", 0.0))
    max_utility = float(result.get("maximum_utility", 0.0))
    return {
        "jobs": len(outcomes),
        "completed": len(completed),
        "completed_on_time": len(on_time),
        "completed_late": sum(row.get("status") == "completed_late" for row in outcomes),
        "failed_transport": sum(row.get("status") == "failed_transport" for row in outcomes),
        "expired_waiting": sum(row.get("status") == "expired_waiting" for row in outcomes),
        "overflow_drop": sum(row.get("status") == "overflow_drop" for row in outcomes),
        "horizon_censored": sum(row.get("status") == "horizon_censored" for row in outcomes),
        "final_utility": float(result.get("final_utility", 0.0)),
        "episode_reward": float(result.get("episode_reward", 0.0)),
        "normalized_utility_auc": normalized_utility_auc(utility_points, horizon_ms=horizon, maximum_utility=max_utility),
        "priority_delivery_auc": priority_delivery_auc(result),
        "priority_evaluation_window_ms": priority_evaluation_window_ms(result),
        "mean_wait_ms": mean(waits) if waits else None,
        "median_wait_ms": percentile(waits, 50),
        "p95_wait_ms": percentile(waits, 95),
        "max_wait_ms": max(waits) if waits else None,
        "mean_latency_ms": mean(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95),
        "max_starvation_ms": max(starvation) if starvation else None,
        "deadline_miss_rate": sum(row.get("status") in {"completed_late", "expired_waiting"}
                                  for row in outcomes) / len(outcomes) if outcomes else None,
        "critical_on_time_rate": completed_critical / len(critical) if critical else None,
        "distinct_evidence_groups_delivered": len(covered),
        "weighted_priority_inversion": inversion_weight,
        "priority_inversion_count": inversion_count,
        "total_resource_units": resource_units,
        "mean_policy_inference_ms": mean(inference) if inference else None,
        "p95_policy_inference_ms": percentile(inference, 95),
        "jain_fairness": jain_fairness(served.get(ue, 0) for ue in all_ues),
    }
