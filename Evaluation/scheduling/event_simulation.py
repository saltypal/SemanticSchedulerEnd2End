"""Event simulation for non-preemptive whole-latent semantic scheduling.

Only this environment changes queues, time, and job lifecycle. Policy input is
an immutable snapshot, and evaluator-only priority/group fields stay private.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from time import perf_counter_ns
from typing import Protocol

from Evaluation.scheduling.semantic_contracts import (
    FEATURE_NAMES, EpisodeJobSpec, EpisodeSpec, FeatureValue, PolicyDecision,
    ResourceBudgetSemantic, SemanticCandidateSnapshot, ServiceOutcome,
)
from Evaluation.evaluation.semantic import (
    delivery_utility, normalized_priority_weight, transition_reward,
)
from Evaluation.scheduling.requests import encode_scheduling_request
from Archived.owners_before_simplification_20261027.SemParam.history.delivered import DeliveredKnowledgeState, novelty_from_history


class SemanticPolicy(Protocol):
    name: str

    def reset(self, episode_metadata: object = None) -> None: ...
    def decide(self, snapshot: SemanticCandidateSnapshot, budget: ResourceBudgetSemantic) -> PolicyDecision: ...


class SemanticSchedulingEnvironment:
    """A deterministic, replayable episode with one active transmission."""

    def __init__(self, episode: EpisodeSpec, budget: ResourceBudgetSemantic,
                 *, visible_features: tuple[str, ...] = FEATURE_NAMES,
                 cost_coefficient: float = 0.1,
                 enable_delivered_history: bool = True) -> None:
        self.episode = episode
        self.budget = budget
        if len(set(visible_features)) != len(visible_features) or set(visible_features) - set(FEATURE_NAMES):
            raise ValueError("visible_features contains duplicate or unknown fields")
        self.visible_features = frozenset(visible_features)
        if not math.isfinite(cost_coefficient) or cost_coefficient < 0:
            raise ValueError("cost_coefficient must be finite and nonnegative")
        self.cost_coefficient = cost_coefficient
        self.enable_delivered_history = enable_delivered_history
        self._ordered = tuple(sorted(episode.jobs, key=lambda item: (
            item.candidate.arrival_ms, item.candidate.job_id)))
        units = {item.candidate.resource_unit for item in self._ordered}
        if units != {budget.resource_unit}:
            raise ValueError(f"Episode resource units {units} differ from budget {budget.resource_unit}")
        if any(not item.candidate.features.embedding for item in self._ordered):
            raise ValueError("Every semantic job needs a cached image embedding")
        self._max_cost = max(item.candidate.resource_units for item in self._ordered)
        self.reset()

    def reset(self) -> SemanticCandidateSnapshot | None:
        self.now_ms = 0.0
        self._next_arrival = 0
        self._waiting: dict[str, EpisodeJobSpec] = {}
        self._outcomes: dict[str, ServiceOutcome] = {}
        self._trace: list[dict[str, object]] = []
        self._utility_curve: list[tuple[float, float]] = [(0.0, 0.0)]
        self._utility = 0.0
        self._history = DeliveredKnowledgeState(self.episode.task_id)
        self._served = {item.candidate.ue_id: 0 for item in self._ordered}
        self._done = False
        self._admit_at_current_time()
        self._advance_idle()
        return self.snapshot()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def history_version(self) -> int:
        return self._history.snapshot().history_version

    def _record(self, item: EpisodeJobSpec, status: str, *,
                started_ms: float | None = None, completed_ms: float | None = None,
                utility_delta: float = 0.0) -> ServiceOutcome:
        candidate = item.candidate
        completion = completed_ms
        outcome = ServiceOutcome(
            job_id=candidate.job_id,
            status=status,
            started_ms=started_ms,
            completed_ms=completion,
            wait_ms=None if started_ms is None else started_ms - candidate.arrival_ms,
            deadline_met=None if completion is None or candidate.deadline_ms is None else completion <= candidate.deadline_ms,
            transport_success=(status in {"completed_on_time", "completed_late"}) if started_ms is not None else None,
            utility_delta=utility_delta,
        )
        self._outcomes[candidate.job_id] = outcome
        self._trace.append({
            **asdict(outcome), "ue_id": candidate.ue_id, "image_id": candidate.image_id,
            "arrival_ms": candidate.arrival_ms, "deadline_ms": candidate.deadline_ms,
            "resource_units": candidate.resource_units,
            "terminal_ms": self.now_ms,
        })
        return outcome

    def _expire_waiting(self) -> list[EpisodeJobSpec]:
        expired = [item for item in self._waiting.values()
                   if item.candidate.deadline_ms is not None and item.candidate.deadline_ms <= self.now_ms]
        for item in sorted(expired, key=lambda row: row.candidate.job_id):
            self._waiting.pop(item.candidate.job_id)
            self._record(item, "expired_waiting")
        return expired

    def _admit_at_current_time(self) -> None:
        while (self._next_arrival < len(self._ordered)
               and self._ordered[self._next_arrival].candidate.arrival_ms <= self.now_ms + 1e-9):
            item = self._ordered[self._next_arrival]
            self._next_arrival += 1
            if len(self._waiting) >= self.episode.queue_capacity:
                self._record(item, "overflow_drop")
            else:
                self._waiting[item.candidate.job_id] = item

    def _next_waiting_deadline(self) -> float:
        return min((item.candidate.deadline_ms for item in self._waiting.values()
                    if item.candidate.deadline_ms is not None), default=math.inf)

    def _next_arrival_time(self) -> float:
        if self._next_arrival >= len(self._ordered):
            return math.inf
        return self._ordered[self._next_arrival].candidate.arrival_ms

    def _service_ms(self, item: EpisodeJobSpec) -> float:
        raw = item.candidate.resource_units / self.budget.service_rate_units_per_ms
        return float(max(10, math.ceil(raw / 10.0) * 10))

    def _eligible(self, item: EpisodeJobSpec) -> bool:
        limit = self.budget.max_service_ms
        return limit is None or self._service_ms(item) <= limit

    def _advance_idle(self) -> None:
        while not self._done:
            if self.now_ms >= self.episode.horizon_ms:
                self._censor_remaining()
                return
            if any(self._eligible(item) for item in self._waiting.values()):
                return
            if len(self._outcomes) == len(self._ordered):
                self._done = True
                return
            event_time = min(self._next_arrival_time(), self._next_waiting_deadline(), self.episode.horizon_ms)
            if event_time == math.inf or event_time >= self.episode.horizon_ms:
                self.now_ms = self.episode.horizon_ms
                self._censor_remaining()
                return
            if event_time <= self.now_ms + 1e-9:
                raise RuntimeError("Simulator could not advance the idle clock")
            self.now_ms = event_time
            self._expire_waiting()
            self._admit_at_current_time()
            self._expire_waiting()

    def _censor_remaining(self) -> None:
        for item in tuple(self._waiting.values()):
            self._record(item, "horizon_censored")
        self._waiting.clear()
        while self._next_arrival < len(self._ordered):
            item = self._ordered[self._next_arrival]
            self._next_arrival += 1
            self._record(item, "horizon_censored")
        self._done = True

    def _advance_busy_before(self, completion_ms: float) -> list[EpisodeJobSpec]:
        """Admit and expire waiting jobs at events strictly before completion."""
        newly_expired: list[EpisodeJobSpec] = []
        while True:
            event_time = min(self._next_arrival_time(), self._next_waiting_deadline())
            if event_time >= completion_ms - 1e-9:
                break
            if event_time <= self.now_ms + 1e-9:
                raise RuntimeError("Simulator could not advance the busy clock")
            self.now_ms = event_time
            newly_expired.extend(self._expire_waiting())
            self._admit_at_current_time()
            newly_expired.extend(self._expire_waiting())
        return newly_expired

    def snapshot(self) -> SemanticCandidateSnapshot | None:
        if self._done:
            return None
        history = self._history.snapshot()
        jobs = []
        eligible = []
        for item in sorted(self._waiting.values(), key=lambda row: (
                row.candidate.arrival_ms, row.candidate.job_id)):
            if self.enable_delivered_history:
                novelty = novelty_from_history(
                    image_embedding=item.candidate.features.embedding, history=history)
                updated = FeatureValue(
                    value=novelty.value, available=novelty.available,
                    provenance=f"{novelty.source}:{novelty.method}",
                    extractor_version=novelty.version,
                    latency_ms=0.0 if novelty.latency_ms is None else float(novelty.latency_ms),
                )
            else:
                updated = FeatureValue.unavailable("history_disabled", "track-a/v2")
            visible_features = replace(
                item.candidate.features.with_novelty(updated), embedding=())
            hidden = {name: FeatureValue.unavailable("policy_ablation", "track-a/v2")
                      for name in FEATURE_NAMES if name not in self.visible_features}
            if hidden:
                visible_features = replace(visible_features, **hidden)
            jobs.append(replace(item.candidate, features=visible_features))
            eligible.append(self._eligible(item))
        return SemanticCandidateSnapshot(
            now_ms=self.now_ms, jobs=tuple(jobs), eligible_mask=tuple(eligible),
            history_version=history.history_version,
            delivered_count=history.successful_delivery_count,
            queue_capacity=self.episode.queue_capacity,
            per_ue_served=tuple(sorted(self._served.items())),
            scheduling_headers=tuple(encode_scheduling_request(
                job, history_version=history.history_version,
                delivered_count=history.successful_delivery_count) for job in jobs),
        )

    def step(self, action_index: int) -> tuple[SemanticCandidateSnapshot | None, float, bool, dict[str, object]]:
        snapshot = self.snapshot()
        if snapshot is None:
            raise RuntimeError("Episode has finished")
        if type(action_index) is not int or action_index not in snapshot.eligible_indices:
            raise ValueError(f"Invalid action {action_index}; eligible={snapshot.eligible_indices}")
        selected = snapshot.jobs[action_index]
        item = self._waiting.pop(selected.job_id)
        started_ms = self.now_ms
        completion_ms = started_ms + self._service_ms(item)
        normalized_cost = item.candidate.resource_units / self._max_cost
        if completion_ms > self.episode.horizon_ms:
            self._advance_busy_before(self.episode.horizon_ms)
            self.now_ms = self.episode.horizon_ms
            outcome = self._record(item, "horizon_censored", started_ms=started_ms)
            self._censor_remaining()
            reward = -self.cost_coefficient * normalized_cost
            return None, reward, True, {"outcome": outcome, "history_version": self.history_version}

        expired = self._advance_busy_before(completion_ms)
        self.now_ms = completion_ms
        breakdown = None
        if item.transport_success:
            previous = (self._history.snapshot().best_sufficiency_by_group.get(
                item.evidence_group_id) if self.enable_delivered_history else None)
            breakdown = delivery_utility(
                item, wait_ms=started_ms - item.candidate.arrival_ms,
                previous_group_sufficiency=previous)
            sufficiency = item.candidate.features.evidence_sufficiency.value
            if self.enable_delivered_history:
                self._history.record_success(
                    image_id=item.candidate.image_id,
                    image_embedding=item.candidate.features.embedding,
                    delivered_at_ms=completion_ms,
                    evidence_group_id=item.evidence_group_id,
                    evidence_sufficiency=None if sufficiency is None else int(sufficiency),
                )
            self._utility += breakdown.utility_delta
            self._utility_curve.append((completion_ms, self._utility))
            status = "completed_on_time" if (item.candidate.deadline_ms is None
                     or completion_ms <= item.candidate.deadline_ms) else "completed_late"
        else:
            status = "failed_transport"
        outcome = self._record(item, status, started_ms=started_ms,
                               completed_ms=completion_ms,
                               utility_delta=0.0 if breakdown is None else breakdown.utility_delta)
        self._served[item.candidate.ue_id] += 1
        expired.extend(self._expire_waiting())
        self._admit_at_current_time()
        expired.extend(self._expire_waiting())
        reward = transition_reward(
            breakdown, normalized_service_cost=normalized_cost,
            newly_expired=expired, failed_job=None if item.transport_success else item,
            cost_coefficient=self.cost_coefficient)
        self._advance_idle()
        return self.snapshot(), reward, self._done, {
            "outcome": outcome, "history_version": self.history_version,
            "expired_job_ids": tuple(job.candidate.job_id for job in expired),
            "utility_delta": outcome.utility_delta,
        }

    def result(self, *, policy_name: str, decision_trace: list[dict[str, object]] | None = None) -> dict[str, object]:
        if not self._done:
            raise RuntimeError("Episode must finish before results are exported")
        # Utility is accumulated with normalized priority weights (critical=1,
        # emergency=.75, moderate=.5, normal=.25). Keep the AUC denominator
        # in the same units; using the raw 4/3/2/1 weights here would depress
        # every normalized score by exactly a factor of four.
        upper = sum(normalized_priority_weight(item.base_priority) for item in self._ordered)
        return {
            "episode_id": self.episode.episode_id,
            "policy": policy_name,
            "seed": self.episode.seed,
            "outcomes": tuple(self._trace),
            "utility_curve": tuple(self._utility_curve),
            "final_utility": self._utility,
            "maximum_utility": upper,
            "horizon_ms": self.episode.horizon_ms,
            "service_rate_units_per_ms": self.budget.service_rate_units_per_ms,
            "decision_trace": tuple(decision_trace or ()),
            "history_version": self.history_version,
            "enable_delivered_history": self.enable_delivered_history,
            "job_facts": tuple({
                "job_id": item.candidate.job_id,
                "image_id": item.candidate.image_id,
                "ue_id": item.candidate.ue_id,
                "base_priority": item.base_priority,
                "evidence_group_id": item.evidence_group_id,
                "arrival_ms": item.candidate.arrival_ms,
                "deadline_ms": item.candidate.deadline_ms,
                "resource_units": item.candidate.resource_units,
                "resource_unit": item.candidate.resource_unit,
            } for item in self._ordered),
        }


def run_episode(episode: EpisodeSpec, policy: SemanticPolicy,
                budget: ResourceBudgetSemantic,
                *, visible_features: tuple[str, ...] = FEATURE_NAMES,
                cost_coefficient: float = 0.1,
                enable_delivered_history: bool = True) -> dict[str, object]:
    """Run one policy on an immutable episode and return complete traces."""
    environment = SemanticSchedulingEnvironment(
        episode, budget, visible_features=visible_features,
        cost_coefficient=cost_coefficient,
        enable_delivered_history=enable_delivered_history)
    policy.reset({"episode_id": episode.episode_id, "seed": episode.seed})
    decisions: list[dict[str, object]] = []
    while not environment.done:
        snapshot = environment.snapshot()
        if snapshot is None:
            break
        started = perf_counter_ns()
        decision = policy.decide(snapshot, budget)
        inference_ms = (perf_counter_ns() - started) / 1_000_000.0
        if decision.action_index not in snapshot.eligible_indices:
            raise ValueError(f"Policy {policy.name} chose an ineligible action")
        chosen = snapshot.jobs[decision.action_index]
        next_snapshot, reward, done, info = environment.step(decision.action_index)
        decisions.append({
            "time_ms": snapshot.now_ms, "job_id": chosen.job_id,
            "image_id": chosen.image_id, "ue_id": chosen.ue_id,
            "action_index": decision.action_index, "score": decision.score,
            "reason": decision.reason, "inference_latency_ms": inference_ms,
            "history_version": snapshot.history_version, "reward": reward,
            "outcome_status": info["outcome"].status,
            "header_size_bytes": len(snapshot.scheduling_headers[decision.action_index]),
            "header_hex": snapshot.scheduling_headers[decision.action_index].hex(),
            "candidate_snapshot": tuple({
                "job_id": item.job_id,
                "image_id": item.image_id,
                "ue_id": item.ue_id,
                "eligible": snapshot.eligible_mask[index],
                "features": item.features.normalized_values(),
                "raw_features": tuple(value.value for value in item.features.values),
                "availability": item.features.availability_mask(),
                "resource_units": item.resource_units,
                "deadline_ms": item.deadline_ms,
            } for index, item in enumerate(snapshot.jobs)),
        })
        if hasattr(policy, "observe"):
            policy.observe(info["outcome"])
        if done:
            break
    result = environment.result(policy_name=policy.name, decision_trace=decisions)
    result["visible_features"] = visible_features
    result["episode_reward"] = sum(float(row["reward"]) for row in decisions)
    result["cost_coefficient"] = cost_coefficient
    return result
