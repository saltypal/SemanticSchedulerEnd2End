"""Whole-latent scheduling policies over immutable semantic snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np

from INFRA.semantic_contracts import (
    PolicyDecision, ResourceBudgetSemantic, SemanticCandidateSnapshot,
)
from Scheduler.feature_adapter import STATE_DIM, model_state, static_feature_row


class SemanticPolicyBase:
    name = "semantic_base"

    def reset(self, episode_metadata=None) -> None:
        pass

    def observe(self, outcome) -> None:
        pass

    def _pick(self, snapshot: SemanticCandidateSnapshot, scores: Mapping[int, float],
              reason: str) -> PolicyDecision:
        eligible = snapshot.eligible_indices
        if not eligible:
            raise ValueError("Policy received a snapshot with no eligible jobs")
        selected = min(eligible, key=lambda index: (
            -float(scores[index]), snapshot.jobs[index].arrival_ms,
            snapshot.jobs[index].job_id))
        return PolicyDecision(self.name, snapshot.now_ms, selected,
                              float(scores[selected]), reason)


class ClassicalSemanticPolicy(SemanticPolicyBase):
    """FIFO, UE round robin, priority, EDF, SJF and aging comparators."""

    NAMES = ("fifo", "ue_round_robin", "fixed_priority", "edf",
             "priority_edf", "sjf", "aging")

    def __init__(self, name: str, *, priority_by_image: Mapping[str, int] | None = None,
                 aging_threshold_ms: float = 100.0) -> None:
        if name not in self.NAMES:
            raise ValueError(f"Unknown classical semantic policy {name}")
        self.name = name
        self.priority_by_image = dict(priority_by_image or {})
        self.aging_threshold_ms = float(aging_threshold_ms)
        self._last_ue: str | None = None

    def reset(self, episode_metadata=None) -> None:
        self._last_ue = None

    def _priority(self, image_id: str) -> int:
        if image_id not in self.priority_by_image:
            raise ValueError(f"Privileged priority baseline lacks {image_id}")
        return int(self.priority_by_image[image_id])

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        eligible = snapshot.eligible_indices
        if not eligible:
            raise ValueError("No eligible jobs")
        if self.name == "ue_round_robin":
            users = sorted({snapshot.jobs[index].ue_id for index in eligible})
            following = [ue for ue in users if self._last_ue is None or ue > self._last_ue]
            user = (following or users)[0]
            index = min((i for i in eligible if snapshot.jobs[i].ue_id == user),
                        key=lambda i: (snapshot.jobs[i].arrival_ms, snapshot.jobs[i].job_id))
            self._last_ue = user
            return PolicyDecision(self.name, snapshot.now_ms, index, None,
                                  f"next UE {user} in round robin")

        def key(index: int):
            job = snapshot.jobs[index]
            fifo = (job.arrival_ms, job.job_id)
            deadline = float("inf") if job.deadline_ms is None else job.deadline_ms
            if self.name == "fixed_priority":
                return self._priority(job.image_id), *fifo
            if self.name == "priority_edf":
                return self._priority(job.image_id), deadline, *fifo
            if self.name == "edf":
                return deadline, *fifo
            if self.name == "sjf":
                return job.resource_units, *fifo
            if self.name == "aging":
                aged = snapshot.now_ms - job.arrival_ms >= self.aging_threshold_ms
                return (0 if aged else 1), *fifo
            return fifo

        index = min(eligible, key=key)
        return PolicyDecision(self.name, snapshot.now_ms, index, None,
                              f"minimum {self.name} key {key(index)}")


class SemanticGreedyPolicy(SemanticPolicyBase):
    name = "semantic_greedy"

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        scores = {}
        for index in snapshot.eligible_indices:
            job = snapshot.jobs[index]
            values = job.features.normalized_values(cold_start_novelty=1.0)
            mask = job.features.availability_mask()
            scene = values[0] if mask[0] else 0.5
            relevance = values[1] if mask[1] else 0.5
            novelty = values[2] if mask[2] else 1.0
            confidence = values[3] if mask[3] else 0.5
            impact = values[4] if mask[4] else 0.5
            urgency = values[5] if mask[5] else 0.5
            sufficiency = values[6] if mask[6] else 0.5
            cost = job.resource_units / max(1.0, budget.service_rate_units_per_ms * 200.0)
            scores[index] = ((0.25 + 0.75 * scene)
                             * (0.25 + 0.75 * relevance)
                             * (0.25 + 0.75 * novelty)
                             * (0.25 + 0.75 * confidence)
                             * (0.25 + 0.75 * impact)
                             * (0.25 + 0.75 * urgency)
                             * (0.25 + 0.75 * sufficiency) / max(cost, 0.1))
        return self._pick(snapshot, scores, "transparent semantic heuristic")


class SemanticWeightedPolicy(SemanticPolicyBase):
    """Observable-feature weighted-sum semantic priority comparator.

    This is intentionally separate from ``SemanticGreedyPolicy``: the product
    heuristic tests multiplicative interaction, while this policy tests a
    standard additive multi-criteria decision rule. It never reads labels,
    evidence groups, or evaluator-only priority facts.
    """

    name = "semantic_weighted"
    WEIGHTS = (0.18, 0.14, 0.18, 0.10, 0.16, 0.16, 0.08)

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        scores = {}
        for index in snapshot.eligible_indices:
            job = snapshot.jobs[index]
            values = job.features.normalized_values(cold_start_novelty=1.0)
            mask = job.features.availability_mask()
            usable = tuple(value if available else (1.0 if pos == 2 else 0.5)
                           for pos, (value, available) in enumerate(zip(values, mask)))
            cost = job.resource_units / max(1.0, budget.service_rate_units_per_ms * 200.0)
            scores[index] = float(sum(weight * value for weight, value
                                      in zip(self.WEIGHTS, usable))
                                  - 0.10 * min(1.0, cost))
        return self._pick(snapshot, scores, "weighted semantic multi-criteria score")


class SemanticLexicographicPolicy(SemanticPolicyBase):
    """Strict semantic-priority comparator with deadline and FIFO tie-breaks."""

    name = "semantic_lexicographic"

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        if not snapshot.eligible_indices:
            raise ValueError("No eligible jobs")

        def key(index: int):
            job = snapshot.jobs[index]
            values = job.features.normalized_values(cold_start_novelty=1.0)
            mask = job.features.availability_mask()
            usable = tuple(value if available else (1.0 if pos == 2 else 0.5)
                           for pos, (value, available) in enumerate(zip(values, mask)))
            slack = (float("inf") if job.deadline_ms is None else
                     max(0.0, job.deadline_ms - snapshot.now_ms))
            # Higher semantic evidence wins; urgency and impact precede
            # relevance/novelty, then the earliest deadline breaks ties.
            return (-usable[5], -usable[4], -usable[0], -usable[1],
                    -usable[2], -usable[3], -usable[6], slack,
                    job.arrival_ms, job.job_id)

        selected = min(snapshot.eligible_indices, key=key)
        return PolicyDecision(self.name, snapshot.now_ms, selected, None,
                              "lexicographic semantic priority with deadline tie-break")


class PrivilegedMarginalUtilityPolicy(SemanticPolicyBase):
    """Evaluator-only, one-step utility reference with reviewed hidden facts.

    This policy is deliberately privileged. It is a useful myopic reference,
    not an upper bound on the optimal full-episode schedule.
    """

    name = "privileged_marginal_utility"

    def __init__(self, facts_by_image: Mapping[str, Mapping[str, object]]) -> None:
        self.facts_by_image = {image_id: dict(facts) for image_id, facts in facts_by_image.items()}
        self.reset()

    def reset(self, episode_metadata=None) -> None:
        self._best_sufficiency_by_group: dict[str, int] = {}
        self._selected_image_by_job: dict[str, str] = {}

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        scores = {}
        for index in snapshot.eligible_indices:
            job = snapshot.jobs[index]
            if job.image_id not in self.facts_by_image:
                raise ValueError(f"Privileged utility reference lacks {job.image_id}")
            facts = self.facts_by_image[job.image_id]
            priority = int(facts["base_priority"])
            group = str(facts["evidence_group_id"])
            sufficiency = int(facts["evidence_sufficiency"])
            urgency = int(facts["urgency"])
            previous = self._best_sufficiency_by_group.get(group)
            coverage = (1.0 if previous is None else
                        0.25 if sufficiency > previous else 0.05)
            wait_ms = max(0.0, snapshot.now_ms - job.arrival_ms)
            delay = math.exp(-(urgency / 3.0) * wait_ms / 200.0)
            scores[index] = ((4 - priority) / 4.0 * coverage
                             * (1.0 if sufficiency else 0.25) * delay)
        decision = self._pick(snapshot, scores, "privileged one-step evaluator utility")
        selected = snapshot.jobs[decision.action_index]
        self._selected_image_by_job[selected.job_id] = selected.image_id
        return decision

    def observe(self, outcome) -> None:
        image_id = self._selected_image_by_job.pop(outcome.job_id)
        if outcome.status not in {"completed_on_time", "completed_late"}:
            return
        facts = self.facts_by_image[image_id]
        group = str(facts["evidence_group_id"])
        sufficiency = int(facts["evidence_sufficiency"])
        self._best_sufficiency_by_group[group] = max(
            self._best_sufficiency_by_group.get(group, 0), sufficiency)


class LADIClassifierPriorityPolicy(SemanticPolicyBase):
    """Use cached *predicted* labels, never dataset ground-truth labels."""

    name = "ladi_classifier_priority"
    PRIORITY_LABELS = {
        "flooding_structures": 0,
        "buildings_affected_or_greater": 1,
        "roads_damage": 1,
        "debris_any": 2,
        "trees_damage": 2,
    }

    def __init__(self, predictions_by_image: Mapping[str, Mapping[str, float]],
                 *, threshold: float = 0.5) -> None:
        self.predictions = {
            key: {str(label).lower(): float(score) for label, score in value.items()}
            for key, value in predictions_by_image.items()
        }
        self.threshold = float(threshold)

    def predict_priority(self, image_id: str) -> int:
        """Map cached image-model outputs to the frozen task-priority rubric."""
        if image_id not in self.predictions:
            raise ValueError(f"Missing cached classifier prediction for {image_id}")
        probabilities = self.predictions[image_id]
        return min((priority for label, priority in self.PRIORITY_LABELS.items()
                    if probabilities.get(label, 0.0) >= self.threshold), default=3)

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        scores = {}
        for index in snapshot.eligible_indices:
            job = snapshot.jobs[index]
            predicted = self.predict_priority(job.image_id)
            scores[index] = -float(predicted)
        return self._pick(snapshot, scores, "cached LADI predicted labels mapped to task rubric")


class ClipImagePriorityPolicy(SemanticPolicyBase):
    """Schedule using image-only class probabilities trained on frozen CLIP vectors."""

    NAMES = ("clip_image_priority", "clip_image_priority_edf")

    def __init__(self, predictions_by_image: Mapping[str, Mapping[str, float]],
                 *, use_deadline: bool = False) -> None:
        self.name = "clip_image_priority_edf" if use_deadline else "clip_image_priority"
        self.predictions = {image_id: {int(label): float(probability)
                                      for label, probability in row.items()}
                            for image_id, row in predictions_by_image.items()}
        for image_id, row in self.predictions.items():
            if not row or not set(row).issubset({0, 1, 2, 3}):
                raise ValueError(f"Invalid priority classes for {image_id}")
            if any(not math.isfinite(value) or value < 0 for value in row.values()):
                raise ValueError(f"Invalid class probabilities for {image_id}")
            if not math.isclose(sum(row.values()), 1.0, abs_tol=1e-4):
                raise ValueError(f"Class probabilities do not sum to one for {image_id}")

    def predict_priority(self, image_id: str) -> int:
        if image_id not in self.predictions:
            raise ValueError(f"Missing image-only priority prediction for {image_id}")
        return max(self.predictions[image_id], key=self.predictions[image_id].get)

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        if not snapshot.eligible_indices:
            raise ValueError("No eligible jobs")

        def key(index: int):
            job = snapshot.jobs[index]
            if job.image_id not in self.predictions:
                raise ValueError(f"Missing image-only priority prediction for {job.image_id}")
            probabilities = self.predictions[job.image_id]
            expected_class = sum(label * probability for label, probability
                                 in probabilities.items())
            deadline = (float("inf") if job.deadline_ms is None else
                        job.deadline_ms)
            if self.name == "clip_image_priority_edf":
                predicted_class = self.predict_priority(job.image_id)
                return predicted_class, deadline, expected_class, job.arrival_ms, job.job_id
            return expected_class, job.arrival_ms, job.job_id

        selected = min(snapshot.eligible_indices, key=key)
        return PolicyDecision(
            self.name, snapshot.now_ms, selected, -float(key(selected)[0]),
            "lowest image-predicted expected priority class"
            + (" with deadline tie-break" if self.name.endswith("_edf") else ""),
        )


class OrdinalClassifierPolicy(SemanticPolicyBase):
    name = "feature_ordinal_classifier"

    def __init__(self, artifact: str | Path) -> None:
        with np.load(artifact, allow_pickle=False) as data:
            self.coefficients = np.asarray(data["coefficients"], dtype=np.float32)
            self.intercepts = np.asarray(data["intercepts"], dtype=np.float32)
            self.mean = np.asarray(data["mean"], dtype=np.float32)
            self.scale = np.asarray(data["scale"], dtype=np.float32)
            self.classes = np.asarray(data["classes"], dtype=np.float32)
        if (self.coefficients.shape[1] != 14 or self.mean.shape != (14,)
                or self.scale.shape != (14,)):
            raise ValueError("Ordinal classifier feature schema is incompatible")

    def predict_probabilities(self, job) -> np.ndarray:
        row = (static_feature_row(job) - self.mean) / self.scale
        logits = self.coefficients @ row + self.intercepts
        shifted = logits - np.max(logits)
        values = np.exp(shifted)
        return values / values.sum()

    def predict_priority(self, job) -> int:
        return int(self.classes[int(np.argmax(self.predict_probabilities(job)))])

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        scores = {}
        for index in snapshot.eligible_indices:
            probabilities = self.predict_probabilities(snapshot.jobs[index])
            scores[index] = -float(probabilities @ self.classes)
        return self._pick(snapshot, scores, "lowest expected priority class")


class PairwiseRankerPolicy(SemanticPolicyBase):
    name = "pairwise_linear_ranker"

    def __init__(self, artifact: str | Path) -> None:
        with np.load(artifact, allow_pickle=False) as data:
            self.weights = np.asarray(data["weights"], dtype=np.float32)
            self.mean = np.asarray(data["mean"], dtype=np.float32)
            self.scale = np.asarray(data["scale"], dtype=np.float32)
        if self.weights.shape != (14,) or self.mean.shape != (14,) or self.scale.shape != (14,):
            raise ValueError("Pairwise ranker feature schema is incompatible")

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        scores = {index: float(self.weights @ ((static_feature_row(snapshot.jobs[index])
                         - self.mean) / self.scale)) for index in snapshot.eligible_indices}
        return self._pick(snapshot, scores, "linear pairwise preference score")


class NumpyDQNPolicy(SemanticPolicyBase):
    name = "dqn"

    def __init__(self, artifact: str | Path) -> None:
        with np.load(artifact, allow_pickle=False) as data:
            self.weights = tuple(np.asarray(data[key], dtype=np.float32)
                                 for key in ("w1", "b1", "w2", "b2", "w3", "b3"))
        w1, b1, w2, b2, w3, b3 = self.weights
        if (w1.shape != (64, STATE_DIM) or b1.shape != (64,)
                or w2.shape != (32, 64) or b2.shape != (32,)
                or w3.shape != (25, 32) or b3.shape != (25,)):
            raise ValueError(f"DQN checkpoint is incompatible with {STATE_DIM}-64-32-25 schema")

    def q_values(self, snapshot: SemanticCandidateSnapshot,
                 budget: ResourceBudgetSemantic) -> tuple[np.ndarray, np.ndarray]:
        state, mask = model_state(snapshot, budget)
        w1, b1, w2, b2, w3, b3 = self.weights
        hidden1 = np.maximum(w1 @ state + b1, 0.0)
        hidden2 = np.maximum(w2 @ hidden1 + b2, 0.0)
        values = w3 @ hidden2 + b3
        return values, mask

    def decide(self, snapshot: SemanticCandidateSnapshot,
               budget: ResourceBudgetSemantic) -> PolicyDecision:
        values, mask = self.q_values(snapshot, budget)
        if not mask.any():
            raise ValueError("DQN has no valid candidate")
        masked = np.where(mask, values, -np.inf)
        index = int(np.argmax(masked))
        return PolicyDecision(self.name, snapshot.now_ms, index,
                              float(values[index]), "maximum valid DQN Q-value")
