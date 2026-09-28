"""Single construction point for semantic scheduling comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from Scheduler.semantic_policies import (
    ClassicalSemanticPolicy, LADIClassifierPriorityPolicy, NumpyDQNPolicy,
    OrdinalClassifierPolicy, PairwiseRankerPolicy, PrivilegedMarginalUtilityPolicy,
    SemanticGreedyPolicy,
)


DETERMINISTIC_POLICIES = (
    "fifo", "ue_round_robin", "fixed_priority", "edf", "priority_edf",
    "sjf", "aging", "semantic_greedy",
)
LEARNED_POLICIES = (
    "ladi_classifier_priority", "feature_ordinal_classifier",
    "pairwise_linear_ranker", "dqn",
)


def create_semantic_policy(
    name: str, *, priority_by_image: Mapping[str, int] | None = None,
    predictions_by_image: Mapping[str, Mapping[str, float]] | None = None,
    oracle_facts_by_image: Mapping[str, Mapping[str, object]] | None = None,
    model_artifact: str | Path | None = None,
):
    if name in ClassicalSemanticPolicy.NAMES:
        return ClassicalSemanticPolicy(name, priority_by_image=priority_by_image)
    if name == "semantic_greedy":
        return SemanticGreedyPolicy()
    if name == "privileged_marginal_utility":
        if oracle_facts_by_image is None:
            raise ValueError("Privileged utility reference requires evaluator-only facts")
        return PrivilegedMarginalUtilityPolicy(oracle_facts_by_image)
    if name == "ladi_classifier_priority":
        if predictions_by_image is None:
            raise ValueError("LADI classifier policy requires cached predictions")
        return LADIClassifierPriorityPolicy(predictions_by_image)
    if model_artifact is None:
        raise ValueError(f"Policy {name} requires a trained model artifact")
    if name == "feature_ordinal_classifier":
        return OrdinalClassifierPolicy(model_artifact)
    if name == "pairwise_linear_ranker":
        return PairwiseRankerPolicy(model_artifact)
    if name == "dqn":
        return NumpyDQNPolicy(model_artifact)
    raise ValueError(f"Unknown semantic policy {name}")
