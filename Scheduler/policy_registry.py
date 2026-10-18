"""Single construction point for semantic scheduling comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from Scheduler.semantic_policies import (
    ClassicalSemanticPolicy, ClipImagePriorityPolicy, LADIClassifierPriorityPolicy, NumpyDQNPolicy,
    OrdinalClassifierPolicy, PairwiseRankerPolicy, PrivilegedMarginalUtilityPolicy,
    SemanticGreedyPolicy, SemanticLexicographicPolicy, SemanticWeightedPolicy,
)


DETERMINISTIC_POLICIES = (
    "fifo", "ue_round_robin", "fixed_priority", "edf", "priority_edf",
    "sjf", "aging", "semantic_greedy",
    "semantic_weighted", "semantic_lexicographic",
)
LEARNED_POLICIES = (
    "ladi_classifier_priority", "clip_image_priority", "clip_image_priority_edf",
    "feature_ordinal_classifier",
    "pairwise_linear_ranker", "dqn",
)


def create_semantic_policy(
    name: str, *, priority_by_image: Mapping[str, int] | None = None,
    predictions_by_image: Mapping[str, Mapping[str, float]] | None = None,
    image_priority_predictions_by_image: Mapping[str, Mapping[str, float]] | None = None,
    oracle_facts_by_image: Mapping[str, Mapping[str, object]] | None = None,
    model_artifact: str | Path | None = None,
):
    if name in ClassicalSemanticPolicy.NAMES:
        return ClassicalSemanticPolicy(name, priority_by_image=priority_by_image)
    if name == "semantic_greedy":
        return SemanticGreedyPolicy()
    if name == "semantic_weighted":
        return SemanticWeightedPolicy()
    if name == "semantic_lexicographic":
        return SemanticLexicographicPolicy()
    if name == "privileged_marginal_utility":
        if oracle_facts_by_image is None:
            raise ValueError("Privileged utility reference requires evaluator-only facts")
        return PrivilegedMarginalUtilityPolicy(oracle_facts_by_image)
    if name == "ladi_classifier_priority":
        if predictions_by_image is None:
            raise ValueError("LADI classifier policy requires cached predictions")
        return LADIClassifierPriorityPolicy(predictions_by_image)
    if name in ClipImagePriorityPolicy.NAMES:
        if image_priority_predictions_by_image is None:
            raise ValueError("CLIP image priority policy requires cached class probabilities")
        return ClipImagePriorityPolicy(image_priority_predictions_by_image,
                                       use_deadline=name.endswith("_edf"))
    if model_artifact is None:
        raise ValueError(f"Policy {name} requires a trained model artifact")
    if name == "feature_ordinal_classifier":
        return OrdinalClassifierPolicy(model_artifact)
    if name == "pairwise_linear_ranker":
        return PairwiseRankerPolicy(model_artifact)
    if name == "dqn":
        return NumpyDQNPolicy(model_artifact)
    raise ValueError(f"Unknown semantic policy {name}")
