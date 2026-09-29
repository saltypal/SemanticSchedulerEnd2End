import numpy as np
import pytest

from INFRA.event_simulation import run_episode
from INFRA.semantic_contracts import (
    EpisodeJobSpec, EpisodeSpec, FeatureValue, LatentJobSpec,
    ResourceBudgetSemantic, SchedulingFeatureVectorV2,
)
from Scheduler.feature_adapter import STATE_DIM, model_state
from Scheduler.policy_registry import create_semantic_policy


def _features(relevance, impact):
    def value(v):
        return FeatureValue(v, True, "test", "v2")
    return SchedulingFeatureVectorV2(
        value(relevance), FeatureValue.unavailable("test", "v2"),
        value(impact), value(impact), value(1), (1.0, 0.0),
    )


def _episode():
    jobs = []
    for image, relevance, impact, priority in (("low", 0.0, 0, 3), ("high", 1.0, 3, 0)):
        candidate = LatentJobSpec(
            image, image, "ue1", "task", image, 0, 100,
            10, "simulated_units", _features(relevance, impact),
        )
        jobs.append(EpisodeJobSpec(candidate, priority, image))
    return EpisodeSpec("two", "task", tuple(jobs), 42)


@pytest.mark.parametrize("name", ("fifo", "ue_round_robin", "fixed_priority", "edf",
                                  "priority_edf", "sjf", "aging", "semantic_greedy",
                                  "semantic_weighted", "semantic_lexicographic"))
def test_classical_and_semantic_policies_run(name):
    episode = _episode()
    policy = create_semantic_policy(name, priority_by_image={"low": 3, "high": 0})
    result = run_episode(episode, policy, ResourceBudgetSemantic(1.0))
    assert len(result["outcomes"]) == 2
    assert len(result["decision_trace"]) == 2
    if name in ("fixed_priority", "priority_edf", "semantic_greedy",
                "semantic_weighted", "semantic_lexicographic"):
        assert result["decision_trace"][0]["image_id"] == "high"


def test_dqn_state_shape_and_mask_excludes_padded_slots():
    from INFRA.event_simulation import SemanticSchedulingEnvironment
    episode = _episode()
    budget = ResourceBudgetSemantic(1.0)
    snapshot = SemanticSchedulingEnvironment(episode, budget).snapshot()
    state, mask = model_state(snapshot, budget)
    assert state.shape == (STATE_DIM,) == (459,)
    assert mask.shape == (25,)
    assert np.flatnonzero(mask).tolist() == [0, 1]
    assert not any(field in snapshot.__dataclass_fields__ for field in
                   ("base_priority", "evidence_group_id", "future_outcomes"))


def test_privileged_priority_mapping_is_explicit():
    episode = _episode()
    policy = create_semantic_policy("fixed_priority")
    with pytest.raises(ValueError, match="lacks"):
        run_episode(episode, policy, ResourceBudgetSemantic(1.0))


def test_privileged_utility_reference_uses_reviewed_facts_and_tracks_success():
    episode = _episode()
    facts = {
        item.candidate.image_id: {
            "base_priority": item.base_priority,
            "evidence_group_id": item.evidence_group_id,
            "evidence_sufficiency": 1,
            "urgency": 3,
        } for item in episode.jobs
    }
    policy = create_semantic_policy(
        "privileged_marginal_utility", oracle_facts_by_image=facts)
    result = run_episode(episode, policy, ResourceBudgetSemantic(1.0))
    assert result["decision_trace"][0]["image_id"] == "high"
    assert result["history_version"] == 2
