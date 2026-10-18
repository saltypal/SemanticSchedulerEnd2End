from dataclasses import replace

from INFRA.event_simulation import SemanticSchedulingEnvironment, run_episode
from INFRA.semantic_evaluator import (
    aggregate_episode_metrics, priority_delivery_auc,
    priority_evaluation_window_ms, static_ranking_metrics,
)
from INFRA.semantic_streams import with_transport_probability
from SemParam.semantic_header import SemanticSchedulingHeaderV2Codec
from INFRA.semantic_contracts import (
    EpisodeJobSpec, EpisodeSpec, FeatureValue, LatentJobSpec,
    PolicyDecision, ResourceBudgetSemantic, SchedulingFeatureVectorV2,
)


def _value(value):
    return FeatureValue(value, True, "test", "v2")


def _job(image_id, arrival, *, success=True, group="group", deadline=100):
    features = SchedulingFeatureVectorV2(
        _value(1.0), FeatureValue.unavailable("test", "v2"),
        _value(3), _value(3), _value(1), (1.0, 0.0),
    )
    candidate = LatentJobSpec(
        image_id, image_id, "ue1", "task", image_id, arrival,
        deadline, 10, "simulated_units", features,
    )
    return EpisodeJobSpec(candidate, 0, group, success)


class FirstPolicy:
    name = "first"

    def reset(self, episode_metadata=None):
        pass

    def decide(self, snapshot, budget):
        return PolicyDecision(self.name, snapshot.now_ms,
                              snapshot.eligible_indices[0], None, "first eligible")


def test_history_updates_only_after_success_and_novelty_changes():
    episode = EpisodeSpec("episode", "task", (
        _job("a", 0, success=False),
        _job("b", 0, success=True),
        _job("c", 0, success=True),
    ), 42)
    environment = SemanticSchedulingEnvironment(
        episode, ResourceBudgetSemantic(1.0))
    assert environment.snapshot().jobs[0].features.delivered_history_novelty.available is False
    header_before = SemanticSchedulingHeaderV2Codec().decode(
        environment.snapshot().scheduling_headers[0])
    assert header_before.delivered_history_novelty is None
    environment.step(0)
    assert environment.history_version == 0
    environment.step(0)
    assert environment.history_version == 1
    assert environment.snapshot().jobs[0].features.delivered_history_novelty.value == 0.0
    header_after = SemanticSchedulingHeaderV2Codec().decode(
        environment.snapshot().scheduling_headers[0])
    assert header_after.history_version == 1
    assert header_after.delivered_history_novelty == 0.0
    environment.step(0)
    assert environment.done
    assert environment.history_version == 2


def test_no_history_protocol_keeps_novelty_unavailable_and_gain_independent_of_group():
    episode = EpisodeSpec("no-history", "task", (
        _job("a", 0, group="shared"),
        _job("b", 0, group="shared"),
    ), 42)
    environment = SemanticSchedulingEnvironment(
        episode, ResourceBudgetSemantic(1.0), enable_delivered_history=False)
    assert not environment.snapshot().jobs[0].features.delivered_history_novelty.available
    _, _, _, first = environment.step(0)
    assert environment.history_version == 0
    assert not environment.snapshot().jobs[0].features.delivered_history_novelty.available
    _, _, _, second = environment.step(0)
    assert environment.history_version == 0
    assert second["utility_delta"] > 0.25 * first["utility_delta"]


def test_policies_replay_identical_episode_without_mutating_it():
    episode = EpisodeSpec("episode", "task", (
        _job("a", 0, group="a"), _job("b", 0, group="b")
    ), 42)
    budget = ResourceBudgetSemantic(1.0)
    first = run_episode(episode, FirstPolicy(), budget)
    second = run_episode(episode, FirstPolicy(), budget)
    assert first["outcomes"] == second["outcomes"]
    assert [row["job_id"] for row in first["decision_trace"]] == [
        row["job_id"] for row in second["decision_trace"]]
    assert tuple(row["status"] for row in first["outcomes"]) == (
        "completed_on_time", "completed_on_time")
    assert first["history_version"] == 2
    assert episode.jobs[0].candidate.features.delivered_history_novelty.available is False


def test_failed_channel_does_not_create_semantic_utility():
    episode = EpisodeSpec("episode", "task", (_job("a", 0, success=False),), 42)
    result = run_episode(episode, FirstPolicy(), ResourceBudgetSemantic(1.0))
    assert result["final_utility"] == 0.0
    assert result["outcomes"][0]["status"] == "failed_transport"
    assert result["history_version"] == 0


def test_policy_snapshot_hides_embedding_and_ablation_does_not_change_reward():
    episode = EpisodeSpec("episode", "task", (
        _job("a", 0, group="a"), _job("b", 0, group="b")), 42)
    budget = ResourceBudgetSemantic(1.0)
    snapshot = SemanticSchedulingEnvironment(episode, budget).snapshot()
    assert all(job.features.embedding == () for job in snapshot.jobs)
    complete = run_episode(episode, FirstPolicy(), budget)
    ablated = run_episode(episode, FirstPolicy(), budget,
                          visible_features=("context_relevance", "delivered_history_novelty"))
    assert complete["final_utility"] == ablated["final_utility"]
    availability = ablated["decision_trace"][0]["candidate_snapshot"][0]["availability"]
    assert availability[1:3] == (True, False)
    assert sum(availability) == 1


def test_static_ranking_and_aggregate_metrics_have_evaluator_only_facts():
    high = _job("high", 0, group="high")
    low = replace(_job("low", 0, group="low"), base_priority=3)
    episode = EpisodeSpec("static", "task", (high, low), 42)
    result = run_episode(episode, FirstPolicy(), ResourceBudgetSemantic(1.0))
    ranking = static_ranking_metrics(result)
    assert ranking["ndcg"] == 1.0
    assert ranking["ndcg_at_5"] == 1.0
    assert ranking["critical_recall_at_5"] == 1.0
    assert ranking["priority_pair_agreement"] == 1.0
    metrics = aggregate_episode_metrics(result)
    assert metrics["distinct_evidence_groups_delivered"] == 2
    assert metrics["weighted_priority_inversion"] == 0


def test_priority_delivery_auc_rewards_earlier_critical_delivery():
    high = _job("z-high", 0, group="high")
    low = replace(_job("a-low", 0, group="low"), base_priority=3)
    episode = EpisodeSpec("static", "task", (low, high), 42, horizon_ms=100)

    class HighFirst(FirstPolicy):
        name = "high_first"

        def decide(self, snapshot, budget):
            preferred = next((i for i, job in enumerate(snapshot.jobs)
                              if job.image_id == "z-high"), snapshot.eligible_indices[0])
            return PolicyDecision(self.name, snapshot.now_ms,
                                  preferred, None, "critical first")

    budget = ResourceBudgetSemantic(1.0)
    fifo = run_episode(episode, FirstPolicy(), budget)
    semantic = run_episode(episode, HighFirst(), budget)
    assert priority_delivery_auc(semantic) > priority_delivery_auc(fifo)
    assert aggregate_episode_metrics(semantic)["priority_delivery_auc"] == priority_delivery_auc(semantic)
    assert priority_evaluation_window_ms(semantic) == 30.0
    assert priority_delivery_auc(semantic) == 0.6
    assert priority_delivery_auc(fifo) == 0.4


def test_transport_tape_is_stable_and_policy_independent():
    episode = EpisodeSpec("episode", "task", tuple(
        _job(str(index), 0) for index in range(10)), 123)
    first = with_transport_probability(episode, 0.9)
    second = with_transport_probability(episode, 0.9)
    assert first == second
    assert all(not job.transport_success for job in with_transport_probability(episode, 0).jobs)
    assert all(job.transport_success for job in with_transport_probability(episode, 1).jobs)


def test_horizon_at_service_completion_closes_queue_without_another_decision():
    episode = EpisodeSpec("horizon", "task", (
        _job("first", 0), _job("second", 0)), 42, horizon_ms=10)
    result = run_episode(episode, FirstPolicy(), ResourceBudgetSemantic(1.0))
    assert len(result["decision_trace"]) == 1
    assert {row["job_id"]: row["status"] for row in result["outcomes"]} == {
        "first": "completed_on_time", "second": "horizon_censored"}
