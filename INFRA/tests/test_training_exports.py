import numpy as np
import pytest

from INFRA.event_simulation import SemanticSchedulingEnvironment
from INFRA.learning.supervised_training import fit_supervised_models
from INFRA.semantic_contracts import (
    EpisodeJobSpec, EpisodeSpec, FeatureValue, LatentJobSpec,
    ResourceBudgetSemantic, SchedulingFeatureVectorV2,
)
from Scheduler.policy_registry import create_semantic_policy


def _item(index, priority):
    def value(number):
        return FeatureValue(float(number), True, "fixture", "v2")
    features = SchedulingFeatureVectorV2(
        value(1.0 - priority / 3), FeatureValue.unavailable("fixture", "v2"),
        value(3 - priority), value(3 - priority), value(1),
        (1.0, 0.0) if index % 2 else (0.0, 1.0),
    )
    candidate = LatentJobSpec(
        f"job{index}", f"image{index}", "ue1", "task", f"latent{index}",
        0, 100, 10, "simulated_units", features,
    )
    return EpisodeJobSpec(candidate, priority, f"group{index}")


def test_supervised_exports_are_consumed_by_runtime_policies(tmp_path):
    items = [_item(index, index % 4) for index in range(8)]
    paths = fit_supervised_models(items, tmp_path)
    episode = EpisodeSpec("demo", "task", tuple(items[:4]), 42)
    snapshot = SemanticSchedulingEnvironment(episode, ResourceBudgetSemantic(1.0)).snapshot()
    for name, artifact in (("feature_ordinal_classifier", paths["classifier"]),
                           ("pairwise_linear_ranker", paths["ranker"])):
        policy = create_semantic_policy(name, model_artifact=artifact)
        decision = policy.decide(snapshot, ResourceBudgetSemantic(1.0))
        assert decision.action_index in snapshot.eligible_indices


def test_dqn_numpy_export_matches_torch_network(tmp_path):
    torch = pytest.importorskip("torch")
    from INFRA.learning.dqn_training import DQNConfig, build_q_network, train_dqn
    from Scheduler.feature_adapter import model_state

    items = tuple(_item(index, index % 4) for index in range(4))
    episode = EpisodeSpec("demo", "task", items, 42)
    budget = ResourceBudgetSemantic(1.0)
    train_dqn(
        [episode] * 3, [episode], budget, tmp_path,
        config=DQNConfig(episodes=3, batch_size=2, replay_capacity=20,
                         replay_warmup=2, target_update_steps=2,
                         validate_every=1, max_seconds=None), device="cpu",
    )
    policy = create_semantic_policy("dqn", model_artifact=tmp_path / "dqn_seed42_best.npz")
    checkpoint = torch.load(tmp_path / "dqn_seed42_best.pt", map_location="cpu",
                            weights_only=False)
    network = build_q_network()
    network.load_state_dict(checkpoint["online"])
    snapshot = SemanticSchedulingEnvironment(episode, budget).snapshot()
    state, _ = model_state(snapshot, budget)
    with torch.no_grad():
        expected = network(torch.from_numpy(state)).numpy()
    actual, mask = policy.q_values(snapshot, budget)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert int(mask.sum()) == 4


def test_dqn_wall_clock_limit_still_exports_valid_best_checkpoint(tmp_path):
    from INFRA.learning.dqn_training import DQNConfig, train_dqn

    items = tuple(_item(index, index % 4) for index in range(4))
    episode = EpisodeSpec("budget-demo", "task", items, 42)
    manifest = train_dqn(
        [episode] * 3, [episode], ResourceBudgetSemantic(1.0), tmp_path,
        config=DQNConfig(episodes=3, batch_size=2, replay_capacity=20,
                         replay_warmup=2, validate_every=3, max_seconds=1e-9),
        device="cpu",
    )
    assert manifest["stopped_by_time_budget"] is True
    assert 1 <= manifest["completed_episodes"] < 3
    assert (tmp_path / "dqn_seed42_best.npz").is_file()
    assert (tmp_path / "dqn_seed42_last.npz").is_file()
