from __future__ import annotations

import csv
from pathlib import Path

import pytest

from INFRA.class_priority_profiles import (
    CLASS_RUBRIC, PRESENTATION_UE_PROFILES, validate_manifest_rubric,
)
from INFRA.event_simulation import SemanticSchedulingEnvironment
from INFRA.learning.priority_match_objective import (
    PriorityPairCounts, class_order_reward, count_selected_priority_pairs,
)
from INFRA.semantic_contracts import (
    EpisodeJobSpec, EpisodeSpec, FeatureValue, LatentJobSpec,
    ResourceBudgetSemantic, SchedulingFeatureVectorV2, ServiceOutcome,
)


def _job(index: int, priority: int) -> EpisodeJobSpec:
    field = FeatureValue(1.0, True, "test", "v2")
    features = SchedulingFeatureVectorV2(
        field, field, field, field, field, embedding=(1.0, 0.0),
    )
    candidate = LatentJobSpec(
        f"job-{index}", f"image-{index}", "ue_drone", "task", f"latent-{index}",
        0.0, 100.0, 10, "simulated_units", features,
    )
    return EpisodeJobSpec(candidate, priority, f"group-{index}")


def test_pair_count_only_compares_different_priority_eligible_jobs():
    jobs = (_job(0, 0), _job(1, 1), _job(2, 3), _job(3, 3))
    episode = EpisodeSpec("pair-test", "task", jobs, seed=42)
    snapshot = SemanticSchedulingEnvironment(episode, ResourceBudgetSemantic(1.0)).snapshot()
    assert snapshot is not None
    priorities = {job.candidate.job_id: job.base_priority for job in jobs}
    assert count_selected_priority_pairs(snapshot, 0, priorities) == PriorityPairCounts(3, 0)
    assert count_selected_priority_pairs(snapshot, 2, priorities) == PriorityPairCounts(0, 2)
    with pytest.raises(ValueError, match="not eligible"):
        count_selected_priority_pairs(snapshot, 100, priorities)


def test_class_reward_uses_order_and_delivery_outcome():
    on_time = ServiceOutcome("job-0", "completed_on_time", 0.0, 10.0, 0.0, True, True)
    assert class_order_reward(PriorityPairCounts(3, 0), on_time, 0) == 1.25
    assert class_order_reward(PriorityPairCounts(0, 2), on_time, 0) == -0.75
    assert class_order_reward(PriorityPairCounts(0, 0), on_time, 0) == 0.25


def test_presentation_class_rubric_matches_curated_manifest_when_available():
    root = Path(__file__).resolve().parents[2]
    path = root / "results/ladi-curation/v2/ladi260_semantic_scheduler_v2.csv"
    if not path.is_file():
        pytest.skip("Curated local manifest is not part of the source checkout")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest_rubric(rows)
    assert len(CLASS_RUBRIC) == 12
    assert len(PRESENTATION_UE_PROFILES) == 5


def test_dqn_can_train_with_class_alignment_objective(tmp_path):
    pytest.importorskip("torch")
    from INFRA.learning.dqn_training import DQNConfig, train_dqn

    episode = EpisodeSpec("train-pair", "task", (_job(0, 0), _job(1, 2), _job(2, 3)), seed=42)
    config = DQNConfig(
        objective="class_priority_alignment", episodes=2, batch_size=2,
        replay_capacity=10, replay_warmup=2, validate_every=1,
        max_seconds=None,
    )
    manifest = train_dqn([episode] * 2, [episode], ResourceBudgetSemantic(1.0),
                         tmp_path, config=config, device="cpu")
    assert manifest["config"]["objective"] == "class_priority_alignment"
    assert 0 <= manifest["best_validation_priority_match_rate"] <= 1
    assert (tmp_path / "dqn_seed42_best.npz").is_file()
