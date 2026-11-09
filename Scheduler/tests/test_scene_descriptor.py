"""Engineering tests for the pooled eight-value RankNet training path.

The manifests in this file are unit fixtures for parsing and mechanics only.
They are created under pytest's temporary directories, never persisted in the
repository and never reported as review data; no reviewed label exists in
this checkout.
"""
import json

import numpy as np
import pytest
import torch

from Scheduler.training.scene_descriptor import (
    PooledSceneProjection,
    ReviewedPairUnavailable,
    RankNetScorer,
    check_group_disjointness,
    exact_uint8,
    fake_uint8,
    load_reviewed_pairs,
    main,
    ordered_pairs,
    pairwise_logistic_loss,
    pool_grid,
    qualification_report,
    swapped_order_metrics,
    train,
)
from Scheduler.training.scene_vlm_review import MODEL_REVISIONS, REVIEW_KIND


def test_pool_grid_concatenates_global_mean_and_max():
    grid = torch.tensor([[[1.0, 3.0], [5.0, 7.0]],
                         [[2.0, 0.0], [4.0, 8.0]]])
    pooled = pool_grid(grid)
    assert pooled.shape == (4,)
    assert pooled[:2].tolist() == pytest.approx([4.0, 3.5])
    assert pooled[2:].tolist() == pytest.approx([7.0, 8.0])
    # Batched single-image grids take the same path.
    assert torch.equal(pool_grid(grid[None]), pooled)
    with pytest.raises(ValueError):
        pool_grid(torch.zeros(2, 2, 2, 2))


def test_projection_stays_bounded_and_supports_one_value_ablation():
    pooled = torch.randn(5, 2 * 16)
    eight = PooledSceneProjection(16, 8)
    out = eight(pooled)
    assert out.shape == (5, 8)
    assert torch.all(out > 0) and torch.all(out < 1)
    one = PooledSceneProjection(16, 1)
    assert one(pooled).shape == (5, 1)


def test_fake_uint8_rounds_to_uint8_grid_and_keeps_gradients():
    values = torch.tensor([0.0, 0.5, 0.5001, 0.999, 1.0], requires_grad=True)
    quantized = fake_uint8(values)
    steps = quantized.detach() * 255
    assert torch.allclose(steps, steps.round(), atol=1e-6)
    quantized.sum().backward()
    assert values.grad is not None and torch.all(values.grad == 1.0)


def test_exact_uint8_matches_fake_quantization_on_eval_points():
    rng = np.random.default_rng(7)
    values = torch.tensor(rng.random(64), dtype=torch.float32)
    exact = exact_uint8(values).astype(np.float32) / 255.0
    assert np.allclose(exact, fake_uint8(values).detach().numpy(), atol=1e-6)


def test_pairwise_logistic_loss_prefers_the_ordered_margin():
    preferred = torch.tensor([2.0, 1.5])
    other = torch.tensor([0.5, 0.0])
    loss = pairwise_logistic_loss(preferred, other)
    assert loss.item() == pytest.approx(float(torch.nn.functional.softplus(
        torch.tensor([-(2.0 - 0.5), -(1.5 - 0.0)]).mean())))
    assert pairwise_logistic_loss(other, preferred).item() > loss.item()
    assert pairwise_logistic_loss(torch.zeros(3), torch.zeros(3)).item() == \
        pytest.approx(float(torch.log(torch.tensor(2.0))))


def _fixture_manifest(tmp_path, *, review_kind="independent_scene_pair_review",
                      pairs=None):
    if pairs is None:
        pairs = [
            {"a": "img-a", "b": "img-b", "grade_a": 2, "grade_b": 1,
             "status": "strict", "reviewer": "unit-fixture", "same_tier": True},
            {"a": "img-a", "b": "img-c", "grade_a": 2, "grade_b": 0,
             "status": "strict", "reviewer": "unit-fixture", "same_tier": True},
        ]
    path = tmp_path / "fixture_manifest.json"
    path.write_text(json.dumps({
        "review_kind": review_kind,
        "grades": [0, 1, 2],
        "pairs": pairs,
    }), encoding="utf-8")
    return path


def test_missing_review_manifest_blocks_the_gate_without_labels(tmp_path):
    with pytest.raises(ReviewedPairUnavailable, match="acceptance gate stays blocked"):
        load_reviewed_pairs(tmp_path / "does_not_exist.json")
    with pytest.raises(ReviewedPairUnavailable):
        train({}, tmp_path / "does_not_exist.json")


def test_non_review_and_native_label_records_are_rejected(tmp_path):
    path = _fixture_manifest(tmp_path, review_kind="road_damage_pilot")
    with pytest.raises(ValueError, match="not scene review"):
        load_reviewed_pairs(path)
    native = tmp_path / "native.json"
    native.write_text(json.dumps({
        "positive_labels": ["roads_damage"],
        "grades": [0, 1, 2],
        "pairs": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="native detector labels"):
        load_reviewed_pairs(native)


def test_status_grade_and_reviewer_validation(tmp_path):
    good = {"a": "x", "b": "y", "grade_a": 2, "grade_b": 1,
            "status": "strict", "reviewer": "r", "same_tier": True}
    broken = dict(good, status="plausible")
    with pytest.raises(ValueError, match="invalid status"):
        load_reviewed_pairs(_fixture_manifest(tmp_path, pairs=[broken]))
    broken = dict(good, grade_a=7)
    with pytest.raises(ValueError, match="outside 0/1/2"):
        load_reviewed_pairs(_fixture_manifest(tmp_path, pairs=[broken]))
    broken = dict(good, reviewer="  ")
    with pytest.raises(ValueError, match="no named reviewer"):
        load_reviewed_pairs(_fixture_manifest(tmp_path, pairs=[broken]))
    broken = {key: value for key, value in good.items() if key != "same_tier"}
    with pytest.raises(ValueError, match="missing review fields"):
        load_reviewed_pairs(_fixture_manifest(tmp_path, pairs=[broken]))


def test_uncertain_tie_cross_tier_and_equal_grade_pairs_are_excluded_and_counted(tmp_path):
    pairs = [
        {"a": "x", "b": "y", "grade_a": 2, "grade_b": 1, "status": "strict",
         "reviewer": "r", "same_tier": True},
        {"a": "x", "b": "z", "grade_a": 2, "grade_b": 1, "status": "uncertain",
         "reviewer": "r", "same_tier": True},
        {"a": "y", "b": "z", "grade_a": 1, "grade_b": 1, "status": "tie",
         "reviewer": "r", "same_tier": True},
        {"a": "x", "b": "w", "grade_a": 2, "grade_b": 0, "status": "strict",
         "reviewer": "r", "same_tier": False},
        {"a": "y", "b": "w", "grade_a": 1, "grade_b": 1, "status": "strict",
         "reviewer": "r", "same_tier": True},
    ]
    manifest = load_reviewed_pairs(_fixture_manifest(tmp_path, pairs=pairs))
    assert len(manifest["strict_pairs"]) == 1
    assert manifest["excluded"] == {"uncertain": 1, "tie": 1, "cross_tier": 1,
                                    "equal_grade": 1}
    preferred, other, image_ids = ordered_pairs(
        [{"a": "low", "b": "high", "grade_a": 0, "grade_b": 2}])
    assert preferred == ["high"] and other == ["low"]
    assert image_ids == ["low", "high"]


def test_training_path_runs_end_to_end_on_fixture(tmp_path):
    """Mechanics only: shapes, quantization, optimization and artifact fields."""
    rng = np.random.default_rng(3)
    grids = {name: rng.random((4, 3, 3), dtype=np.float32)
             for name in ("img-a", "img-b", "img-c")}
    artifact = train(grids, _fixture_manifest(tmp_path), values=8, steps=40,
                     lr=0.2, seed=0)
    assert artifact["values"] == 8 and artifact["channels"] == 4
    assert artifact["pair_counts"]["strict_ordered"] == 2
    assert artifact["training_quantization"] == "fake_uint8"
    assert len(artifact["projection_weight"][0]) == 8
    losses = [entry["loss"] for entry in artifact["history"]]
    assert losses[-1] < losses[0]
    assert "no wait, service time" in artifact["inputs"]
    ablation = train(grids, _fixture_manifest(tmp_path), values=1, steps=5, seed=0)
    assert ablation["values"] == 1


def test_training_path_refuses_to_run_without_reviewed_data(tmp_path):
    grids = {"img-a": np.zeros((4, 3, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match="missing cached pre-mask grids"):
        train(grids, _fixture_manifest(tmp_path))
    assert main(["--manifest", str(tmp_path / "absent.json"),
                 "--grids", str(tmp_path), "--output",
                 str(tmp_path / "out.json")]) == 2


def test_scorer_uses_shared_linear_head_over_compact_descriptor():
    scorer = RankNetScorer(8)
    with torch.no_grad():
        scorer.scorer.weight.fill_(0.5)
        scorer.scorer.bias.zero_()
    scene = torch.tensor([[0.1] * 8, [0.9] * 8])
    scores = scorer(scene)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


# --- Trial 2 preparation: VLM swapped-order metrics and group checks -------
# Fixtures below are pytest-temp mechanics data in the explicit VLM review
# format; they are never persisted as review data and never reported as
# measured model judgments.

MODEL_A, MODEL_B = list(MODEL_REVISIONS)


def _vlm_image(image_id, split, tier=2, **overrides):
    record = {"split": split, "event": "evt-" + image_id,
              "flight": "flt-" + image_id,
              "near_duplicate": "nd-" + image_id, "spi_tier": tier}
    record.update(overrides)
    return record


def _vlm_review(pairs, images, **overrides):
    manifest = {"review_kind": REVIEW_KIND,
                "model_revisions": MODEL_REVISIONS,
                "prompt_version": "fixture-v1",
                "images": images, "pairs": pairs}
    manifest.update(overrides)
    return manifest


def _strict_votes():
    """Forward A / reverse B maps both orders back to image a."""
    return {model: {"forward": "A", "reverse": "B"}
            for model in MODEL_REVISIONS}


def _stable_pair(a, b, votes=None):
    return {"a": a, "b": b, "votes": votes or _strict_votes()}


def _metrics_images_pairs(splits_by_index):
    images, pairs = {}, []
    for index, split in splits_by_index:
        a, b = "p%d-a" % index, "p%d-b" % index
        images[a] = _vlm_image(a, split)
        images[b] = _vlm_image(b, split)
        pairs.append(_stable_pair(a, b))
    return images, pairs


def test_swapped_order_metrics_report_pass_case_without_raising():
    images, pairs = _metrics_images_pairs(
        [(index, "calibration") for index in range(1, 11)]
        + [(11, "train"), (12, "train")])
    metrics = swapped_order_metrics(_vlm_review(pairs, images))
    assert metrics["review_kind"] == REVIEW_KIND
    assert metrics["split"] == "calibration"
    assert metrics["calibration_pairs"] == 10
    assert metrics["pairs_by_split"] == {"calibration": 10, "train": 2}
    assert metrics["pairs_unresolved"] == 0
    for values in metrics["models"].values():
        assert values == {"pairs_two_valid_responses": 10,
                          "order_consistent_pairs": 10,
                          "order_consistency": 1.0,
                          "stable_strict_pairs": 10,
                          "stable_strict_coverage": 1.0}
    joint = metrics["joint"]
    assert joint["overlap_pairs"] == 10
    assert joint["overlap_fraction"] == 1.0
    assert joint["agreed_pairs"] == 10 and joint["agreement"] == 1.0
    assert joint["disagreement_pairs"] == 0
    assert joint["disagreement_examples"] == []
    report = qualification_report(metrics)
    assert report["overlap_pairs"] == 10
    assert report["cross_model_agreement"] == {
        "value": 1.0, "threshold": 0.80, "pass": True}
    assert report["qualified"] is True
    assert all(gate["pass"]
               for per_model in report["models"].values()
               for gate in per_model.values())


def test_swapped_order_metrics_record_shortfall_instead_of_raising():
    images, pairs = _metrics_images_pairs(
        [(index, "calibration") for index in range(1, 8)])
    # Pair 5: both reviewers tie. Pair 6: mapped orders disagree.
    pairs[4]["votes"] = {model: {"forward": "tie", "reverse": "tie"}
                         for model in MODEL_REVISIONS}
    pairs[5]["votes"] = {model: {"forward": "A", "reverse": "A"}
                         for model in MODEL_REVISIONS}
    # Pair 7: one malformed answer from MODEL_A only.
    pairs[6]["votes"] = {MODEL_A: {"forward": "maybe", "reverse": "B"},
                         MODEL_B: {"forward": "A", "reverse": "B"}}
    metrics = swapped_order_metrics(_vlm_review(pairs, images))
    assert metrics["calibration_pairs"] == 7
    model_a = metrics["models"][MODEL_A]
    assert model_a["pairs_two_valid_responses"] == 6
    assert model_a["order_consistent_pairs"] == 5
    assert model_a["order_consistency"] == pytest.approx(5 / 6)
    assert model_a["stable_strict_pairs"] == 4
    assert model_a["stable_strict_coverage"] == pytest.approx(4 / 7)
    model_b = metrics["models"][MODEL_B]
    assert model_b["pairs_two_valid_responses"] == 7
    assert model_b["order_consistency"] == pytest.approx(6 / 7)
    assert model_b["stable_strict_coverage"] == pytest.approx(5 / 7)
    joint = metrics["joint"]
    assert joint["overlap_pairs"] == 4
    assert joint["overlap_fraction"] == pytest.approx(4 / 7)
    assert joint["agreement"] == 1.0
    report = qualification_report(metrics)
    assert report["models"][MODEL_A]["order_consistency"]["pass"] is False
    assert report["models"][MODEL_B]["order_consistency"]["pass"] is False
    assert report["models"][MODEL_A]["stable_strict_coverage"]["pass"] is True
    assert report["cross_model_agreement"]["pass"] is True
    assert report["qualified"] is False

    # No calibration cohort: empty denominators never pass a gate.
    train_images, train_pairs = _metrics_images_pairs([(1, "train")])
    empty = swapped_order_metrics(_vlm_review(train_pairs, train_images))
    assert empty["calibration_pairs"] == 0
    assert empty["models"][MODEL_A]["order_consistency"] is None
    assert empty["models"][MODEL_A]["stable_strict_coverage"] is None
    assert empty["joint"]["agreement"] is None
    assert qualification_report(empty)["qualified"] is False

    # Wrong manifest kind and non-metrics input still raise.
    with pytest.raises(ValueError, match="review_kind"):
        swapped_order_metrics({"review_kind": "independent_scene_pair_review"})
    with pytest.raises(ValueError, match="swapped_order_metrics"):
        qualification_report({"models": {}})


def test_group_disjointness_reports_crossings_without_raising():
    images = {
        "g1": _vlm_image("g1", "train", event="evt-shared"),
        "g2": _vlm_image("g2", "train"),
        "g3": _vlm_image("g3", "development", event="evt-shared"),
        "g4": _vlm_image("g4", "development"),
        "g5": _vlm_image("g5", "train"),
        "g6": _vlm_image("g6", "holdout"),
        "g7": _vlm_image("g7", "train", tier=3),
        "g8": _vlm_image("g8", "train"),
    }
    del images["g5"]["near_duplicate"]     # missing group field
    del images["g8"]["spi_tier"]           # missing tier on a paired image
    pairs = [{"a": "g1", "b": "g2"},        # clean
             {"a": "g1", "b": "g4"},        # spans splits
             {"a": "g1", "b": "g7"},        # tier mismatch
             {"a": "g1", "b": "g8"},        # tier missing
             {"a": "g1", "b": "ghost"}]     # no image record
    report = check_group_disjointness(images, pairs)
    assert report["disjoint"] is False
    assert len(report["violations"]) == 7
    assert {item["type"] for item in report["violations"]} == {
        "malformed_image_record", "group_crosses_split",
        "pair_crosses_split", "pair_tier_mismatch", "pair_tier_missing",
        "pair_missing_image"}
    crossing = next(item for item in report["violations"]
                    if item["type"] == "group_crosses_split")
    assert crossing["dimension"] == "event"
    assert crossing["group"] == "evt-shared"
    assert crossing["splits"] == ["development", "train"]
    assert report["splits"] == {"train": 4, "development": 2}
    assert report["dimensions"] == {"event": 5, "flight": 6,
                                    "near_duplicate": 6}

    clean = {key: dict(images[key]) for key in ("g1", "g2")}
    ok = check_group_disjointness(clean, [{"a": "g1", "b": "g2"}])
    assert ok["disjoint"] is True and ok["violations"] == []
    assert ok["splits"] == {"train": 2}
    assert ok["dimensions"] == {"event": 2, "flight": 2,
                                 "near_duplicate": 2}

    # Cohort freeze merges manifests: one event must not straddle the merge.
    merged = {"cal": _vlm_image("cal", "calibration", event="evt-m"),
              "trn": _vlm_image("trn", "train", event="evt-m")}
    merged_report = check_group_disjointness(merged)
    assert merged_report["disjoint"] is False
    assert any(item["type"] == "group_crosses_split"
               and item["dimension"] == "event"
               for item in merged_report["violations"])
