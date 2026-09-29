from SemParam.ladi_prediction_provider import (
    map_ladi_probabilities, semantic_scene_features,
)


def test_ladi_mapping_uses_public_rubric_order():
    stratum, priority = map_ladi_probabilities(
        {"roads_damage": 0.9, "flooding_structures": 0.8}
    )
    assert stratum == "flooding_structures"
    assert priority == 0


def test_ladi_mapping_covers_normal_priority_native_labels():
    assert map_ladi_probabilities({"trees_any": 0.9}) == ("trees_any", 3)


def test_scene_importance_and_confidence_are_bounded_model_proxies():
    scene, confidence = semantic_scene_features({
        "flooding_structures": 0.9, "trees_any": 0.1,
    })
    assert scene == 0.9
    assert 0.0 <= confidence <= 1.0
    _, uncertain = semantic_scene_features({
        "flooding_structures": 0.5, "trees_any": 0.5,
    })
    assert confidence > uncertain
