from SemParam.ladi_prediction_provider import map_ladi_probabilities


def test_ladi_mapping_uses_public_rubric_order():
    stratum, priority = map_ladi_probabilities(
        {"roads_damage": 0.9, "flooding_structures": 0.8}
    )
    assert stratum == "flooding_structures"
    assert priority == 0


def test_ladi_mapping_does_not_call_unknown_normal():
    assert map_ladi_probabilities({"trees_any": 0.9}) == (None, None)
