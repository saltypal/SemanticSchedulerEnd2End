"""Tier-mapping rules for parameters.edge.labels (plan v3.1 sec. 3)."""
import json

import pytest

from parameters.edge import labels

RULES = labels._rules(labels.load_config("data")["tier_rules"])


def tier(**kw):
    sets = {k: ({v} if isinstance(v, str) else set(v)) for k, v in kw.items()}
    return labels.assign_tier(sets, RULES)


@pytest.mark.parametrize("kw,expected", [
    (dict(damage="severe_damage"), "Critical"),
    (dict(humanitarian="injured_or_dead_people"), "Critical"),
    (dict(humanitarian="missing_or_found_people", informative="informative"), "Critical"),
    (dict(damage="mild_damage"), "High"),
    (dict(humanitarian="vehicle_damage"), "High"),
    (dict(humanitarian="affected_individuals", informative="informative"), "Medium"),   # plan v3 sec. 3
    (dict(damage="little_or_no_damage"), "Medium"),
    (dict(humanitarian="other_relevant_information", informative="informative"), "Medium"),
    (dict(humanitarian="rescue_volunteering_or_donation_effort", informative="informative"), "Medium"),
    (dict(humanitarian="not_humanitarian", informative="not_informative"), "Low"),
    (dict(), "Low"),
])
def test_tier_rules(kw, expected):
    assert tier(**kw)[0] == expected


def test_first_match_wins_most_severe():
    assert tier(damage="mild_damage", humanitarian="injured_or_dead_people")[0] == "Critical"
    assert tier(damage="little_or_no_damage", humanitarian="infrastructure_and_utility_damage")[0] == "High"


def test_medium_via_humanitarian_needs_informative():
    got, flags = tier(humanitarian="other_relevant_information", informative="not_informative")
    assert got == "Low" and "humanitarian_vs_informative_conflict" in flags


def test_damage_label_keeps_medium_even_if_flag_says_not_informative():
    got, flags = tier(damage="little_or_no_damage", informative="not_informative")
    assert got == "Medium" and "informative_flag_overridden" in flags


def test_conflicting_labels_are_flagged_and_take_the_most_severe():
    got, flags = tier(damage={"mild_damage", "severe_damage"})
    assert got == "Critical" and "label_conflict" in flags


def test_alternative_mapping_promotes_affected_individuals():
    """Plan v3 sec. 3: affected_individuals is Medium in the main mapping and High in the sensitivity mapping."""
    cfg = labels.load_config("data")
    alt = labels._rules(cfg["tier_rules"], "affected_high", cfg["alternative_mappings"])
    lab = {"humanitarian": {"affected_individuals"}, "informative": {"informative"}}
    assert labels.assign_tier(lab, alt)[0] == "High"
    assert labels.assign_tier(lab, RULES)[0] == "Medium"


def test_retier_rows_only_changes_crisismmd():
    rows = [{"source": "crisismmd", "tier": "High", "damage": [], "humanitarian": ["affected_individuals"],
             "informative": ["informative"], "flags": []},
            {"source": "places365", "tier": "Low", "damage": [], "humanitarian": [], "informative": [], "flags": []}]
    out, changed = labels.retier_rows(rows)
    assert [r["tier"] for r in out] == ["Medium", "Low"] and changed == {("High", "Medium"): 1}


def test_normals_are_low_and_blacklist_is_dropped():
    items = [{"path": "a.jpg", "category": "kitchen"}, {"path": "b.jpg", "category": "ruin"},
             {"path": "c.jpg", "category": "landfill"}]
    rows, dropped = labels.normal_rows(items, "places365")
    assert [r["image_path"] for r in rows] == ["a.jpg"] and dropped == 2
    assert rows[0]["tier"] == "Low" and rows[0]["event"] is None


def test_blacklist_applies_only_to_places365():
    rows, dropped = labels.normal_rows([{"path": "p.jpg", "category": "ruin"}], "coco_people")
    assert len(rows) == 1 and dropped == 0


def _write(tmp, task, split, rows):
    (tmp / "{}_{}.json".format(task, split)).write_text(json.dumps(rows), encoding="utf-8")


def test_load_and_table_end_to_end(tmp_path):
    base = dict(event_name="hurricane_maria", tweet_id=1, image_id="1_0", tweet_text="t")
    _write(tmp_path, "damage", "train", [dict(base, image_path="a.jpg", label="severe_damage")])
    _write(tmp_path, "humanitarian", "train", [
        dict(base, image_path="a.jpg", label="infrastructure_and_utility_damage"),
        dict(base, image_path="b.jpg", label="not_humanitarian", tweet_id=2)])
    _write(tmp_path, "informative", "dev", [dict(base, image_path="b.jpg", label="not_informative", tweet_id=2)])
    rows = labels.label_table(labels.load_crisismmd(tmp_path))
    assert {r["image_path"]: r["tier"] for r in rows} == {"a.jpg": "Critical", "b.jpg": "Low"}
    report = labels.report_counts(rows)
    assert report["per_event_tier"]["hurricane_maria"]["Critical"] == 1


def test_missing_metadata_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        labels.load_crisismmd(tmp_path)
