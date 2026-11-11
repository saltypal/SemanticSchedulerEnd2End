from parameters.edge.freeze_splits import freeze

CFG = {"train_events": ["e_tr"], "val_events": ["e_va"], "test_events": ["e_te"], "normals_split": [0.6, 0.1, 0.3],
       "dedup": {"phash_max": 8, "simhash_max": 1}, "gates": {"min_test_critical": 1, "min_val_critical": 0}}


def _row(uid, event, tier="Critical"):
    return {"uid": uid, "event": event, "tier": tier}


def test_copy_in_test_is_dropped_train_untouched():
    labels = [_row("a", "e_tr"), _row("b", "e_te"), _row("c", "e_te"), _row("d", "e_va")]
    keys = {"a": (0, 0), "b": (0b1, 0xFFFF_FFFF), "c": (0xFFFF_0000_FFFF, 0xFFF << 20), "d": (0xF0F0, 0b111)}
    splits, report, kept = freeze(labels, keys, CFG)
    assert "a" in splits and splits["a"] == "train"
    assert "b" not in splits            # SimHash distance 1 to train -> copy
    assert "d" not in splits            # pHash distance 3 to train -> copy
    assert splits["c"] == "test"        # far on both keys -> kept
    assert report["n_dropped"] == 2 and report["gates"]["test_critical"] == 1


def test_rows_without_keys_are_kept_and_reported():
    labels = [_row("a", "e_tr"), _row("z", "e_te")]
    splits, report, _ = freeze(labels, {"a": (0, 0)}, CFG)
    assert splits["z"] == "test" and report["no_keys"] == ["z"]


def test_label_by_rule_d14():
    from parameters.edge.dup_pairs import label_by_rule
    out = label_by_rule([{"d_sem": "1", "d_perc": "30"}, {"d_sem": "9", "d_perc": "8"}, {"d_sem": "2", "d_perc": "9"}], 1, 8)
    assert [p["label"] for p in out] == ["duplicate", "duplicate", "distinct"]
