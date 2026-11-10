import json

import pytest

from Scheduler.training.scene_descriptor import load_reviewed_pairs
from Scheduler.training.scene_vlm_review import MODEL_REVISIONS, REVIEW_KIND


def test_only_stable_vlm_training_consensus_is_loaded(tmp_path):
    images = {
        image_id: {"split": split, "event": split, "flight": split,
                   "near_duplicate": image_id, "spi_tier": 2}
        for split in ("calibration", "train", "development")
        for image_id in (f"{split}-a", f"{split}-b")
    }
    votes = {model: {"forward": "A", "reverse": "B"} for model in MODEL_REVISIONS}
    pairs = [{"a": f"{split}-a", "b": f"{split}-b", "votes": votes}
             for split in ("calibration", "train", "development")]
    manifest = {"review_kind": REVIEW_KIND, "model_revisions": MODEL_REVISIONS,
                "prompt_version": "fixture-v1", "images": images, "pairs": pairs}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_reviewed_pairs(path)
    assert [pair["a"] for pair in loaded["strict_pairs"]] == ["train-a"]
    assert loaded["strict_pairs"][0]["grade_a"] == 1
    assert loaded["calibration"]["joint_agreement"] == 1

    images["development-b"]["near_duplicate"] = "train-b"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="near_duplicate group overlaps"):
        load_reviewed_pairs(path)

    images["development-b"]["near_duplicate"] = "development-b"
    pairs[0]["votes"] = {
        model: {"forward": "A", "reverse": "A"} for model in MODEL_REVISIONS
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration thresholds"):
        load_reviewed_pairs(path)
