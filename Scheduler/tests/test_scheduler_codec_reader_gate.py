from copy import deepcopy

import pytest

from Evaluation.scheduler_codec_reader_gate import assess


def receipt(prefix):
    rows = []
    for i in range(6):
        positive = i % 2 == 0
        score = 0.2 if positive else 0.01
        rows.append(dict(source_sha256=f"{prefix}-{i}", observation=dict(
            identity=f"{prefix}-{i}", reference_positive=positive),
            reader_score=score, source_reader_score=score, resized_reader_score=score))
    return dict(codec="sqgan", sqgan_checkpoint_sha256="a", reader_checkpoint_sha256="b",
                preprocessing="test", threshold=0.5, rows=rows)


def test_reader_gate_uses_only_calibration_negatives_for_threshold():
    calibration, extension = receipt("a"), receipt("b")
    result = assess(calibration, extension)
    assert result["candidate_threshold"] > 0.01
    assert result["candidate"]["true_positive"] == 3
    assert result["candidate"]["false_positive"] == 0
    assert result["frozen"]["reader_score"]["true_positive"] == 0
    assert not result["candidate_passes_pilot_gate"]
    extension["rows"][1]["reader_score"] = 0.3
    assert assess(calibration, extension)["candidate_threshold"] == result["candidate_threshold"]


def test_reader_gate_rejects_overlap_and_changed_codec():
    calibration, extension = receipt("a"), receipt("b")
    changed = deepcopy(extension)
    changed["reader_checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="configurations differ"):
        assess(calibration, changed)
    extension["rows"][0]["source_sha256"] = calibration["rows"][0]["source_sha256"]
    with pytest.raises(ValueError, match="shared source"):
        assess(calibration, extension)
