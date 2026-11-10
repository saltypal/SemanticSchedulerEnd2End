from dataclasses import replace

import numpy as np
import pytest

from Evaluation.scheduler_codec_replay import (
    Job, Observation, ReadyHeader, choose, decode_cue, encode_cue, replay, road_retention, trace,
)


def observed(name, **changes):
    return replace(Observation(name, "a" * 64, 2., 10, 3., 1., True, True, 1, False, 0.5),
                   **changes)


def test_road_retention_reads_spatially_indexed_tokens_and_one_byte_quantization():
    road = np.zeros((256, 512), dtype=bool)
    road[0:16, 16:32] = True
    road[16:24, 0:16] = True
    assert road_retention(road, [1]) == pytest.approx(2 / 3)
    assert road_retention(road, [32]) == pytest.approx(1 / 3)
    assert road_retention(road, [0]) == 0
    assert road_retention(np.zeros_like(road), [0]) is None
    assert decode_cue(encode_cue(None)) is None
    assert decode_cue(encode_cue(1)) == 1
    with pytest.raises(ValueError):
        road_retention(road, [512])


def test_encoder_cue_breaks_ties_without_crossing_spi_tiers():
    a = ReadyHeader(1, 0, 1, 0.1)
    b = ReadyHeader(2, 1, 1, 0.9)
    critical = ReadyHeader(3, 2, 0, 0.0)
    assert choose([a, b], "spi").token == 1
    assert choose([a, b], "spi_cue").token == 2
    assert choose([a, b, critical], "spi_cue").token == 3
    assert choose([a, replace(b, cue=None)], "spi_cue").token == 1


def test_sender_serializes_per_source_and_charges_delivery_bytes():
    rows = [observed("a"), observed("b", sender_ms=3.)]
    jobs, _ = trace(rows, 4., sources=1, header_bytes=10, link_bytes_per_ms=10)
    assert (jobs[0].capture, jobs[0].sender_start, jobs[0].arrival) == (0., 0., 4.)
    assert jobs[1].capture == 1.
    assert jobs[1].sender_start == 2.
    assert jobs[1].arrival == 7.


def test_failed_and_unfinished_jobs_cannot_be_credited_and_arrival_is_causal():
    jobs = (
        Job(0, observed("good", reference_critical=True), 0, 0, 1, 5),
        Job(1, observed("failed", predicted_positive=None, reader_ms=0), 0, 0, 2, 9),
        Job(2, observed("late", reference_critical=True, decode_ms=12), 0, 0, 3, 9),
        Job(3, observed("future", spi=0), 0, 0, 100, 105),
    )
    result = replay(jobs, 10, "spi_cue")
    assert result["timely_correct_positive_ids"] == 1
    assert result["critical_missed"] == 1
    assert [row["state"] for row in result["timeline"]] == ["completed", "failed", "unfinished", "not_arrived"]
    assert result["timeline"][0]["decode_end_ms"] == 4
    assert result["timeline"][0]["decision_ms"] == 5


def test_three_policies_replay_same_payloads_and_count_repeat_once():
    jobs, horizon = trace([observed("a"), observed("a"), observed("b")], .5)
    for policy in ("fifo", "spi", "spi_cue"):
        result = replay(jobs, horizon, policy)
        assert result["offered_positive_ids"] == result["timely_correct_positive_ids"] == 2
        assert result["critical_reference_available"] == 3
        assert {r["payload_sha256"] for r in result["timeline"]} == {"a" * 64}

def test_native_pilot_checkpoint_pin_is_full_sha256():
    from Evaluation.scheduler_codec_pilot import PINNED_SHA

    assert len(bytes.fromhex(PINNED_SHA)) == 32


@pytest.mark.parametrize("burst", [False, True])
def test_fixed_deadlines_and_load_use_separate_time_bases(burst):
    rows = [observed("a"), observed("b", decode_ms=13.)]
    jobs, horizon = trace(rows, 1.2, burst=burst, fixed_deadline_ms=100.)
    work = sum(row.decode_ms + row.reader_ms for row in rows)
    assert work / (horizon - 100.) == pytest.approx(1.2)
    assert all(job.deadline - job.capture == pytest.approx(100.) for job in jobs)


def test_detailed_replay_counts_correct_critical_separately_from_completion():
    jobs = (
        Job(0, observed("wrong", reference_critical=True, predicted_positive=False), 0, 0, 1, 5),
        Job(1, observed("late", reference_critical=True), 0, 0, 2, 7),
        Job(2, observed("failed", predicted_positive=None, reader_ms=0), 0, 0, 3, 10),
    )
    report = replay(jobs, 20, "fifo", detailed=True)
    assert report["critical_completed_on_time"] == 1
    assert report["critical_correct_on_time"] == 0
    assert report["late_decisions"] == report["failed"] == 1
