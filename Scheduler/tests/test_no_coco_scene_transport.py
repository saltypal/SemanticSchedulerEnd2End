import struct

import pytest

from Evaluation.compact_scene_transport import CompactSceneTransport
from Evaluation.no_coco_scene_transport import NoCocoSceneTransport


def setup_codecs():
    legacy = CompactSceneTransport(37, "12" * 32, "34" * 32, [True] * 85 + [False] * 8)
    return legacy, NoCocoSceneTransport(legacy)


def source_wire(legacy, entries, available=(), flags=1):
    counts, areas = [0] * 93, bytearray(93)
    cm = am = 0
    for i, count, area in entries:
        counts[i], areas[i] = count, area
        if count:
            cm |= 1 << i
        am |= 1 << i
    for i in available:
        am |= 1 << i
        if i < 85:
            cm |= 1 << i
    return (bytes([2, flags]) + legacy.registry + legacy.producer + struct.pack("<93H", *counts)
            + areas + cm.to_bytes(12, "little") + am.to_bytes(12, "little"))


def test_removes_coco_values_and_masks_but_preserves_zeros_and_regions():
    legacy, trial = setup_codecs()
    retained = [(80, 65535, 255), (92, 0, 12)]
    original = source_wire(legacy, [(0, 2, 25), (79, 1, 32)] + retained, [0, 81, 90], flags=3)
    expected = source_wire(legacy, retained, [81, 90], flags=3)
    assert legacy.decode(legacy.encode(original)) == original
    packet = trial.encode(original)
    assert len(packet) == 16
    assert trial.decode(packet) == expected
    with pytest.raises(ValueError):
        legacy.decode(packet)
    with pytest.raises(ValueError):
        trial.decode(legacy.encode(original))


def test_dense_is_bounded_and_failed_scene_round_trips():
    legacy, trial = setup_codecs()
    wire = source_wire(legacy, [(i, int(i < 85), 255) for i in range(80, 93)])
    assert len(trial.encode(wire)) == 30
    assert trial.decode(trial.encode(wire)) == wire
    failed = source_wire(legacy, [], flags=0)
    assert len(trial.encode(failed)) == 8
    assert trial.decode(trial.encode(failed)) == failed


def test_rejects_malformed_scene():
    legacy, trial = setup_codecs()
    packet = trial.encode(source_wire(legacy, [(80, 1, 2)]))
    for bad in (packet[:7], packet[:-1], packet + b"\x00", packet[:2] + b"\x00\x00" + packet[4:],
                packet[:5] + b"\x80" + packet[6:], packet[:8] + b"\x0c" + packet[9:]):
        with pytest.raises(ValueError):
            trial.decode(bad)
