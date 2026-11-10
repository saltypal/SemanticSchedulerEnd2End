import numpy as np
import pytest

from Evaluation.compact_scene_transport import CompactSceneTransport
from Evaluation.no_coco_scene_transport import NoCocoSceneTransport
from Evaluation.scheduler_trial_no_coco_audit import reconstruct
from Evaluation.semantic_learning_env import Message
from Evaluation.trial_scheduler_packet import CRC, pack, unpack


def test_full_header_preserves_scalars_and_payload_and_checks_corruption():
    legacy = CompactSceneTransport(1, NoCocoSceneTransport.VOCABULARY_SHA256, "34" * 32,
                                   [True] * 85 + [False] * 8)
    reduced = NoCocoSceneTransport(legacy)
    features = np.zeros(387, dtype=np.float32)
    features[[0, 186, 372, 374, 380]] = [np.log(2), 1, 1, 1, 1]
    features[[378, 379, 384, 385]] = [0.25, 0.75, 1, 1]
    message = Message(1, 2, 3, 4., 5, features)
    wire = reconstruct(features, legacy)
    old = pack(message, legacy.encode(wire), b"test body", 1., 2.)
    new = pack(message, reduced.encode(wire), b"test body", 1., 2.)
    a, a_payload, *a_time = unpack(old, legacy, "fixture")
    b, b_payload, *b_time = unpack(new, reduced, "fixture")
    assert a_payload == b_payload == b"test body"
    assert a_time == b_time == [1., 2.]
    assert np.array_equal(a.features[372:], b.features[372:])
    assert b.features[0] == b.features[186] == 0
    assert len(new) == len(b.header_bytes) + len(b_payload) + CRC.size
    for malformed, codec in ((new[:-1], reduced), (new[:-5] + b"X" + new[-4:], reduced),
                             (new, legacy)):
        with pytest.raises(ValueError):
            unpack(malformed, codec, "fixture")
