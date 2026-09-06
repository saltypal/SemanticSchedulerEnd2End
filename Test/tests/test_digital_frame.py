import numpy as np

from Baselines.DigitalPHY.bit_utils import frame_payload_bits, recover_payload_bits


def test_digital_frame_records_and_checks_crc() -> None:
    source = np.unpackbits(np.frombuffer(b"semantic communication", dtype=np.uint8), bitorder="big")
    frame = frame_payload_bits(source)
    recovered, success, reason = recover_payload_bits(frame)
    assert success is True
    assert reason == "ok"
    assert np.array_equal(recovered, source)


def test_digital_frame_does_not_hide_corruption() -> None:
    source = np.unpackbits(np.frombuffer(b"semantic communication", dtype=np.uint8), bitorder="big")
    frame = frame_payload_bits(source)
    frame[-1] ^= 1
    recovered, success, reason = recover_payload_bits(frame)
    assert recovered is None
    assert success is False
    assert reason == "crc_mismatch"
