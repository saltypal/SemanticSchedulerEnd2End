import struct
import zlib

import pytest

from parameters.spi_scene_packet import (
    CRC,
    OVERHEAD_BYTES,
    PREFIX,
    PREFIX_BYTES,
    PROFILES,
    VERSION,
    pack,
    register_profile,
    unpack,
)

PAYLOAD = b"sq01-fixture-payload"


def test_round_trip_preserves_identity_payload_spi_and_scene():
    spi = [10, 20, 30, 195]
    scene = [0, 1, 127, 128, 200, 254, 255, 42]
    wire = pack(1, 0x01020304, 0x0A0B0C0D, 0x00000063, PAYLOAD,
                spi_probabilities=spi, scene_values=scene)
    fields = unpack(wire)
    assert fields["version"] == VERSION
    assert fields["profile_id"] == 1
    assert fields["source"] == 0x01020304
    assert fields["session"] == 0x0A0B0C0D
    assert fields["sequence"] == 0x00000063
    assert fields["payload"] == PAYLOAD
    assert fields["scene_values"] == scene
    assert fields["spi_probabilities"] == pytest.approx(
        [value / sum(spi) for value in spi])
    assert sum(fields["spi_probabilities"]) == pytest.approx(1.0)


def test_overhead_is_36_bytes_and_wire_is_prefix_payload_crc():
    wire = pack(1, 1, 2, 3, PAYLOAD)
    assert OVERHEAD_BYTES == 36 == PREFIX_BYTES + CRC.size
    assert len(wire) == PREFIX_BYTES + len(PAYLOAD) + CRC.size
    assert PREFIX.unpack_from(wire)[6] == len(PAYLOAD)  # payload_length field


def test_network_byte_order_for_multi_byte_fields():
    wire = pack(1, 0x01020304, 0x05060708, 0x090A0B0C, PAYLOAD)
    assert wire[0] == VERSION
    assert wire[1:3] == b"\x00\x01"          # profile, big-endian
    assert wire[4:8] == b"\x01\x02\x03\x04"  # source, big-endian
    assert wire[8:12] == b"\x05\x06\x07\x08"
    assert wire[12:16] == b"\x09\x0a\x0b\x0c"
    assert wire[16:20] == struct.pack(">I", len(PAYLOAD))
    assert wire[-CRC.size:] == CRC.pack(zlib.crc32(wire[:-CRC.size]))


def test_missing_scene_and_available_zero_scene_values_differ():
    missing = unpack(pack(1, 1, 1, 1, PAYLOAD))
    zeros = unpack(pack(1, 1, 1, 1, PAYLOAD, scene_values=[0] * 8))
    assert missing["scene_values"] is None
    assert zeros["scene_values"] == [0] * 8
    assert missing["spi_probabilities"] is None
    # The distinction lives in the flag byte, not in the scene bytes themselves.
    assert pack(1, 1, 1, 1, PAYLOAD)[3] == 0x00
    assert pack(1, 1, 1, 1, PAYLOAD, scene_values=[0] * 8)[3] == 0x02


def test_available_spi_probabilities_are_normalized_on_receive():
    fields = unpack(pack(1, 1, 1, 1, PAYLOAD, spi_probabilities=[1, 1, 1, 1]))
    assert fields["spi_probabilities"] == pytest.approx([0.25] * 4)


def test_truncated_and_non_bytes_input_is_rejected():
    wire = pack(1, 1, 1, 1, PAYLOAD, scene_values=[1] * 8)
    with pytest.raises(ValueError, match="truncated"):
        unpack(wire[:PREFIX_BYTES + 2])
    with pytest.raises(ValueError, match="truncated"):
        unpack(b"")
    with pytest.raises(ValueError, match="bytes"):
        unpack(None)


def test_corruption_is_rejected_by_length_or_crc():
    wire = pack(1, 1, 1, 1, PAYLOAD, spi_probabilities=[9, 9, 9, 228],
                scene_values=[3] * 8)
    flipped = bytearray(wire)
    flipped[PREFIX_BYTES] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        unpack(bytes(flipped))
    with pytest.raises(ValueError, match="length"):
        unpack(wire[:-1])
    with pytest.raises(ValueError, match="length"):
        unpack(wire + b"\x00")
    with pytest.raises(ValueError, match="CRC"):
        unpack(wire[:-CRC.size] + b"\x00\x00\x00\x00")


def test_invalid_version_profile_and_flags_are_rejected():
    wire = pack(1, 1, 1, 1, PAYLOAD, scene_values=[1] * 8)
    wrong_version = bytes([2]) + wire[1:]
    wrong_version = wrong_version[:-CRC.size] + CRC.pack(
        zlib.crc32(wrong_version[:-CRC.size]))
    with pytest.raises(ValueError, match="version"):
        unpack(wrong_version)
    bad_profile = PREFIX.pack(VERSION, 999, 2, 1, 1, 1, len(PAYLOAD), *([0] * 12))
    body = bad_profile + PAYLOAD
    with pytest.raises(ValueError, match="profile"):
        unpack(body + CRC.pack(zlib.crc32(body)))
    reserved = bytearray(wire)
    reserved[3] |= 0x80
    reserved = bytes(reserved)
    reserved = reserved[:-CRC.size] + CRC.pack(zlib.crc32(reserved[:-CRC.size]))
    with pytest.raises(ValueError, match="reserved|flags"):
        unpack(reserved)


def test_flag_and_byte_inconsistencies_are_rejected():
    # Scene bytes without the scene flag (same wire shape as an unavailable scene).
    prefix = PREFIX.pack(VERSION, 1, 0x00, 1, 1, 1, len(PAYLOAD), *([0] * 4),
                         *([5] * 8))
    body = prefix + PAYLOAD
    with pytest.raises(ValueError, match="scene values present"):
        unpack(body + CRC.pack(zlib.crc32(body)))
    # Availability flag with all-zero SPI probabilities cannot be normalized.
    with pytest.raises(ValueError, match="sum above zero"):
        pack(1, 1, 1, 1, PAYLOAD, spi_probabilities=[0, 0, 0, 0])
    prefix = PREFIX.pack(VERSION, 1, 0x01, 1, 1, 1, len(PAYLOAD), *([0] * 4),
                         *([0] * 8))
    body = prefix + PAYLOAD
    with pytest.raises(ValueError, match="all-zero"):
        unpack(body + CRC.pack(zlib.crc32(body)))
    # SPI bytes present while the SPI flag is clear.
    prefix = PREFIX.pack(VERSION, 1, 0x00, 1, 1, 1, len(PAYLOAD), *([1, 2, 3, 4]),
                         *([0] * 8))
    body = prefix + PAYLOAD
    with pytest.raises(ValueError, match="SPI probabilities present"):
        unpack(body + CRC.pack(zlib.crc32(body)))


def test_pack_rejects_empty_payload_and_out_of_range_values():
    with pytest.raises(ValueError, match="payload"):
        pack(1, 1, 1, 1, b"")
    with pytest.raises(ValueError, match="source"):
        pack(1, 1 << 32, 1, 1, PAYLOAD)
    with pytest.raises(ValueError, match="scene_values"):
        pack(1, 1, 1, 1, PAYLOAD, scene_values=[0] * 7)
    with pytest.raises(ValueError, match="scene_values\\[3\\]"):
        pack(1, 1, 1, 1, PAYLOAD, scene_values=[0, 0, 0, 256, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="unknown producer profile"):
        pack(7, 1, 1, 1, PAYLOAD)


def test_profile_registry_binds_all_five_fields():
    profile = PROFILES[1]
    assert set(profile) == {"name", "spi_producer", "checkpoint_sha256",
                            "preprocessing", "projection", "quantization"}
    assert profile["checkpoint_sha256"] == (PROFILES[1]["checkpoint_sha256"])
    with pytest.raises(ValueError, match="missing bindings"):
        register_profile(42, {"name": "incomplete"})
    with pytest.raises(ValueError, match="already registered"):
        register_profile(1, {"name": "dup", "spi_producer": "p",
                             "checkpoint_sha256": "c", "preprocessing": "x",
                             "projection": "y", "quantization": "z"})
