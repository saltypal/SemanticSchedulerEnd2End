import dataclasses

import numpy as np
import pytest

from Evaluation.edge.contracts import Header
from parameters.edge import header64 as H

TOL16 = 0.5 / 65535 + 1e-12   # half a 16-bit step
TOL4 = 0.5 / 15 + 1e-12       # half a 4-bit step


def rand_header(rng):
    return Header(capture_time=float(rng.integers(0, 10 ** 6)) / 1000, arrival_time=0.0,
                  payload_bytes=int(rng.integers(0, 2 ** 20)), mission=int(rng.integers(0, 65536)),
                  codec_id=int(rng.integers(0, 16)), m_index=int(rng.integers(0, 16)),
                  tier_probs=tuple(rng.random(4)), relevance=float(rng.random()),
                  sem_key=int(rng.integers(0, 2 ** 63)), perc_key=int(rng.integers(0, 2 ** 63)),
                  sts=tuple(rng.random(13)))


def _fix_crc(bad):
    bad[48:50] = H.crc16_ccitt(bytes(bad[:48])).to_bytes(2, "big")
    return bytes(bad)


def test_round_trip_each_field():
    rng = np.random.default_rng(0)
    for i in range(50):
        h = rand_header(rng)
        b = H.pack(h, i)
        assert len(b) == 64
        g, mid = H.unpack(b)
        assert mid == i and g.sem_key == h.sem_key and g.perc_key == h.perc_key
        assert g.payload_bytes == h.payload_bytes and g.mission == h.mission
        assert (g.codec_id, g.m_index) == (h.codec_id, h.m_index)
        assert np.allclose(g.tier_probs, h.tier_probs, atol=TOL16, rtol=0)   # SPI u16 x4
        assert abs(g.relevance - h.relevance) <= TOL16                       # relevance u16
        assert np.allclose(g.sts, h.sts, atol=TOL4, rtol=0)                  # STS 4-bit nibbles


def test_exactly_64_bytes_with_and_without_sts():
    rng = np.random.default_rng(1)
    h = rand_header(rng)
    assert len(H.pack(h, 0)) == 64
    no_sts = dataclasses.replace(h, sts=())
    b = H.pack(no_sts, 5)
    assert len(b) == 64 and b[0] & H.F_STS == 0 and not any(b[33:40])
    g, _ = H.unpack(b)
    assert g.sts == ()


def test_crc_detects_flipped_byte():
    b = H.pack(rand_header(np.random.default_rng(2)), 7)
    for i in range(50):      # covered bytes 0..47 and the CRC bytes 48..49
        bad = bytearray(b)
        bad[i] ^= 0x01
        with pytest.raises(ValueError, match="CRC"):
            H.unpack(bytes(bad))


def test_reserved_and_length():
    b = bytearray(H.pack(rand_header(np.random.default_rng(2)), 7))
    for i in range(50, 64):
        bad = bytearray(b)
        bad[i] = 1
        with pytest.raises(ValueError, match="reserved"):
            H.unpack(bytes(bad))
    with pytest.raises(ValueError, match="64"):
        H.unpack(bytes(b[:63]))


def test_sts_padding_nibble_must_be_zero():
    h = Header(capture_time=0.0, arrival_time=0.0, payload_bytes=1, mission=0, codec_id=0, m_index=0,
               tier_probs=(0.5, 0.5, 0.5, 0.5), relevance=0.5, sem_key=1, perc_key=2,
               sts=tuple([0.5] * 13))
    bad = bytearray(H.pack(h, 0))
    bad[39] |= 0x01                       # the 14th (padding) nibble
    with pytest.raises(ValueError, match="padding"):
        H.unpack(_fix_crc(bad))


def test_crc_known_vector():
    assert H.crc16_ccitt(b"123456789") == 0x29B1


def test_sts_nibble_packing_13_values():
    q = list(range(13))                   # nibbles 0..12
    h = Header(capture_time=0.0, arrival_time=0.0, payload_bytes=0, mission=0, codec_id=0, m_index=0,
               tier_probs=(0.0, 0.0, 0.0, 0.0), relevance=0.0, sem_key=0, perc_key=0,
               sts=tuple(v / 15 for v in q))
    b = H.pack(h, 0)
    assert b[33:40] == bytes([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xC0])   # last nibble, then zero pad
    g, _ = H.unpack(b)
    assert np.allclose(g.sts, [v / 15 for v in q], atol=1e-12, rtol=0)
    assert H.unpack_sts(b[33:40]) == q
    assert H.pack_sts(q) == bytes([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xC0])


def test_version_2_default():
    h = rand_header(np.random.default_rng(4))
    assert H.VERSION == 2
    assert H.pack(h, 0)[0] >> 4 == 2
    assert H.pack(h, 0, version=3)[0] >> 4 == 3


def test_flip_rate_monotone():
    rng = np.random.default_rng(3)
    hs = [rand_header(rng) for _ in range(300)]
    rate = H.flip_rate_study(hs, [2, 4, 8], lambda h: int(np.argmax(h.tier_probs)))
    assert rate[8] <= rate[4] <= rate[2] + 0.05 and rate[8] < 0.02


def test_byte_offsets():
    h = Header(capture_time=1.5, arrival_time=0.0, payload_bytes=0x010203, mission=0xABCD, codec_id=3,
               m_index=5, tier_probs=(1.0, 0.0, 0.0, 1.0), relevance=1.0, sem_key=0x1122334455667788,
               perc_key=0x99AABBCCDDEEFF00, sts=tuple([1.0] * 13))
    b = H.pack(h, 0xDEADBEEF)
    assert b[0] == 0x2F and b[1:5] == bytes.fromhex("deadbeef") and b[5:9] == (1500).to_bytes(4, "big")
    assert b[9:12] == bytes.fromhex("010203") and b[12] == 0x35 and b[13:15] == bytes.fromhex("abcd")
    assert b[15:23] == bytes.fromhex("ffff00000000ffff")          # SPI u16 x4
    assert b[23:25] == bytes.fromhex("ffff")                        # relevance u16
    assert b[25:33] == bytes.fromhex("1122334455667788")
    assert b[33:40] == bytes([255] * 6) + bytes([0xF0])             # 13 x 0xF nibbles, zero pad
    assert b[40:48] == bytes.fromhex("99aabbccddeeff00")
    assert int.from_bytes(b[48:50], "big") == H.crc16_ccitt(b[:48]) and b[50:] == bytes(14)
