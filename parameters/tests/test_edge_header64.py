import numpy as np
import pytest

from Evaluation.edge.contracts import Header
from parameters.edge import header64 as H


def rand_header(rng):
    return Header(capture_time=float(rng.integers(0, 10 ** 6)) / 1000, arrival_time=0.0,
                  payload_bytes=int(rng.integers(0, 2 ** 20)), mission=int(rng.integers(0, 65536)),
                  codec_id=int(rng.integers(0, 16)), m_index=int(rng.integers(0, 16)),
                  tier_probs=tuple(rng.random(4)), relevance=float(rng.random()),
                  sem_key=int(rng.integers(0, 2 ** 63)), perc_key=int(rng.integers(0, 2 ** 63)),
                  sts=tuple(rng.random(13)))


def test_round_trip():
    rng = np.random.default_rng(0)
    tol = 1 / 510 + 1e-9
    for i in range(50):
        h = rand_header(rng)
        b = H.pack(h, i)
        assert len(b) == 64
        g, mid = H.unpack(b)
        assert mid == i and g.sem_key == h.sem_key and g.perc_key == h.perc_key
        assert g.payload_bytes == h.payload_bytes and g.mission == h.mission
        assert (g.codec_id, g.m_index) == (h.codec_id, h.m_index)
        assert np.allclose(g.tier_probs, h.tier_probs, atol=tol)
        assert np.allclose(g.sts, h.sts, atol=tol)
        assert abs(g.relevance - h.relevance) <= tol


def test_crc_and_reserved_and_length():
    b = bytearray(H.pack(rand_header(np.random.default_rng(2)), 7))
    bad = bytearray(b)
    bad[20] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        H.unpack(bytes(bad))
    bad = bytearray(b)
    bad[60] = 1
    with pytest.raises(ValueError, match="reserved"):
        H.unpack(bytes(bad))
    with pytest.raises(ValueError, match="64"):
        H.unpack(bytes(b[:63]))


def test_crc_known_vector():
    assert H.crc16_ccitt(b"123456789") == 0x29B1


def test_flip_rate_monotone():
    rng = np.random.default_rng(3)
    hs = [rand_header(rng) for _ in range(300)]
    rate = H.flip_rate_study(hs, [2, 4, 8], lambda h: int(np.argmax(h.tier_probs)))
    assert rate[8] <= rate[4] <= rate[2] + 0.05 and rate[8] < 0.02


def test_byte_offsets():
    h = Header(capture_time=1.5, arrival_time=0.0, payload_bytes=0x010203, mission=0xABCD, codec_id=3,
               m_index=5, tier_probs=(1.0, 0.0, 0.0, 1.0), relevance=1.0, sem_key=0x1122334455667788,
               perc_key=0x99AABBCCDDEEFF00, sts=tuple([1.0] * 13))
    b = H.pack(h, 0xDEADBEEF, version=2)
    assert b[0] == 0x2F and b[1:5] == bytes.fromhex("deadbeef") and b[5:9] == (1500).to_bytes(4, "big")
    assert b[9:12] == bytes.fromhex("010203") and b[12] == 0x35 and b[13:15] == bytes.fromhex("abcd")
    assert b[15:19] == bytes([255, 0, 0, 255]) and b[19] == 255
    assert b[20:28] == bytes.fromhex("1122334455667788") and b[28:41] == bytes([255] * 13)
    assert b[41:49] == bytes.fromhex("99aabbccddeeff00")
    assert int.from_bytes(b[49:51], "big") == H.crc16_ccitt(b[:49]) and b[51:] == bytes(13)
