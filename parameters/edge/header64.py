"""64-byte semantic header pack/unpack (plan v3.1 sec. 4) and bit-depth flip-rate study (S6).

Big-endian layout (offset: bytes): 0:1 version<<4|flags, 1:4 message id, 5:4 capture ms mod 2^32,
9:3 payload length, 12:1 codec<<4|m, 13:2 mission, 15:4 SPI u8 x4, 19:1 relevance u8, 20:8 sem key,
28:13 STS u8, 41:8 perceptual key, 49:2 CRC-16/CCITT-FALSE over bytes 0..48, 51:13 reserved zero.
Flags (low nibble): bit0 SPI, bit1 STS, bit2 sem key, bit3 p-hash valid. Field values come from the
``Evaluation.edge.contracts.Header``; arrival_time is not transmitted (0.0 on unpack).
"""
import struct

import numpy as np

from Evaluation.edge.contracts import Header

SIZE = 64
CRC_END = 49
F_SPI, F_STS, F_SEM, F_PH = 1, 2, 4, 8


def quantise(x, bits=8):
    """Map [0,1] to integer levels 0..2^bits-1 (clipped, rounded)."""
    top = (1 << bits) - 1
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) * top), 0, top).astype(np.int64)


def dequantise(q, bits=8):
    return np.asarray(q, dtype=np.float64) / ((1 << bits) - 1)


def crc16_ccitt(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _check(name, value, lo, hi):
    if not lo <= value <= hi:
        raise ValueError("%s=%r outside [%d, %d]" % (name, value, lo, hi))


def pack(header, message_id, version=1):
    """Encode a contract Header into exactly 64 bytes."""
    h = header
    if len(h.tier_probs) != 4:
        raise ValueError("tier_probs must have 4 values")
    if len(h.sts) not in (0, 13):
        raise ValueError("sts must have 13 values (or be empty)")
    _check("version", version, 0, 15)
    _check("message_id", message_id, 0, 2 ** 32 - 1)
    _check("payload_bytes", h.payload_bytes, 0, 2 ** 24 - 1)
    _check("codec_id", h.codec_id, 0, 15)
    _check("m_index", h.m_index, 0, 15)
    _check("mission", h.mission, 0, 65535)
    _check("sem_key", h.sem_key, 0, 2 ** 64 - 1)
    _check("perc_key", h.perc_key, 0, 2 ** 64 - 1)
    flags = F_SPI | F_SEM | F_PH | (F_STS if h.sts else 0)
    buf = bytearray(SIZE)
    buf[0] = (version << 4) | flags
    struct.pack_into(">I", buf, 1, message_id)
    struct.pack_into(">I", buf, 5, int(round(h.capture_time * 1000)) % 2 ** 32)
    buf[9:12] = int(h.payload_bytes).to_bytes(3, "big")
    buf[12] = (h.codec_id << 4) | h.m_index
    struct.pack_into(">H", buf, 13, h.mission)
    buf[15:19] = bytes(quantise(h.tier_probs).tolist())
    buf[19] = int(quantise(h.relevance))
    struct.pack_into(">Q", buf, 20, h.sem_key)
    if h.sts:
        buf[28:41] = bytes(quantise(h.sts).tolist())
    struct.pack_into(">Q", buf, 41, h.perc_key)
    struct.pack_into(">H", buf, CRC_END, crc16_ccitt(bytes(buf[:CRC_END])))
    return bytes(buf)


def unpack(data):
    """Decode 64 bytes into ``(Header, message_id)``; raises ValueError on any inconsistency."""
    if len(data) != SIZE:
        raise ValueError("header must be %d bytes, got %d" % (SIZE, len(data)))
    if struct.unpack_from(">H", data, CRC_END)[0] != crc16_ccitt(bytes(data[:CRC_END])):
        raise ValueError("header CRC mismatch")
    if any(data[51:]):
        raise ValueError("reserved bytes must be zero")
    flags = data[0] & 0x0F
    need = F_SPI | F_SEM | F_PH
    if flags & need != need:
        raise ValueError("SPI, semantic-key and p-hash flags must be set")
    sts = tuple(dequantise(list(data[28:41])).tolist()) if flags & F_STS else ()
    header = Header(
        capture_time=struct.unpack_from(">I", data, 5)[0] / 1000.0, arrival_time=0.0,
        payload_bytes=int.from_bytes(data[9:12], "big"), mission=struct.unpack_from(">H", data, 13)[0],
        codec_id=data[12] >> 4, m_index=data[12] & 15,
        tier_probs=tuple(dequantise(list(data[15:19])).tolist()), relevance=float(data[19]) / 255.0,
        sem_key=struct.unpack_from(">Q", data, 20)[0], perc_key=struct.unpack_from(">Q", data, 41)[0],
        sts=sts)
    return header, struct.unpack_from(">I", data, 1)[0]


def requantised(header, bits):
    """Copy of ``header`` with SPI / relevance / STS snapped to ``bits`` levels."""
    def snap(v):
        return tuple(dequantise(quantise(v, bits), bits).tolist())
    return Header(header.capture_time, header.arrival_time, header.payload_bytes, header.mission,
                  header.codec_id, header.m_index, snap(header.tier_probs),
                  float(snap([header.relevance])[0]), header.sem_key, header.perc_key,
                  snap(header.sts) if header.sts else ())


def flip_rate_study(headers, bits_list, decision_fn):
    """Fraction of headers whose ``decision_fn(header)`` changes after re-quantising to each depth.

    Returns ``{bits: flip_rate}`` versus the full-precision header.
    """
    headers = list(headers)
    if not headers:
        raise ValueError("no headers")
    base = [decision_fn(h) for h in headers]
    return {int(b): sum(decision_fn(requantised(h, b)) != d for h, d in zip(headers, base)) / len(headers)
            for b in bits_list}
