"""Versioned 36-byte SPI/scene application header for Trial 2.

Wire layout, network byte order (big-endian):

    offset  size  field
    0       1     version
    1       2     producer profile
    3       1     flags (bit0 SPI available, bit1 scene available, rest 0)
    4       4     source id
    8       4     session id
    12      4     sequence id
    16      4     payload length
    20      4     four quantized SPI probabilities (one byte each)
    24      8     eight quantized scene values (one byte each)
    32      L     codec payload
    32+L    4     CRC32 (zlib) over prefix + payload

Application overhead is PREFIX + CRC = 36 bytes regardless of payload size.
Source/session/sequence identify the message for receiver bookkeeping only;
they never enter any score. Available zero scene values and missing scene
evidence are distinguished by the scene-availability flag, not by the bytes.
"""
import struct
import zlib

PREFIX = struct.Struct(">BHBIIII4B8B")
CRC = struct.Struct(">I")

VERSION = 1
PREFIX_BYTES = PREFIX.size          # 32
CRC_BYTES = CRC.size                # 4
OVERHEAD_BYTES = PREFIX_BYTES + CRC_BYTES  # 36
SPI_SLOTS = 4
SCENE_VALUES = 8

SPI_AVAILABLE = 1 << 0
SCENE_AVAILABLE = 1 << 1
RESERVED_FLAGS = 0xFF & ~(SPI_AVAILABLE | SCENE_AVAILABLE)

# The verified teammate bundle checkpoint from the Test 1 preflight.
PINNED_CHECKPOINT_SHA256 = "706396402d86e84276393ce3fe39f0b0b1b545ff024c0ecae3b8ab72d22f3d6d"

PROFILE_FIELDS = ("spi_producer", "checkpoint_sha256", "preprocessing",
                  "projection", "quantization")

# Every profile binds one SPI producer, codec checkpoint, preprocessing,
# scene projection and quantizer. Unknown profiles are rejected on both
# encode and decode.
PROFILES = {
    1: {
        "name": "trial2_pinned_sqgan_spi_scene_v1",
        "spi_producer": "trial2_reference_tier_producer_v1",
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "preprocessing": "RGB ImageNet normalization; paired 20-class trainIds; 256x512",
        "projection": "mean_max_pool8_projection_v0",
        "quantization": "uint8_exact",
    },
}


def register_profile(profile_id, fields):
    """Bind a producer profile; both the id and all five binding fields are required."""
    if not isinstance(profile_id, int) or not 0 <= profile_id <= 0xFFFF:
        raise ValueError("producer profile id must fit uint16")
    if profile_id in PROFILES:
        raise ValueError("producer profile already registered: %d" % profile_id)
    missing = [key for key in ("name",) + PROFILE_FIELDS if not fields.get(key)]
    if missing:
        raise ValueError("producer profile missing bindings: %s" % ", ".join(missing))
    entry = {key: fields[key] for key in ("name",) + PROFILE_FIELDS}
    PROFILES[profile_id] = entry
    return entry


def _require_range(name, value, high):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= high:
        raise ValueError("%s must be an integer in [0, %d]" % (name, high))
    return value


def _bytes_field(name, values, count):
    if values is None:
        return b"\x00" * count, 0
    if len(values) != count:
        raise ValueError("%s requires exactly %d values" % (name, count))
    out = bytes(_require_range("%s[%d]" % (name, i), value, 255)
                for i, value in enumerate(values))
    return out, (SPI_AVAILABLE if name == "spi_probabilities" else SCENE_AVAILABLE)


def pack(profile_id, source, session, sequence, payload,
         spi_probabilities=None, scene_values=None):
    """Build one complete framed message; overhead is exactly 36 bytes."""
    if profile_id not in PROFILES:
        raise ValueError("unknown producer profile: %r" % (profile_id,))
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise ValueError("non-empty codec payload bytes required")
    payload = bytes(payload)
    spi_bytes, spi_flag = _bytes_field("spi_probabilities", spi_probabilities, SPI_SLOTS)
    if spi_flag and not sum(spi_bytes):
        raise ValueError("available SPI probabilities must sum above zero")
    scene_bytes, scene_flag = _bytes_field("scene_values", scene_values, SCENE_VALUES)
    prefix = PREFIX.pack(
        VERSION, profile_id, spi_flag | scene_flag,
        _require_range("source", source, 0xFFFFFFFF),
        _require_range("session", session, 0xFFFFFFFF),
        _require_range("sequence", sequence, 0xFFFFFFFF),
        len(payload), *spi_bytes, *scene_bytes)
    body = prefix + payload
    return body + CRC.pack(zlib.crc32(body))


def unpack(packet):
    """Validate and return one framed message; any defect raises ValueError."""
    if not isinstance(packet, (bytes, bytearray, memoryview)):
        raise ValueError("packet must be bytes")
    packet = bytes(packet)
    if len(packet) < PREFIX_BYTES + CRC_BYTES:
        raise ValueError("truncated SPI/scene packet")
    version, profile_id, flags, source, session, sequence, length, *tail = \
        PREFIX.unpack_from(packet)
    if version != VERSION:
        raise ValueError("unsupported SPI/scene packet version: %d" % version)
    if profile_id not in PROFILES:
        raise ValueError("unknown producer profile: %d" % profile_id)
    if flags & RESERVED_FLAGS:
        raise ValueError("reserved SPI/scene flags are set: 0x%02x" % flags)
    if length == 0:
        raise ValueError("zero payload length")
    if len(packet) != PREFIX_BYTES + length + CRC_BYTES:
        raise ValueError("payload length does not match packet size")
    if CRC.unpack_from(packet, len(packet) - CRC_BYTES)[0] != zlib.crc32(packet[:-CRC_BYTES]):
        raise ValueError("SPI/scene packet CRC mismatch")
    spi_available = bool(flags & SPI_AVAILABLE)
    scene_available = bool(flags & SCENE_AVAILABLE)
    spi_bytes = tuple(tail[:SPI_SLOTS])
    scene_bytes = tuple(tail[SPI_SLOTS:SPI_SLOTS + SCENE_VALUES])
    if spi_available:
        total = sum(spi_bytes)
        if total == 0:
            raise ValueError("SPI availability flag set with all-zero probabilities")
        spi_probabilities = [value / total for value in spi_bytes]
    else:
        if any(spi_bytes):
            raise ValueError("SPI probabilities present without availability flag")
        spi_probabilities = None
    if scene_available:
        scene_values = list(scene_bytes)
    else:
        if any(scene_bytes):
            raise ValueError("scene values present without availability flag")
        scene_values = None
    return {
        "version": version,
        "profile_id": profile_id,
        "profile": dict(PROFILES[profile_id]),
        "source": source,
        "session": session,
        "sequence": sequence,
        "spi_probabilities": spi_probabilities,
        "scene_values": scene_values,
        "payload": packet[PREFIX_BYTES:-CRC_BYTES],
        "overhead_bytes": OVERHEAD_BYTES,
    }
