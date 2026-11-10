"""Trial packet adapter for the received-feature scheduler."""

import math
import struct
import zlib

import numpy as np

from Evaluation.semantic_learning_env import Message


# Version, job identity, arrival/capture/budget, SPI/R/N, masks, lengths.
CORE = struct.Struct("<BIIII3d6fBHI")
CRC = struct.Struct("<I")


def pack(message, scene, payload, capture, budget):
    scalars = message.features[374:380]
    flags = message.features[380:387]
    if not np.isin(flags, (0, 1)).all() or not payload:
        raise ValueError("invalid scalar availability or empty codec payload")
    mask = sum(int(flag) << i for i, flag in enumerate(flags))
    core = CORE.pack(1, message.token, message.source, message.session, message.sequence,
                     message.arrival, capture, budget, *scalars, mask, len(scene), len(payload))
    body = core + scene + payload
    return body + CRC.pack(zlib.crc32(body))


def unpack(packet, transport, profile_id):
    if len(packet) < CORE.size + CRC.size:
        raise ValueError("truncated scheduler packet")
    fields = CORE.unpack_from(packet)
    version, token, source, session, sequence, arrival, capture, budget = fields[:8]
    mask, scene_size, payload_size = fields[-3:]
    end = CORE.size + scene_size
    if (version != 1 or mask >> 7 or payload_size == 0
            or len(packet) != end + payload_size + CRC.size
            or not all(math.isfinite(x) for x in (arrival, capture, budget))
            or capture < 0 or arrival < capture or budget <= 0):
        raise ValueError("invalid scheduler packet fields or lengths")
    if CRC.unpack_from(packet, len(packet) - CRC.size)[0] != zlib.crc32(packet[:-CRC.size]):
        raise ValueError("scheduler packet CRC mismatch")
    wire = transport.decode(packet[CORE.size:end])
    count = np.frombuffer(wire, dtype="<u2", count=93, offset=66).astype(float)
    area = np.frombuffer(wire, dtype="u1", count=93, offset=252) / 255
    cm, am = (int.from_bytes(wire[i:i+12], "little") for i in (345, 357))
    features = np.array([*np.log1p(count), *area,
                         *((cm >> i) & 1 for i in range(93)), *((am >> i) & 1 for i in range(93)),
                         wire[1] & 1, (wire[1] >> 1) & 1, *fields[8:14],
                         *((mask >> i) & 1 for i in range(7))], dtype=np.float32)
    # Trailer CRC is packet overhead, counted with header rather than payload.
    return (Message(token, source, session, arrival, sequence, features, packet[:end], profile_id),
            packet[end:-CRC.size], capture, budget)
