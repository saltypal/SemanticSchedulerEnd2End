"""Compact, versioned semantic-header codec independent of Stage 1A VQ framing."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Iterable

from INFRA.Artifacts import ParameterDefinition, ParameterProducerLevel, SemanticParameterBundle
from SemParam.catalog import resolve_catalog


HEADER_MAGIC = b"SHP"
HEADER_VERSION = 1
SCHEDULING_HEADER_VERSION = 2
SCHEDULING_HEADER_SCHEMA_ID = 0x0201
SCHEDULING_HEADER_BYTES = 24

_FIELD_IDS = {
    "AoI_seconds": 1,
    "freshness": 2,
    "novelty": 3,
    "relevance": 4,
    "reliability": 5,
    "urgency": 6,
    "task_importance": 7,
}
_PRODUCER_IDS = {
    ParameterProducerLevel.UE: 0,
    ParameterProducerLevel.ORAN: 1,
    ParameterProducerLevel.EXTERNAL: 2,
}
_PRODUCER_FROM_ID = {identifier: level for level, identifier in _PRODUCER_IDS.items()}


@dataclass(frozen=True, slots=True)
class DecodedSemanticHeaderField:
    name: str
    available: bool
    producer_level: ParameterProducerLevel
    value: float | None
    units: str | None


@dataclass(frozen=True, slots=True)
class DecodedSemanticHeader:
    version: int
    fields: tuple[DecodedSemanticHeaderField, ...]


@dataclass(frozen=True, slots=True)
class SemanticSchedulingHeaderV2:
    """Fixed 24-byte header for Track-A scheduler requests.

    This header is deliberately separate from the V1 variable-length semantic
    bundle above.  V1 remains the compatibility codec for existing notebooks.
    """

    context_relevance: float | None
    delivered_history_novelty: float | None
    task_impact: int | None
    urgency: int | None
    evidence_sufficiency: int | None
    history_version: int = 0
    history_count: int = 0
    schema_id: int = SCHEDULING_HEADER_SCHEMA_ID

    def __post_init__(self) -> None:
        if not 0 <= self.schema_id <= 0xFFFF:
            raise ValueError("schema_id must fit uint16.")
        if not 0 <= self.history_version <= 0xFFFFFFFF:
            raise ValueError("history_version must fit uint32.")
        if not 0 <= self.history_count <= 0xFFFF:
            raise ValueError("history_count must fit uint16.")
        if self.context_relevance is not None and not -1.0 <= self.context_relevance <= 1.0:
            raise ValueError("context_relevance must be in [-1, 1].")
        if self.delivered_history_novelty is not None and not 0.0 <= self.delivered_history_novelty <= 1.0:
            raise ValueError("delivered_history_novelty must be in [0, 1].")
        for name, value in (("task_impact", self.task_impact), ("urgency", self.urgency)):
            if value is not None and value not in range(4):
                raise ValueError(f"{name} must be an integer in [0, 3].")
        if self.evidence_sufficiency is not None and self.evidence_sufficiency not in (0, 1):
            raise ValueError("evidence_sufficiency must be 0, 1, or None.")

    @property
    def presence_mask(self) -> int:
        values = (
            self.context_relevance,
            self.delivered_history_novelty,
            self.task_impact,
            self.urgency,
            self.evidence_sufficiency,
        )
        return sum(1 << index for index, value in enumerate(values) if value is not None)


class SemanticSchedulingHeaderV2Codec:
    """Encode the fixed 24-byte semantic scheduling sidecar with CRC32."""

    _STRUCT = struct.Struct(">3sBHBhHBBBIHI")

    @staticmethod
    def _encode_relevance(value: float | None) -> int:
        return 0 if value is None else int(round(value * 32767.0))

    @staticmethod
    def _encode_novelty(value: float | None) -> int:
        return 0 if value is None else int(round(value * 65535.0))

    @staticmethod
    def _decode_relevance(value: int, present: bool) -> float | None:
        return value / 32767.0 if present else None

    @staticmethod
    def _decode_novelty(value: int, present: bool) -> float | None:
        return value / 65535.0 if present else None

    def encode(self, header: SemanticSchedulingHeaderV2) -> bytes:
        mask = header.presence_mask
        body = self._STRUCT.pack(
            HEADER_MAGIC,
            SCHEDULING_HEADER_VERSION,
            header.schema_id,
            mask,
            self._encode_relevance(header.context_relevance),
            self._encode_novelty(header.delivered_history_novelty),
            0 if header.task_impact is None else header.task_impact,
            0 if header.urgency is None else header.urgency,
            0 if header.evidence_sufficiency is None else header.evidence_sufficiency,
            header.history_version,
            header.history_count,
            0,
        )
        checksum = zlib.crc32(body[:-4]) & 0xFFFFFFFF
        encoded = body[:-4] + struct.pack(">I", checksum)
        if len(encoded) != SCHEDULING_HEADER_BYTES:
            raise AssertionError("Scheduling header layout is not 24 bytes.")
        return encoded

    def decode(self, payload: bytes) -> SemanticSchedulingHeaderV2:
        if len(payload) != SCHEDULING_HEADER_BYTES:
            raise ValueError(f"Scheduling header must be exactly {SCHEDULING_HEADER_BYTES} bytes.")
        (
            magic,
            version,
            schema_id,
            mask,
            relevance,
            novelty,
            impact,
            urgency,
            sufficiency,
            history_version,
            history_count,
            checksum,
        ) = self._STRUCT.unpack(payload)
        if magic != HEADER_MAGIC:
            raise ValueError("Scheduling header magic is absent or corrupt.")
        if version != SCHEDULING_HEADER_VERSION:
            raise ValueError(f"Unsupported scheduling header version: {version}")
        if mask & ~0b1_1111:
            raise ValueError("Scheduling header presence mask has reserved bits set.")
        expected_checksum = zlib.crc32(payload[:-4]) & 0xFFFFFFFF
        if checksum != expected_checksum:
            raise ValueError("Scheduling header CRC32 is invalid.")
        decoded = SemanticSchedulingHeaderV2(
            context_relevance=self._decode_relevance(relevance, bool(mask & 0b00001)),
            delivered_history_novelty=self._decode_novelty(novelty, bool(mask & 0b00010)),
            task_impact=impact if mask & 0b00100 else None,
            urgency=urgency if mask & 0b01000 else None,
            evidence_sufficiency=sufficiency if mask & 0b10000 else None,
            history_version=history_version,
            history_count=history_count,
            schema_id=schema_id,
        )
        return decoded


class SemanticHeaderCodec:
    """TLV codec for a logical sidecar, not an over-air Stage 1A VQ replacement."""

    def __init__(self, definitions: Iterable[ParameterDefinition] | None = None) -> None:
        selected = tuple(definitions or resolve_catalog())
        missing_ids = [definition.name for definition in selected if definition.name not in _FIELD_IDS]
        if missing_ids:
            raise ValueError(f"Semantic header field ids are not registered for: {missing_ids}")
        self.definitions = selected
        self._by_id = {_FIELD_IDS[definition.name]: definition for definition in selected}

    @staticmethod
    def _encode_score(value: float) -> bytes:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Normalized semantic score is outside [0, 1]: {value}")
        return struct.pack(">H", int(round(value * 65535.0)))

    @staticmethod
    def _decode_score(payload: bytes) -> float:
        if len(payload) != 2:
            raise ValueError("Normalized semantic score payload must be exactly two bytes.")
        return struct.unpack(">H", payload)[0] / 65535.0

    @staticmethod
    def _encode_aoi(value: float) -> bytes:
        milliseconds = int(round(value * 1000.0))
        if milliseconds < 0 or milliseconds > 0xFFFFFFFF:
            raise ValueError(f"AoI cannot be encoded as uint32 milliseconds: {value}")
        return struct.pack(">I", milliseconds)

    @staticmethod
    def _decode_aoi(payload: bytes) -> float:
        if len(payload) != 4:
            raise ValueError("AoI payload must be exactly four bytes.")
        return struct.unpack(">I", payload)[0] / 1000.0

    def encode(self, bundle: SemanticParameterBundle) -> bytes:
        records: list[bytes] = []
        for definition in self.definitions:
            parameter = bundle.values.get(definition.name)
            if parameter is None:
                continue
            field_id = _FIELD_IDS[definition.name]
            producer_id = _PRODUCER_IDS[parameter.producer_level]
            flags = (1 if parameter.available else 0) | (producer_id << 1)
            if parameter.available:
                assert parameter.value is not None
                if definition.header_encoding == "aoi_milliseconds":
                    payload = self._encode_aoi(parameter.value)
                elif definition.header_encoding == "normalized_score":
                    payload = self._encode_score(parameter.value)
                else:
                    raise ValueError(f"Unsupported semantic header encoding: {definition.header_encoding}")
            else:
                payload = b""
            records.append(struct.pack(">BBB", field_id, flags, len(payload)) + payload)
        if len(records) > 255:
            raise ValueError("Semantic header supports at most 255 parameter records.")
        return HEADER_MAGIC + bytes((HEADER_VERSION, len(records))) + b"".join(records)

    def decode(self, payload: bytes) -> DecodedSemanticHeader:
        if len(payload) < 5 or payload[:3] != HEADER_MAGIC:
            raise ValueError("Semantic header magic is absent or corrupt.")
        version = payload[3]
        if version != HEADER_VERSION:
            raise ValueError(f"Unsupported semantic header version: {version}")
        field_count = payload[4]
        offset = 5
        decoded: list[DecodedSemanticHeaderField] = []
        for _ in range(field_count):
            if offset + 3 > len(payload):
                raise ValueError("Semantic header ended inside a TLV record.")
            field_id, flags, payload_length = struct.unpack(">BBB", payload[offset : offset + 3])
            offset += 3
            if offset + payload_length > len(payload):
                raise ValueError("Semantic header field length exceeds payload.")
            raw_value = payload[offset : offset + payload_length]
            offset += payload_length
            definition = self._by_id.get(field_id)
            if definition is None:
                raise ValueError(f"Semantic header references an unknown field id: {field_id}")
            available = bool(flags & 1)
            producer_id = (flags >> 1) & 0b11
            producer_level = _PRODUCER_FROM_ID.get(producer_id)
            if producer_level is None:
                raise ValueError(f"Semantic header has an unknown producer level id: {producer_id}")
            if not available:
                if raw_value:
                    raise ValueError("Unavailable semantic header fields must not carry a numeric payload.")
                value = None
            elif definition.header_encoding == "aoi_milliseconds":
                value = self._decode_aoi(raw_value)
            elif definition.header_encoding == "normalized_score":
                value = self._decode_score(raw_value)
            else:
                raise ValueError(f"Unsupported semantic header encoding: {definition.header_encoding}")
            decoded.append(
                DecodedSemanticHeaderField(
                    name=definition.name,
                    available=available,
                    producer_level=producer_level,
                    value=value,
                    units=definition.units,
                )
            )
        if offset != len(payload):
            raise ValueError("Semantic header contains unexpected trailing bytes.")
        return DecodedSemanticHeader(version=version, fields=tuple(decoded))
