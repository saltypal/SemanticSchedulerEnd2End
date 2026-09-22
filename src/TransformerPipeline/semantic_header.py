"""Compact, versioned semantic-header codec independent of Stage 1A VQ framing."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

from INFRA.Artifacts import ParameterDefinition, ParameterProducerLevel, SemanticParameterBundle
from TransformerPipeline.catalog import resolve_catalog


HEADER_MAGIC = b"SHP"
HEADER_VERSION = 1

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
