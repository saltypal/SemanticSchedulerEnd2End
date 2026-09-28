import pytest

from SemParam.semantic_header import (
    SCHEDULING_HEADER_BYTES,
    SemanticSchedulingHeaderV2,
    SemanticSchedulingHeaderV2Codec,
)


def test_v2_header_is_fixed_24_bytes_and_round_trips():
    header = SemanticSchedulingHeaderV2(
        context_relevance=-0.25,
        delivered_history_novelty=0.75,
        task_impact=3,
        urgency=2,
        evidence_sufficiency=1,
        history_version=17,
        history_count=9,
    )
    codec = SemanticSchedulingHeaderV2Codec()
    payload = codec.encode(header)
    assert len(payload) == SCHEDULING_HEADER_BYTES == 24
    decoded = codec.decode(payload)
    assert decoded.context_relevance == pytest.approx(-0.25, abs=1 / 32767)
    assert decoded.delivered_history_novelty == pytest.approx(0.75, abs=1 / 65535)
    assert decoded.task_impact == 3
    assert decoded.history_version == 17


def test_v2_header_preserves_unavailable_values_and_rejects_crc_corruption():
    codec = SemanticSchedulingHeaderV2Codec()
    payload = codec.encode(
        SemanticSchedulingHeaderV2(None, None, 1, None, 0, history_version=0, history_count=0)
    )
    decoded = codec.decode(payload)
    assert decoded.context_relevance is None
    assert decoded.delivered_history_novelty is None
    assert decoded.urgency is None
    corrupt = payload[:-1] + bytes([payload[-1] ^ 0x01])
    with pytest.raises(ValueError, match="CRC32"):
        codec.decode(corrupt)
