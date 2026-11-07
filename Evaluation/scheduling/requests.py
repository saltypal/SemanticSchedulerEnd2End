"""Build logical control-channel sidecars from a decision-time feature view."""

from __future__ import annotations

from Evaluation.scheduling.semantic_contracts import LatentJobSpec
from Archived.owners_before_simplification_20261027.SemParam.headers.semantic import (
    SemanticSchedulingHeaderV2, SemanticSchedulingHeaderV2Codec,
)


_CODEC = SemanticSchedulingHeaderV2Codec()


def encode_scheduling_request(job: LatentJobSpec, *,
                              history_version: int, delivered_count: int) -> bytes:
    """Encode exactly the fields visible for this decision, including P3."""
    values = job.features
    def present(name: str):
        value = getattr(values, name)
        return value.value if value.available else None

    impact = present("task_impact")
    urgency = present("urgency")
    sufficiency = present("evidence_sufficiency")
    header = SemanticSchedulingHeaderV2(
        context_relevance=present("context_relevance"),
        delivered_history_novelty=present("delivered_history_novelty"),
        task_impact=None if impact is None else int(impact),
        urgency=None if urgency is None else int(urgency),
        evidence_sufficiency=None if sufficiency is None else int(sufficiency),
        history_version=history_version,
        history_count=delivered_count,
    )
    return _CODEC.encode(header)
