"""Four-level SPI selection on ready candidates; no labels, payloads or queue mutation."""
from enum import IntEnum
import math

from Scheduler.decoder_scheduler import Decision


class SemanticPriority(IntEnum):
    CRITICAL=0
    HIGH=1
    MEDIUM=2
    NORMAL=3


def validate_priority(value):
    if isinstance(value,SemanticPriority):return int(value)
    if type(value) is not int or value not in range(4):
        raise ValueError('SPI must be an ordinal integer 0..3, Critical through Normal')
    return value


def select_spi_candidate(candidates,priorities,*,within_class='fifo',semantic_scores=None):
    """Consume predicted SPI from a future upstream adapter; preserve current ingress.

    IDs associate permitted metadata, never become numerical model inputs.
    Missing SPI invokes a declared whole-snapshot FIFO fallback. Missing secondary
    scores invoke FIFO within the highest class. Zero is an available score.
    The caller supplies a ready-only immutable snapshot; Evaluation owns dispatch.
    """
    if within_class not in ('fifo','semantic'):raise ValueError('unknown SPI tie policy')
    candidates=tuple(candidates)
    if not candidates:return None
    if len({c.message_id for c in candidates})!=len(candidates):raise ValueError('duplicate candidate identity')
    key=lambda c:(c.arrival_time,c.sequence,c.message_id)
    values={}
    for c in candidates:
        if not math.isfinite(c.arrival_time) or c.arrival_time<0:raise ValueError('invalid candidate arrival')
        p=priorities.get(c.message_id)
        values[c.message_id]=None if p is None else validate_priority(p)
    if any(v is None for v in values.values()):
        chosen=min(candidates,key=key)
        return Decision(chosen.message_id,'spi_fifo',0.,'unavailable_spi_snapshot_fifo')
    highest=min(values.values())
    eligible=tuple(c for c in candidates if values[c.message_id]==highest)
    if within_class=='fifo':
        chosen=min(eligible,key=key)
        return Decision(chosen.message_id,'spi_fifo',float(-highest))
    scores=semantic_scores or {}
    permitted={}
    for c in eligible:
        value=scores.get(c.message_id)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
            raise ValueError('secondary semantic scores must be finite numeric values')
        permitted[c.message_id]=value
    if any(v is None for v in permitted.values()):
        chosen=min(eligible,key=key)
        return Decision(chosen.message_id,'spi_semantic',float(-highest),'unavailable_secondary_score_class_fifo')
    chosen=min(eligible,key=lambda c:(-permitted[c.message_id],key(c)))
    return Decision(chosen.message_id,'spi_semantic',float(permitted[chosen.message_id]))
