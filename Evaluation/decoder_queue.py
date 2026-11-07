"""Receiver ingress and dispatch for complete, opaque digital decoder jobs.

The transport authenticates sources and establishes completeness. CRC is only
an accidental-corruption check. No image, payload or evaluator data is passed
to a policy. This module, not Scheduler, owns queue mutation.
"""
from dataclasses import dataclass, replace
import json
import math

from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterHeader, get_profile, freshness
from Scheduler.decoder_scheduler import SchedulerCandidate


@dataclass(frozen=True)
class ReceivedMessage:
    source_id: str
    session_id: str
    message_token: int
    header_bytes: bytes
    payload_ref: object
    complete: bool
    arrival_time: float
    sequence: int

    def __post_init__(self):
        if not isinstance(self.source_id,str) or not self.source_id or not isinstance(self.session_id,str) or not self.session_id:
            raise ValueError('source and session identities are required')
        if type(self.message_token) is not int or not 0 <= self.message_token < 2**32:
            raise ValueError('transport token must fit uint32')
        if type(self.complete) is not bool:
            raise ValueError('transport completeness must be boolean')
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0 or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('invalid arrival time or sequence')
        object.__setattr__(self,'header_bytes',bytes(self.header_bytes))

    @property
    def message_id(self):
        return json.dumps([self.source_id,self.session_id,self.message_token],separators=(',',':'))


class DecoderQueue:
    """One task/profile queue; dispatch returns a decision and its opaque job.

    ``now`` and arrival_time share one queue clock. Optional utc_now_ms must
    come from a trusted UTC clock; it alone permits receiver freshness updates.
    Decoder success and actual service time belong to the caller/codec adapter.
    """
    def __init__(self, scheduler, profile, registry=None):
        self.profile = get_profile(profile, registry)
        self.registry = registry
        if scheduler.profile_id not in (None,self.profile.profile_id):
            raise ValueError('scheduler and receiver profiles differ')
        self.scheduler = scheduler
        self._jobs = {}
        self._seen = set()
        self._last_dispatch_time = 0.

    def __len__(self):
        return len(self._jobs)

    def enqueue(self, message):
        if not isinstance(message,ReceivedMessage):
            raise TypeError('ReceivedMessage required')
        if not message.complete:
            raise ValueError('incomplete digital payload cannot enter decoder queue')
        if message.message_id in self._seen:
            raise ValueError('message identity already admitted in this source/session')
        header = ParameterHeader.from_bytes(message.header_bytes, self.registry)
        if header.profile_id != self.profile.profile_id:
            raise ValueError('header task/producer profile differs from queue')
        if header.message_token != message.message_token:
            raise ValueError('header and transport tokens do not match')
        self._jobs[message.message_id] = (message,header)
        self._seen.add(message.message_id)

    def ready_snapshot(self, now, utc_now_ms=None):
        """Immutable eligible headers; payload references remain in this queue."""
        if not math.isfinite(now) or now < self._last_dispatch_time:
            raise ValueError('receiver queue clock must be finite and monotonic')
        if utc_now_ms is not None and (not math.isfinite(utc_now_ms) or utc_now_ms <= 0):
            raise ValueError('trusted UTC time must be finite and positive')
        self._last_dispatch_time = now
        candidates = []
        for message,header in self._jobs.values():
            if message.arrival_time > now:
                continue
            candidate = SchedulerCandidate.from_header(message.message_id,message.source_id,
                message.arrival_time,message.sequence,header)
            if utc_now_ms is not None and header.available[4]:
                value = freshness(header.capture_time_ms,utc_now_ms,self.profile.tau_ms,True)
                values,mask = list(candidate.values),list(candidate.available)
                values[4],mask[4] = (0.,False) if value is None else (value,True)
                candidate = replace(candidate,values=tuple(values),available=tuple(mask))
            candidates.append(candidate)
        return tuple(candidates)

    def take(self, message_id, candidates):
        """Remove one job after checking the exact eligible snapshot."""
        if message_id not in {candidate.message_id for candidate in candidates}:
            raise RuntimeError('policy selected a message outside the eligible snapshot')
        message, _ = self._jobs.pop(message_id)
        return message

    def reset(self):
        self._jobs.clear()
        self._seen.clear()
        self._last_dispatch_time = 0.
        self.scheduler.reset()

    def dispatch(self, now, utc_now_ms=None):
        candidates = self.ready_snapshot(now, utc_now_ms)
        decision = self.scheduler.select(candidates)
        if decision is None:
            return None
        return decision, self.take(decision.message_id, candidates)
