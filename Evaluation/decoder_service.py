"""Single-owner nonpreemptive decoder lifecycle. Execution is caller-injected."""
from collections import deque
from dataclasses import dataclass
import math

from Evaluation.decoder_queue import DecoderQueue, ReceivedMessage
from Scheduler.contextual_decoder_scheduler import DecisionContext


@dataclass(frozen=True)
class DispatchedJob:
    decision: object
    message: ReceivedMessage
    started_at: float
    header_snapshot: object


@dataclass(frozen=True)
class CompletionRecord:
    message_id: str
    outcome: str
    started_at: float
    finished_at: float
    service_seconds: float


class DecoderService:
    def __init__(self, policy, profile=None, registry=None, *, queue=None, context_builder=None):
        self.policy = policy
        self.queue = queue if queue is not None else DecoderQueue(policy, profile, registry)
        self.context_builder = context_builder
        self.in_flight = None
        self._history = deque(maxlen=32)
        self._records = []
        self._clock = 0.
        self._admitted = 0
        self._attempts = 0

    @property
    def history(self):
        return tuple(self._history)

    @property
    def completions(self):
        return tuple(self._records)

    def __len__(self):
        return len(self.queue)

    def _check_time(self, now):
        if isinstance(now, bool) or not math.isfinite(now) or now < self._clock:
            raise ValueError('service clock must be finite and monotonic')

    def enqueue(self, message):
        self.queue.enqueue(message)
        self._admitted += 1

    def dispatch(self, now, trusted_utc=None):
        self._check_time(now)
        self._clock = now
        if self.in_flight is not None:
            return None
        candidates = self.queue.ready_snapshot(now, trusted_utc)
        context = (self.context_builder(candidates, self.history, self._attempts, self._admitted, now)
                   if self.context_builder is not None else
                   DecisionContext(candidates, self.history, self._attempts,
                                   self._admitted, self.queue.profile.profile_id))
        decision = self.policy.select(context)
        if decision is None:
            if candidates:
                raise RuntimeError('work-conserving policy returned no selection with ready jobs')
            return None
        snapshot = next((c for c in candidates if c.message_id == decision.message_id), None)
        if snapshot is None:
            raise RuntimeError('policy selected a job outside the ready snapshot')
        message = self.queue.take(decision.message_id, candidates)
        self.in_flight = DispatchedJob(decision, message, now, snapshot)
        return self.in_flight

    def complete(self, message_id, outcome, finish_time):
        self._check_time(finish_time)
        if outcome not in ('succeeded', 'failed'):
            raise ValueError('completion outcome must be succeeded or failed')
        job = self.in_flight
        if job is None or message_id != job.message.message_id:
            raise ValueError('completion must match the one in-flight source/session/message')
        record = CompletionRecord(message_id, outcome, job.started_at, finish_time,
                                  finish_time - job.started_at)
        if outcome == 'succeeded':
            self._history.append(job.header_snapshot)
        self._attempts += 1
        self._records.append(record)
        self._clock = finish_time
        self.in_flight = None
        return record

    def reset(self):
        if self.in_flight is not None:
            raise RuntimeError('cannot reset a decoder while a job is running')
        self.queue.reset()
        self._history.clear()
        self._records.clear()
        self._admitted = 0
        self._attempts = 0
        self._clock = 0.
