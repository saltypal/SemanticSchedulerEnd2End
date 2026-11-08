"""Causal replay for measured codec jobs. Times are milliseconds."""

from dataclasses import dataclass
import math
from statistics import median

import numpy as np


POLICIES = ("fifo", "spi", "spi_cue")


def encode_cue(value):
    if value is None:
        return b"\xff"
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("cue must be in [0, 1]")
    return bytes([round(value * 254)])


def decode_cue(wire):
    if len(wire) != 1:
        raise ValueError("cue requires one byte")
    return None if wire[0] == 255 else wire[0] / 254


def road_retention(road_pixels, positions):
    """Fraction of teacher-road pixel mass covered by retained image tokens."""
    road = np.asarray(road_pixels, dtype=bool)
    positions = tuple(positions)
    if (road.shape != (256, 512) or not positions or len(set(positions)) != len(positions)
            or any(not 0 <= index < 512 for index in positions)):
        raise ValueError("invalid SQ01 road mask or retained positions")
    mass = road.reshape(16, 16, 32, 16).sum(axis=(1, 3)).reshape(-1)
    total = mass.sum()
    return float(mass[list(positions)].sum() / total) if total else None


@dataclass(frozen=True)
class Observation:
    identity: str
    payload_sha256: str
    sender_ms: float
    payload_bytes: int
    decode_ms: float
    reader_ms: float
    predicted_positive: bool | None
    reference_positive: bool
    spi: int | None = None  # Critical = 0
    reference_critical: bool | None = None
    cue: float | None = None

    def __post_init__(self):
        times = (self.sender_ms, self.decode_ms, self.reader_ms)
        if (not self.identity or len(self.payload_sha256) != 64 or self.payload_bytes <= 0
                or any(not math.isfinite(value) or value < 0 for value in times)
                or self.decode_ms <= 0 or self.spi not in (None, 0, 1, 2, 3)):
            raise ValueError("invalid measured observation")
        if type(self.reference_positive) is not bool:
            raise ValueError("independent binary reference is required")
        if any(value is not None and type(value) is not bool
               for value in (self.predicted_positive, self.reference_critical)):
            raise ValueError("invalid reader or Critical reference")
        if self.predicted_positive is None and self.reader_ms:
            raise ValueError("failed decode cannot have a reader decision duration")
        encode_cue(self.cue)


@dataclass(frozen=True)
class Job:
    token: int
    observation: Observation
    capture: float
    sender_start: float
    arrival: float
    deadline: float


def trace(rows, load, *, burst=False, sources=4, header_bytes=0, link_bytes_per_ms=None,
          fixed_deadline_ms=None):
    if not rows or not math.isfinite(load) or load <= 0 or sources < 1 or header_bytes < 0:
        raise ValueError("invalid trace parameters")
    if link_bytes_per_ms is not None and (not math.isfinite(link_bytes_per_ms) or link_bytes_per_ms <= 0):
        raise ValueError("link rate must be positive")
    base = median(row.decode_ms + row.reader_ms for row in rows)
    intervals = [0.25 if burst and i % 8 < 4 else 1.75 if burst else 1.0
                 for i in range(len(rows))]
    if fixed_deadline_ms is not None:
        if not math.isfinite(fixed_deadline_ms) or fixed_deadline_ms <= 0:
            raise ValueError("fixed deadline must be positive")
        base = sum(row.decode_ms + row.reader_ms for row in rows) / sum(intervals)
    deadline = 6 * base if fixed_deadline_ms is None else fixed_deadline_ms
    clocks = [0.] * sources
    capture = 0.
    jobs = []
    for i, (row, step) in enumerate(zip(rows, intervals)):
        sender = i % sources
        start = max(capture, clocks[sender])
        clocks[sender] = start + row.sender_ms
        delivery = 0. if link_bytes_per_ms is None else (row.payload_bytes + header_bytes) / link_bytes_per_ms
        jobs.append(Job(i, row, capture, start, clocks[sender] + delivery, capture + deadline))
        capture += base * step / load
    return tuple(jobs), capture + deadline


@dataclass(frozen=True)
class ReadyHeader:
    token: int
    arrival: float
    spi: int | None
    cue: float | None

def choose(ready, policy):
    oldest = lambda job: (job.arrival, job.token)
    if policy == "fifo" or any(job.spi is None for job in ready):
        return min(ready, key=oldest)
    priority = min(job.spi for job in ready)
    eligible = [job for job in ready if job.spi == priority]
    if policy == "spi_cue" and all(job.cue is not None for job in eligible):
        return min(eligible, key=lambda job: (-encode_cue(job.cue)[0], *oldest(job)))
    return min(eligible, key=oldest)


def replay(jobs, horizon, policy, *, detailed=False):
    if policy not in POLICIES or not jobs or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("invalid replay")
    if len({job.token for job in jobs}) != len(jobs):
        raise ValueError("duplicate job token")
    pending = sorted(jobs, key=lambda job: (job.arrival, job.token))
    ready, outcomes = [], {}
    i, now = 0, 0.
    while (ready or i < len(pending)) and now < horizon:
        if not ready:
            now = max(now, pending[i].arrival)
        if now >= horizon:
            break
        while i < len(pending) and pending[i].arrival <= now:
            ready.append(pending[i])
            i += 1
        visible = [ReadyHeader(j.token, j.arrival, j.observation.spi, j.observation.cue) for j in ready]
        selected = choose(visible, policy)
        job = next(j for j in ready if j.token == selected.token)
        ready.remove(job)
        obs = job.observation
        decode_end = now + obs.decode_ms
        finish = decode_end + obs.reader_ms
        state = ("unfinished" if finish > horizon else
                 "failed" if obs.predicted_positive is None else "completed")
        outcomes[job.token] = dict(dispatch_ms=now, decode_end_ms=decode_end if decode_end <= horizon else None,
                                   decision_ms=finish if state == "completed" else None, state=state)
        now = finish
    positives, delivered = set(), set()
    waits, timeline = [], []
    critical = critical_missed = critical_correct = false_reports = late = 0
    counts = dict(completed=0, failed=0, unfinished=0, unstarted=0, not_arrived=0)
    for job in jobs:
        obs = job.observation
        if obs.reference_positive:
            positives.add(obs.identity)
        missing = "not_arrived" if job.arrival >= horizon else "unstarted"
        result = outcomes.get(job.token, dict(dispatch_ms=None, decode_end_ms=None, decision_ms=None, state=missing))
        counts[result["state"]] += 1
        done = result["decision_ms"]
        timely = done is not None and done <= job.deadline
        late += done is not None and not timely
        if timely and obs.predicted_positive and obs.reference_positive:
            delivered.add(obs.identity)
        false_reports += done is not None and obs.predicted_positive and not obs.reference_positive
        if obs.reference_critical is True:
            critical += 1
            critical_missed += not timely
            critical_correct += timely and obs.predicted_positive == obs.reference_positive
        if job.arrival < horizon:
            wait = (result["dispatch_ms"] if result["dispatch_ms"] is not None else horizon) - job.arrival
            waits.append(wait)
        timeline.append(dict(token=job.token, identity=obs.identity, payload_sha256=obs.payload_sha256,
                             capture_ms=job.capture, sender_start_ms=job.sender_start,
                             sender_finish_ms=job.sender_start + obs.sender_ms, arrival_ms=job.arrival,
                             payload_bytes=obs.payload_bytes,
                             deadline_ms=job.deadline, **result))
    known_critical = sum(job.observation.reference_critical is not None for job in jobs)
    report = dict(offered_positive_ids=len(positives), timely_correct_positive_ids=len(delivered),
                false_reports=int(false_reports), critical_offered=critical if known_critical else None,
                critical_missed=critical_missed if known_critical else None,
                critical_reference_available=known_critical,
                spi_available=sum(job.observation.spi is not None for job in jobs),
                p95_observed_wait_ms=float(np.quantile(waits, .95)) if waits else None,
                **counts, timeline=timeline)
    if detailed:
        report.update(late_decisions=late,
                      critical_completed_on_time=critical-critical_missed if known_critical else None,
                      critical_correct_on_time=critical_correct if known_critical else None)
    return report
