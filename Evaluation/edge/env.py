"""Event-driven decoder-queue simulator.

``EdgeEnv.run(policy)`` replays a job trace against ``workers`` exclusive decoders. Arrivals enter a
queue; whenever a worker is free the policy picks a batch and/or purges. Reuses the shared
``contracts`` types and ``receiver.Receiver``; metrics live in ``metrics.py``.
"""
import heapq
import time
from dataclasses import dataclass
from math import exp

import numpy as np

from Evaluation.edge.contracts import QueuedJob, State

RATE_TAU = 30.0   # seconds, decay constant of the arrival-rate estimate


@dataclass
class RunResult:
    outcomes: list            # one dict per job (all jobs; ``measured`` marks post-warmup ones)
    busy_time: float          # total worker-seconds spent decoding
    decision_times: list      # seconds spent inside policy.select per call
    spec: object
    horizon: float = 0.0
    n_workers: int = 1

    def measured(self):
        return [o for o in self.outcomes if o["measured"]]


class EdgeEnv:
    def __init__(self, jobs, latency, spec, workers, max_batch, receiver, seed,
                 warmup_fraction=0.10, ttl_hard=True):
        self.jobs = sorted(jobs, key=lambda j: (j.header.arrival_time, j.id))
        self.latency, self.spec = latency, spec
        self.workers, self.max_batch = int(workers), int(max_batch)
        self.receiver, self.seed = receiver, seed
        self.warmup_fraction, self.ttl_hard = warmup_fraction, ttl_hard
        if self.workers < 1 or self.max_batch < 1:
            raise ValueError("workers and max_batch must be >= 1")

    def _queued(self, job, now):
        h = job.header
        hit = self._est.get(job.id)
        if hit is None or hit[0] != self._version:   # receiver memory unchanged -> reuse estimates
            hit = (self._version, self.receiver.dup_prob(h), self.receiver.u_hat(h))
            self._est[job.id] = hit
        return QueuedJob(job.id, h, now - h.capture_time, hit[1], hit[2])

    def _validate(self, dec, queue):
        ids = list(dec.batch)
        if len(set(ids)) != len(ids) or len(set(dec.purge)) != len(dec.purge):
            raise ValueError("duplicate ids in decision")
        for i in ids + list(dec.purge):
            if i not in queue:
                raise ValueError("job %r is not queued" % (i,))
        if set(ids) & set(dec.purge):
            raise ValueError("job both batched and purged")
        if ids and (len(ids) not in self.latency.batch_sizes or len(ids) > self.max_batch):
            raise ValueError("invalid batch size %d" % len(ids))

    @staticmethod
    def _row(job, status, finish, start, bsize, value=0.0, dup=False):
        h, t = job.header, job.truth
        return dict(id=job.id, tier=t.tier, event=t.event, source=t.source, status=status,
                    finish_time=finish, delay=None if finish is None else finish - h.capture_time,
                    value=value, u=t.fidelity, dup_flag=dup, batch_size=bsize,
                    wait_start=start, dup_group=t.dup_group, mission=h.mission,
                    arrival_time=h.arrival_time, capture_time=h.capture_time, measured=False)

    def run(self, policy):
        rng = np.random.default_rng(self.seed)
        policy.reset(self.spec, np.random.default_rng(self.seed + 1))
        spec, jobs = self.spec, self.jobs
        n_warm = int(self.warmup_fraction * len(jobs))
        measured_ids = {j.id for j in jobs[n_warm:]}
        h_max = max(spec.hard_deadline(k) for k in range(len(spec.weights)))
        queue, rows, comps = {}, {}, []
        delivered_groups, mission_val = set(), {}
        free, busy, seq, ai = self.workers, 0.0, 0, 0
        rate, last_arr, now = 0.0, None, 0.0
        dtimes = []
        self._est, self._version = {}, 0

        while ai < len(jobs) or comps:
            t_arr = jobs[ai].header.arrival_time if ai < len(jobs) else np.inf
            t_com = comps[0][0] if comps else np.inf
            now = min(t_arr, t_com)
            if t_com <= t_arr:
                _, _, batch, start = heapq.heappop(comps)
                free += 1
                done = {"ids": [], "tier_probs": [], "delays": []}
                for job in batch:
                    k, d = job.truth.tier, now - job.header.capture_time
                    grp = job.truth.dup_group
                    dup = grp is not None and grp in delivered_groups
                    zeta = (spec.zeta_critical_copy if k == 0 else 0.0) if dup else 1.0
                    v = spec.weights[k] * job.truth.fidelity * spec.g(k, d) * zeta
                    rows[job.id] = self._row(job, "decoded", now, start, len(batch), v, dup)
                    if grp is not None:
                        delivered_groups.add(grp)
                    self.receiver.deliver(job.header)
                    self._version += 1
                    m = job.header.mission
                    # header-only served value: expected tier weight from the SPI posterior (no truth)
                    mission_val[m] = mission_val.get(m, 0.0) + sum(p * w for p, w in zip(job.header.tier_probs, spec.weights))
                    done["ids"].append(job.id)
                    done["tier_probs"].append(job.header.tier_probs)
                    done["delays"].append(d)
                policy.observe("complete", dict(done, time=now, batch_size=len(batch)))
            else:
                job = jobs[ai]
                ai += 1
                if last_arr is not None:
                    rate = rate * exp(-(now - last_arr) / RATE_TAU)
                rate += 1.0 / RATE_TAU
                last_arr = now
                queue[job.id] = job
                policy.observe("arrival", dict(id=job.id, time=now,
                                               tier_probs=job.header.tier_probs))

            if self.ttl_hard:
                for jid in [i for i, j in queue.items() if now - j.header.capture_time > h_max]:
                    job = queue.pop(jid)
                    rows[jid] = self._row(job, "expired", now, None, None)
                    policy.observe("expire", dict(ids=[jid], time=now))

            while free > 0 and queue:
                cur = rate * exp(-(now - last_arr) / RATE_TAU) if last_arr is not None else 0.0
                state = State(now, tuple(self._queued(j, now) for j in queue.values()), free,
                              self.workers, self.latency, spec, self.max_batch,
                              arrival_rate=cur, mission_served=tuple(sorted(mission_val.items())))
                t0 = time.perf_counter()
                dec = policy.select(state)
                dtimes.append(time.perf_counter() - t0)
                self._validate(dec, queue)
                for jid in dec.purge:
                    job = queue.pop(jid)
                    rows[jid] = self._row(job, "purged", now, None, None)
                    policy.observe("purge", dict(ids=[jid], time=now))
                if not dec.batch:
                    if dec.purge:
                        continue
                    break
                batch = [queue.pop(i) for i in dec.batch]
                s = self.latency.sample(len(batch), rng)
                busy += s
                free -= 1
                seq += 1
                heapq.heappush(comps, (now + s, seq, batch, now))

        for jid, job in queue.items():
            rows[jid] = self._row(job, "queued_at_end", None, None, None)
        for jid, r in rows.items():
            r["measured"] = jid in measured_ids
        return RunResult(list(rows.values()), busy, dtimes, spec, now, self.workers)
