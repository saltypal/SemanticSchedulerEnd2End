"""Slotted service abstraction: resource feasibility, arrivals, expiry, traces.

No PHY success or reconstructed quality is inferred by this simulator.
Each selected job finishes at the end of the slot. Deadlines are inclusive
at completion, and jobs with deadlines <= now expire before selection.
"""
from dataclasses import asdict
import math
from time import perf_counter_ns
from INFRA.scheduling.contracts import QueueSnapshot, finite_nonnegative

def simulate(jobs, policy, budget, *, slot_ms=10, horizon_ms=200):
    finite_nonnegative(slot_ms, 'slot_ms')
    finite_nonnegative(horizon_ms, 'horizon_ms')
    if slot_ms <= 0 or horizon_ms <= 0:
        raise ValueError('slot_ms and horizon_ms must be positive')
    jobs = tuple(jobs)
    QueueSnapshot(0, jobs)  # Contract validation before policy state changes.
    policy.reset()
    pending = {job.job_id: job for job in jobs}
    records = []
    decisions = []
    for slot in range(math.ceil(horizon_ms / slot_ms)):
        now = slot * slot_ms
        if now + slot_ms > horizon_ms:
            break
        for job in list(pending.values()):
            if job.arrival_ms <= now and job.deadline_ms is not None and job.deadline_ms <= now:
                records.append({**asdict(job), 'status': 'expired', 'completion_ms': None,
                                'wait_ms': now - job.arrival_ms, 'deadline_met': False})
                del pending[job.job_id]
        snapshot = QueueSnapshot(now, tuple(pending.values()))
        start = perf_counter_ns()
        decision = policy.decide(snapshot, budget)
        runtime_ms = (perf_counter_ns() - start) / 1e6
        eligible = {job.job_id: job for job in snapshot.eligible()}
        if len(set(decision.job_ids)) != len(decision.job_ids) or any(key not in eligible for key in decision.job_ids):
            raise ValueError('Policy selected duplicate or ineligible job')
        actual_units = sum(eligible[key].resource_units for key in decision.job_ids)
        if (actual_units != decision.used_units or actual_units > budget.units
                or len(decision.job_ids) > budget.max_jobs or decision.resource_unit != budget.unit
                or decision.now_ms != now):
            raise ValueError('Policy violated resource or decision contract')
        decisions.append({**asdict(decision), 'scheduling_time_ms': runtime_ms})
        for key in decision.job_ids:
            job = pending.pop(key)
            finish = now + slot_ms
            met = None if job.deadline_ms is None else finish <= job.deadline_ms
            records.append({**asdict(job), 'status': 'completed', 'completion_ms': finish,
                            'wait_ms': now - job.arrival_ms, 'deadline_met': met})
        if not pending:
            break
    for job in pending.values():
        arrived = job.arrival_ms < horizon_ms
        expired = arrived and job.deadline_ms is not None and job.deadline_ms <= horizon_ms
        records.append({**asdict(job), 'status': 'expired' if expired else 'pending' if arrived else 'not_arrived',
                        'completion_ms': None, 'wait_ms': max(0, horizon_ms-job.arrival_ms) if arrived else None,
                        'deadline_met': False if expired else None})
    completed = [row for row in records if row['status'] == 'completed']
    deadlines = [row for row in records if row['deadline_ms'] is not None and row['arrival_ms'] < horizon_ms]
    return {'policy': policy.name, 'fixture_type': 'synthetic', 'resource_unit': budget.unit,
            'resource_budget': asdict(budget), 'slot_ms': slot_ms, 'horizon_ms': horizon_ms, 'decisions': decisions,
            'jobs': sorted(records, key=lambda row: row['job_id']),
            'metrics': {'jobs': len(jobs), 'completed': len(completed),
                'expired': sum(row['status'] == 'expired' for row in records),
                'pending': sum(row['status'] == 'pending' for row in records),
                'deadline_jobs': len(deadlines),
                'deadline_met_fraction': sum(row['deadline_met'] is True for row in deadlines) / len(deadlines) if deadlines else None,
                'mean_wait_ms': sum(row['wait_ms'] for row in completed)/len(completed) if completed else None,
                'mean_scheduling_time_ms': sum(row['scheduling_time_ms'] for row in decisions)/len(decisions) if decisions else None,
                'completed_per_ue': {ue: sum(row['ue_id'] == ue for row in completed) for ue in sorted({job.ue_id for job in jobs})}}}
