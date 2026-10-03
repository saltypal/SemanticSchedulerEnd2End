"""Non-preemptive whole-latent service, with static batches as the default.

INFRA owns all pending-job and receiver-history state. The same engine can
accept staggered arrivals; source/device IDs are neither required nor actions.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from time import perf_counter_ns

from INFRA.latent.contracts import LatentCandidate, LatentSnapshot
from SemParam.features.latent import confirmed_novelty


def run_latent_schedule(jobs, policy, config, *, delivery_evidence=None,
                        initial_receiver_embeddings=(), on_step=None,
                        delivery_handler=None):
    jobs = tuple(jobs)
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("A nonempty batch with unique job IDs is required")
    if len({job.task_id for job in jobs}) != 1:
        raise ValueError("One schedule must use one task")
    if any(job.resource_unit != config.resource_unit for job in jobs):
        raise ValueError("Job and service-rate resource units differ")
    if config.mode == "static" and any(job.arrival_ms != 0 for job in jobs):
        raise ValueError("Static scheduling requires every arrival_ms=0")
    evidence = {} if delivery_evidence is None else dict(delivery_evidence)
    if evidence and delivery_handler is not None:
        raise ValueError("Choose cached evidence or live delivery, not both")
    if evidence and set(evidence) != {job.job_id for job in jobs}:
        raise ValueError("Delivery evidence must cover exactly the offered jobs")
    if any(key != value.job_id for key, value in evidence.items()):
        raise ValueError("Delivery evidence key/identity mismatch")
    dimensions = {len(job.embedding) for job in jobs if job.embedding}
    history = [tuple(item) for item in initial_receiver_embeddings]
    dimensions.update(len(item) for item in history)
    dimensions.update(len(item.receiver_embedding) for item in evidence.values() if item.receiver_embedding)
    if len(dimensions) > 1:
        raise ValueError("Source and receiver embedding schemas differ")
    for embedding in history:
        if not embedding or not all(math.isfinite(value) for value in embedding) or not any(embedding):
            raise ValueError("Initial receiver embeddings must be finite and nonzero")
    pending = {job.job_id: job for job in jobs}
    indices = {job.job_id: index for index, job in enumerate(jobs)}
    history_version = 0
    now = 0.0
    outcomes, trace = [], []
    while pending:
        if now >= config.horizon_ms:
            for job in pending.values():
                outcomes.append(_terminal(job, "horizon_censored", now))
            break
        for job in tuple(pending.values()):
            if job.arrival_ms > now:
                continue
            completion = now + config.service_ms(job)
            if job.deadline_ms is not None and completion > job.deadline_ms:
                # Service cannot meet the declared deadline; do not waste the
                # remaining capacity or conceal this as a successful delivery.
                outcomes.append(_terminal(job, "deadline_infeasible", now))
                del pending[job.job_id]
        visible = [job for job in pending.values() if job.arrival_ms <= now]
        candidates = []
        for job in visible:
            service_ms = config.service_ms(job)
            candidates.append(LatentCandidate(
                job_id=job.job_id, image_id=job.image_id, payload_id=job.payload_id,
                input_index=indices[job.job_id], priority_class=job.predicted_priority,
                priority_source=job.priority_source, resource_units=job.resource_units,
                resource_unit=job.resource_unit, service_ms=service_ms,
                deadline_ms=job.deadline_ms,
                slack_ms=None if job.deadline_ms is None else job.deadline_ms - now - service_ms,
                context_relevance=job.context_relevance, task_importance=job.task_importance,
                novelty=confirmed_novelty(job.embedding, history),
                eligible=now + service_ms <= config.horizon_ms))
        snapshot = LatentSnapshot(jobs[0].task_id, now, history_version, tuple(candidates))
        if not any(item.eligible for item in candidates):
            future = [job.arrival_ms for job in pending.values() if job.arrival_ms > now]
            if future:
                now = min(config.horizon_ms, min(future))
                continue
            for job in pending.values():
                outcomes.append(_terminal(job, "horizon_censored", config.horizon_ms))
            break
        started = perf_counter_ns()
        job_id, score, reason = policy.select(snapshot)
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        allowed = {item.job_id for item in candidates if item.eligible}
        if job_id not in allowed:
            raise ValueError("Policy selected an ineligible latent job")
        chosen = pending.pop(job_id)
        completed = now + config.service_ms(chosen)
        # A live transport receives only the selected whole job, after selection.
        # No future channel/task outcome is exposed to the policy.
        delivery = evidence.get(job_id)
        if delivery_handler is not None:
            from INFRA.latent.contracts import DeliveryEvidence
            delivery = delivery_handler(chosen)
            if not isinstance(delivery, DeliveryEvidence) or delivery.job_id != job_id:
                raise ValueError("Live transport returned invalid delivery identity")
            if delivery.receiver_embedding and dimensions and len(delivery.receiver_embedding) not in dimensions:
                raise ValueError("Live receiver embedding schema differs")
            if delivery.receiver_embedding:
                dimensions.add(len(delivery.receiver_embedding))
        status = ("served_unverified" if delivery is None else
                  "completed" if delivery.transport_success else "failed_transport")
        outcome = _terminal(chosen, status, completed)
        outcome.update(started_ms=now, completed_ms=completed,
                       wait_ms=now - chosen.arrival_ms,
                       deadline_met=None if chosen.deadline_ms is None else completed <= chosen.deadline_ms,
                       transport_success=None if delivery is None else delivery.transport_success,
                       task_quality=None if delivery is None else delivery.task_quality,
                       delivery_provenance="unavailable" if delivery is None else delivery.provenance,
                       evidence_group_id=None if delivery is None else delivery.evidence_group_id)
        outcomes.append(outcome)
        trace.append({"step": len(trace) + 1, "now_ms": now, "completed_ms": completed,
                      "job_id": job_id, "payload_id": chosen.payload_id,
                      "history_version": history_version, "score": score, "reason": reason,
                      "policy_inference_ms": elapsed_ms,
                      "candidates": [asdict(item) for item in candidates]})
        if (config.history_mode == "confirmed" and delivery is not None
                and delivery.transport_success and delivery.receiver_embedding):
            # Source embeddings do not establish what a lossy receiver recovered.
            history.append(delivery.receiver_embedding)
            history_version += 1
        if on_step is not None:
            on_step(trace[-1], outcome)
        now = completed
    return {"schema": "latent-schedule/v1", "policy": policy.name,
            "task_id": jobs[0].task_id, "config": asdict(config),
            "jobs": [asdict(job) for job in jobs], "decision_trace": trace,
            "outcomes": outcomes, "history_version": history_version,
            "delivery_mode": ("live_evidence" if delivery_handler is not None else
                              "unverified" if not evidence else "cached_evidence"),
            "history_note": "Only confirmed receiver embeddings update novelty; missing novelty remains unavailable."}


def _terminal(job, status, now):
    return {"job_id": job.job_id, "image_id": job.image_id,
            "payload_id": job.payload_id, "status": status,
            "arrival_ms": job.arrival_ms, "terminal_ms": now,
            "resource_units": job.resource_units, "resource_unit": job.resource_unit,
            "priority_class": job.predicted_priority}
