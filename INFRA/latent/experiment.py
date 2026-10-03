"""Cache-only static latent experiments and explicit adapters for older runs."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from pathlib import Path
import platform
import subprocess
import uuid

from INFRA.latent.contracts import DeliveryEvidence, LatentJob, LatentScheduleConfig
from INFRA.latent.evaluation import evaluate_latent_schedule
from INFRA.latent.simulation import run_latent_schedule
from Scheduler.latent.policies import LatentPolicy, POLICIES


def read_batch(path):
    """Load public jobs plus explicitly segregated evaluator-only evidence."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema") != "latent-batch/v1":
        raise ValueError("Expected schema latent-batch/v1")
    jobs = []
    for record in data["jobs"]:
        row = dict(record)
        row["embedding"] = tuple(row.get("embedding", ()))
        if row.get("priority_probabilities") is not None:
            row["priority_probabilities"] = tuple(row["priority_probabilities"])
        jobs.append(LatentJob(**row))
    config = LatentScheduleConfig(**data["config"])
    evidence = {}
    for row in data.get("evaluation", {}).get("delivery_evidence", []):
        record = dict(row)
        record["receiver_embedding"] = tuple(record.get("receiver_embedding", ()))
        item = DeliveryEvidence(**record)
        if item.job_id in evidence:
            raise ValueError("Duplicate delivery evidence")
        evidence[item.job_id] = item
    references = data.get("evaluation", {}).get("reference_priority", {})
    return tuple(jobs), config, evidence, references, data.get("provenance", {})


def compare_batch(path, output_root, *, policy_names=POLICIES, verbose=True):
    jobs, config, evidence, references, input_provenance = read_batch(path)
    policies = tuple(LatentPolicy(name) for name in policy_names)
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("A nonempty batch with unique job IDs is required")
    if len({job.task_id for job in jobs}) != 1:
        raise ValueError("One schedule must use one task")
    dimensions = {len(job.embedding) for job in jobs if job.embedding}
    dimensions.update(len(item.receiver_embedding) for item in evidence.values() if item.receiver_embedding)
    if len(dimensions) > 1:
        raise ValueError("Source and receiver embedding schemas differ")
    if any(type(value) is not int or value not in range(4) for value in references.values()):
        raise ValueError("Reference priority must be 0..3")
    if not policies or len({policy.name for policy in policies}) != len(policies):
        raise ValueError("Choose unique policies")
    # Fail configuration/input gates before creating a run directory.
    if any(policy.name.startswith("priority_") or policy.name == "semantic" for policy in policies):
        if any(job.predicted_priority is None for job in jobs):
            raise ValueError("Priority comparisons require supplied or predicted classes")
    if any(job.resource_unit != config.resource_unit for job in jobs):
        raise ValueError("Resource unit mismatch")
    if config.mode == "static" and any(job.arrival_ms for job in jobs):
        raise ValueError("Static batch has nonzero arrivals")
    if evidence and set(evidence) != {job.job_id for job in jobs}:
        raise ValueError("Partial delivery evidence is not a comparable experiment")
    if references and set(references) != {job.job_id for job in jobs}:
        raise ValueError("Reference priorities do not cover the batch")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = Path(output_root) / "Scheduler" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    provenance = source_provenance(repository)
    provenance.update(batch_sha256=sha256(Path(path).read_bytes()).hexdigest(),
                      input_provenance=input_provenance, run_id=run_id,
                      python=platform.python_version(), platform=platform.platform(),
                      resource_unit=config.resource_unit)
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for package in ("numpy", "pandas", "matplotlib", "torch", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not_installed"
    provenance["packages"] = packages
    provenance["execution_backend"] = "cached_cpu"
    _json(destination / "provenance.json", provenance)
    _json(destination / "config.resolved.json", asdict(config))
    logs = destination / "steps.jsonl"
    def log(event):
        with logs.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        if verbose:
            print(json.dumps(event, allow_nan=False), flush=True)
    _json(destination / "status.json", {"status": "running"})
    summaries = []
    try:
        log({"stage": "validated", "jobs": len(jobs), "mode": config.mode,
             "resource_unit": config.resource_unit, "delivery_evidence": bool(evidence)})
        for policy in policies:
            log({"stage": "policy_start", "policy": policy.name})
            def on_step(decision, outcome):
                log({"stage": "decision", "policy": policy.name,
                     "step": decision["step"], "job_id": decision["job_id"],
                     "payload_id": decision["payload_id"], "status": outcome["status"],
                     "started_ms": outcome["started_ms"], "completed_ms": outcome["completed_ms"],
                     "history_version": decision["history_version"]})
            result = run_latent_schedule(jobs, policy, config,
                                        delivery_evidence=evidence, on_step=on_step)
            metrics = evaluate_latent_schedule(result, reference_priority=references)
            _json(destination / f"{policy.name}.trace.json", result)
            summaries.append(metrics)
            log({"stage": "policy_complete", "policy": policy.name,
                 "served": metrics["served"], "reference_ndcg": metrics["reference_ndcg"]})
        _json(destination / "metrics.json", summaries)
        flat = [{key: value for key, value in row.items() if key != "per_class"} for row in summaries]
        with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        _json(destination / "status.json", {"status": "completed"})
        log({"stage": "complete", "output": str(destination.resolve())})
    except Exception as error:
        _json(destination / "status.json", {"status": "failed", "error": str(error)})
        log({"stage": "failed", "error": str(error)})
        raise
    return destination


def convert_cached_run(run_dir, output, *, payload_costs=None, task_id=None,
                       service_rate=1.0, limit=None, seed=42):
    """Adapt saved features/predictions, without retraining or invoking codecs.

    Labels go only to evaluation. Predictions must come from a separately
    trained model; no fallback to base_priority is permitted.
    """
    import random
    from INFRA.latent.contracts import finite
    finite(service_rate, "service rate", minimum=1e-12)
    from INFRA.experiments.semantic_scheduler_experiment import _load_feature_rows
    run = Path(run_dir)
    manifest = run / "dataset_manifest.csv"
    features_path = run / "feature_cache.jsonl"
    if not features_path.exists():
        features_path = run / "cache" / "feature_cache.jsonl"
    predictions_path = run / "models" / "clip_priority_predictions.json"
    costs_path = Path(payload_costs) if payload_costs else run / "payload_costs.csv"
    features = _load_feature_rows(features_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with costs_path.open(newline="", encoding="utf-8") as handle:
        costs = {}
        for row in csv.DictReader(handle):
            if row["image_id"] in costs:
                raise ValueError("Duplicate payload cost ID")
            costs[row["image_id"]] = row
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate manifest image IDs")
    random.Random(seed).shuffle(rows)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    jobs, references = [], {}
    for row in rows:
        image_id = row["image_id"]
        feature, source_hash = features[image_id]
        if source_hash != row["source_sha256"]:
            raise ValueError(f"Feature/manifest source mismatch: {image_id}")
        cost = costs[image_id]
        unit = cost["resource_unit"]
        provenance = cost.get("payload_provenance", "simulated_fixed")
        if unit == "complex_channel_uses":
            checkpoint = cost.get("checkpoint_sha256", "")
            if len(checkpoint) != 64:
                raise ValueError("Measured costs require checkpoint SHA256")
            provenance = "swinjscc:" + checkpoint
        units = int(cost["channel_uses"] if unit == "complex_channel_uses" else cost["resource_units"])
        probability = predictions[image_id]
        values = tuple(float(probability.get(str(label), 0)) for label in range(4))
        # The old prototype's task impact is a reference-rubric annotation.
        # Do not inject it into the generic deployable policy as a prediction.
        jobs.append(LatentJob(
            image_id, image_id, task_id or "post_disaster_access_and_damage_triage_v1",
            image_id, units, unit, provenance, priority_probabilities=values,
            priority_source="cached_clip_priority_classifier_unverified_calibration",
            context_relevance=feature.context_relevance.value,
            embedding=feature.embedding))
        references[image_id] = int(row["base_priority"])
    if not jobs or len({job.resource_unit for job in jobs}) != 1:
        raise ValueError("Nonempty homogeneous payload costs are required")
    if jobs[0].resource_unit == "complex_channel_uses" and len({job.payload_provenance for job in jobs}) != 1:
        raise ValueError("Cannot mix codec checkpoint identities")
    total_service = sum(job.resource_units for job in jobs) / service_rate
    batch = {"schema": "latent-batch/v1", "jobs": [asdict(job) for job in jobs],
             "config": asdict(LatentScheduleConfig(service_rate, jobs[0].resource_unit,
                                                    max(1.0, total_service * 1.1))),
             "evaluation": {"reference_priority": references},
             "provenance": {
                 "adapted_run": str(run.resolve()), "seed": seed,
                 "reference_note": "Existing prototype rubric; not independently reviewed general-task labels",
                 "feature_sha256": sha256(features_path.read_bytes()).hexdigest(),
                 "prediction_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
                 "payload_costs_sha256": sha256(costs_path.read_bytes()).hexdigest(),
                 "delivery_note": "Payload demand only; transport and decoded task usefulness remain unavailable"}}
    _json(Path(output), batch)
    return Path(output)


def source_provenance(root):
    digest = sha256()
    for name in ("EncDec", "SemParam", "Scheduler", "INFRA"):
        for path in sorted((root / name).rglob("*.py")):
            if "__pycache__" not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unavailable", "source_archive"
        snapshot = root / "snapshot_provenance.json"
        if snapshot.exists():
            saved = json.loads(snapshot.read_text(encoding="utf-8"))
            if saved["source_digest_sha256"] != digest.hexdigest():
                raise ValueError("Source archive differs from its recorded snapshot")
            commit, dirty = saved["git_commit"], saved["git_dirty"]
    return {"git_commit": commit, "git_dirty": bool(dirty), "source_digest_sha256": digest.hexdigest()}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
