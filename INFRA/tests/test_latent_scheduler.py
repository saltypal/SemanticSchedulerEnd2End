"""Whole-payload scheduling checks independent of a codec or neural framework."""
from dataclasses import replace
import json

import pytest

from INFRA.latent_contracts import DeliveryEvidence, LatentJob, LatentScheduleConfig
from INFRA.latent_evaluation import evaluate_latent_schedule
from INFRA.latent_simulation import run_latent_schedule
from INFRA.experiments.static_latent_experiment import compare_batch
from Scheduler.latent_policies import LatentPolicy


def job(name, priority=1, units=2, **kwargs):
    return LatentJob(name, name, "task", name, units, "simulated_units", "synthetic-test",
                     priority_class=priority, priority_source="explicit-test", **kwargs)


def run(jobs, policy="semantic", evidence=None, **config):
    return run_latent_schedule(jobs, LatentPolicy(policy),
        LatentScheduleConfig(1, "simulated_units", config.pop("horizon_ms", 100), **config),
        delivery_evidence=evidence)


def order(result):
    return [row["job_id"] for row in result["decision_trace"]]


def test_whole_payload_priority_and_baselines():
    jobs = (job("long", 2, 8), job("urgent", 0, 4), job("short", 1, 1))
    assert order(run(jobs, "fifo")) == ["long", "urgent", "short"]
    assert order(run(jobs, "sjf")) == ["short", "urgent", "long"]
    result = run(jobs)
    assert order(result) == ["urgent", "short", "long"]
    assert result["outcomes"][0]["completed_ms"] == 4
    assert evaluate_latent_schedule(result)["confirmed_deliveries"] is None
    assert all("ue_id" not in row for row in result["jobs"])


def test_deadline_exact_boundary_and_infeasible_pruning():
    result = run((job("impossible", 0, 5, deadline_ms=4),
                  job("boundary", 1, 4, deadline_ms=4)))
    assert order(result) == ["boundary"]
    assert next(row for row in result["outcomes"] if row["job_id"] == "impossible")["status"] == "deadline_infeasible"


def test_dynamic_arrivals_and_whole_job_budget():
    result = run((job("first", 1, 4), job("future", 0, 3, arrival_ms=5)),
                 mode="dynamic", horizon_ms=8)
    assert order(result) == ["first", "future"]
    assert result["decision_trace"][1]["now_ms"] == 5
    assert not order(run((job("too_big", units=9),), horizon_ms=8))


def test_receiver_history_uses_reconstruction_only_and_never_failed_delivery():
    jobs = (job("a", embedding=(1., 0.)), job("b", embedding=(1., 0.)),
            job("c", embedding=(0., 1.)))
    evidence = {name: DeliveryEvidence(name, True, "synthetic-test", receiver_embedding=(0., 1.))
                for name in ("a", "b", "c")}
    result = run(jobs, evidence=evidence)
    candidates = {row["job_id"]: row for row in result["decision_trace"][1]["candidates"]}
    assert candidates["b"]["novelty"] == .5
    assert candidates["c"]["novelty"] == 0
    assert result["history_version"] == 3
    evidence["a"] = DeliveryEvidence("a", False, "synthetic-failure")
    result = run(jobs, evidence=evidence)
    assert result["decision_trace"][1]["history_version"] == 0
    assert all(row["novelty"] is None for row in result["decision_trace"][1]["candidates"])
    frozen = run(jobs, evidence=evidence, history_mode="frozen")
    assert frozen["history_version"] == 0


def test_reference_labels_are_evaluation_only_and_quality_is_not_invented():
    result = run((job("a", 0), job("b", 2)))
    metrics = evaluate_latent_schedule(result, reference_priority={"a": 3, "b": 0})
    assert metrics["reference_pair_agreement"] == 0
    assert metrics["reference_ndcg"] < 1
    assert metrics["evidence_quality_auc"] is None
    assert order(result) == ["a", "b"]


def test_ordinal_probability_priority_and_invalid_inputs():
    original = job("x")
    predicted = replace(original, priority_class=None, priority_probabilities=(.3, .3, .4, 0))
    assert predicted.predicted_priority == 1
    with pytest.raises(ValueError):
        replace(original, resource_units=0)
    with pytest.raises(ValueError):
        run((original, replace(original, job_id="y", resource_unit="digital_bits")))
    with pytest.raises(ValueError):
        run((replace(original, arrival_ms=1),))
    with pytest.raises(ValueError):
        run((original,), evidence={"other": DeliveryEvidence("other", True, "test")})


def test_live_delivery_receives_only_selected_payload_and_updates_history():
    sent = []
    def transmit(selected):
        sent.append(selected.payload_id)
        return DeliveryEvidence(selected.job_id, True, "live-test", receiver_embedding=(1., 0.))
    result = run_latent_schedule((job("low", 2, embedding=(0., 1.)),
                                 job("high", 0, embedding=(1., 0.))),
        LatentPolicy(), LatentScheduleConfig(1, "simulated_units", 20), delivery_handler=transmit)
    assert sent == ["high", "low"]
    assert result["decision_trace"][1]["history_version"] == 1
    assert result["delivery_mode"] == "live_evidence"
    assert evaluate_latent_schedule(result)["confirmed_deliveries"] == 2
    assert evaluate_latent_schedule(result)["evidence_quality_auc"] is None


def test_demo_persists_every_policy_and_decision(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    output = compare_batch(root / "Scheduler/config/static_latent_demo.json", tmp_path, verbose=False)
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics) == 6
    assert next(row for row in metrics if row["policy"] == "semantic")["predicted_priority_gate_violations"] == 0
    events = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert sum(row["stage"] == "decision" for row in events) == sum(row["served"] for row in metrics)
