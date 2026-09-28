"""One synthetic, explicitly non-research fixture through the full API boundary."""

from dataclasses import asdict
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from INFRA.experiments.semantic_scheduler_experiment import (
    build_suites_from_manifest, evaluate_policy_suite, load_catalog,
    simulated_fixed_payload_costs,
)
from INFRA.ladi_manifest import MANIFEST_FIELDS, NATIVE_LABELS, STRATA, draft_ladi60_manifest
from INFRA.semantic_config import load_scheduler_config, load_ue_profiles
from INFRA.semantic_contracts import (
    FeatureValue, ResourceBudgetSemantic, SchedulingFeatureVectorV2,
)


class FixtureEmbedding:
    def encode_image(self, path):
        digest = hashlib.sha256(path.read_bytes()).digest()
        return np.asarray((digest[0] + 1, digest[1] + 1, digest[2] + 1), dtype=np.float32)


def _value(value):
    return FeatureValue(value, True, "synthetic_test_fixture", "v2")


def test_manifest_cache_stream_policy_and_metrics_join(tmp_path):
    inputs = []
    for stratum_index, stratum in enumerate(STRATA):
        for image_index in range(10):
            labels = {name: False for name in NATIVE_LABELS}
            labels["roads_any" if stratum == "infrastructure_control" else stratum] = True
            image = Image.new("RGB", (2, 2),
                              (stratum_index * 25, image_index * 20, image_index + 1))
            inputs.append({"image": image, **labels})
    draft = draft_ladi60_manifest(
        inputs, tmp_path / "source", dataset_revision="test-only",
        source_split="train", embedding_backend=FixtureEmbedding(),
        pool_per_stratum=10)
    with draft.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["evidence_sufficiency"] = "1"
        row["reviewer"] = "synthetic_test_fixture"
        row["review_status"] = "reviewed"
        row["annotation_source"] = "synthetic_test_fixture"
    with draft.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    cache = tmp_path / "features.jsonl"
    with cache.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            features = SchedulingFeatureVectorV2(
                _value(0.5), FeatureValue.unavailable("test", "v2"),
                _value(float(row["task_impact"])),
                _value(float(row["urgency"])), _value(1.0),
                embedding=(float(index + 1), 1.0))
            handle.write(json.dumps({"image_id": row["image_id"],
                                     "source_sha256": row["source_sha256"],
                                     "features": asdict(features)}) + "\n")
    costs = simulated_fixed_payload_costs(draft, tmp_path / "costs.csv")
    catalogue = load_catalog(draft, cache, costs)
    root = Path(__file__).resolve().parents[2]
    config = load_scheduler_config(root / "INFRA/config/semantic_scheduler_demo.yaml")
    profiles = load_ue_profiles(root / "INFRA/config/ue_profiles.yaml")
    suites = build_suites_from_manifest(catalogue, draft, config=config,
                                        ue_profiles=profiles)
    assert len(suites["train_by_seed"][42]) == 300
    assert len(suites["faculty_demo"][0].jobs) == 25
    assert set(Counter(job.candidate.ue_id for job in suites["faculty_demo"][0].jobs).values()) == {5}
    ue_priorities = defaultdict(set)
    for episode in suites["train_by_seed"][42][:30]:
        for job in episode.jobs:
            ue_priorities[job.candidate.ue_id].add(job.base_priority)
    assert all(len(priorities) > 1 for priorities in ue_priorities.values())
    priorities = {job.candidate.image_id: job.base_priority for job in catalogue}
    _, metrics = evaluate_policy_suite(
        suites["faculty_demo"], ResourceBudgetSemantic(1.0),
        priority_by_image=priorities, predictions_by_image={}, model_artifacts={},
        policy_names=("fifo", "semantic_greedy", "privileged_marginal_utility"))
    assert {row["policy"] for row in metrics} == {
        "fifo", "semantic_greedy", "privileged_marginal_utility"}
    assert all(row["jobs"] == 25 for row in metrics)
