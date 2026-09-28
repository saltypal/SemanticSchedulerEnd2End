from pathlib import Path

import numpy as np

from SemParam.delivered_history import DeliveredKnowledgeState
from SemParam.scheduler_features import SchedulerAnnotation, SchedulerFeatureExtractor


class FakeEmbeddingBackend:
    def encode_image(self, image_path: Path) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def encode_texts(self, texts):
        return {text: np.array([1.0, 0.0, 0.0], dtype=np.float32) for text in texts}


def test_static_feature_vector_and_delivered_history_novelty(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    extractor = SchedulerFeatureExtractor(backend=FakeEmbeddingBackend())
    features = extractor.extract(
        image,
        image_id="image-a",
        annotation=SchedulerAnnotation(task_impact=3, urgency=2, evidence_sufficiency=1),
    )
    assert features.context_relevance.value == 1.0
    assert features.delivered_history_novelty.available is False
    assert features.normalized_values()[1] == 1.0
    assert features.availability_mask() == (1, 0, 1, 1, 1)

    history = DeliveredKnowledgeState(features.task_profile_id)
    refreshed = features.with_delivered_history(history.snapshot())
    assert refreshed.delivered_history_novelty.available is False
    assert history.record_success(
        image_id="delivered-a",
        image_embedding=(1.0, 0.0, 0.0),
        delivered_at_ms=10.0,
        evidence_group_id="group-a",
        evidence_sufficiency=1,
    )
    delivered = features.with_delivered_history(history.snapshot())
    assert delivered.delivered_history_novelty.available is True
    assert delivered.delivered_history_novelty.value == 0.0
    assert delivered.delivered_history_novelty.provenance["history_version"] == 1


def test_history_only_changes_once_per_successful_image():
    history = DeliveredKnowledgeState("task")
    assert history.record_success(
        image_id="one", image_embedding=(0.0, 1.0), delivered_at_ms=1.0
    )
    assert not history.record_success(
        image_id="one", image_embedding=(0.0, 1.0), delivered_at_ms=2.0
    )
    snapshot = history.snapshot()
    assert snapshot.history_version == 1
    assert snapshot.successful_delivery_count == 1


def test_prototype_annotation_does_not_claim_human_review(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    features = SchedulerFeatureExtractor(backend=FakeEmbeddingBackend()).extract(
        image, image_id="prototype-image",
        annotation=SchedulerAnnotation(
            task_impact=2, urgency=3, evidence_sufficiency=1,
            annotation_source="prototype_default_no_human_review"),
    )
    assert features.task_impact.method == "prototype_task_rubric"
    assert features.urgency.method == "prototype_task_rubric"
    assert features.evidence_sufficiency.method == "prototype_default_sufficiency"
