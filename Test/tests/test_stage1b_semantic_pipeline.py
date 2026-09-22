from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("networkx")

from TransformerPipeline.KnowledgeBases import NetworkXKnowledgeBase
from TransformerPipeline.catalog import SemanticPipelineSettings
from TransformerPipeline.clip_backbone import _feature_tensor
from TransformerPipeline.packet_builder import build_semantic_packet, format_semantic_packet
from TransformerPipeline.pipeline import TransformerParameterPipeline
from TransformerPipeline.semantic_header import SemanticHeaderCodec
from INFRA.Artifacts import ParameterProducerLevel, ParameterValue, TaskContext


NOW = datetime(2026, 9, 16, 0, 5, tzinfo=UTC)


class FakeEmbeddingBackend:
    """No-network CLIP replacement with transparent two-dimensional semantics."""

    def __init__(self) -> None:
        self.image_forward_calls = 0
        self.text_forward_calls = 0

    def encode_image(self, image_path: Path) -> np.ndarray:
        self.image_forward_calls += 1
        if "other" in image_path.name or "flower" in image_path.name:
            return np.asarray((0.0, 1.0), dtype=np.float32)
        return np.asarray((1.0, 0.0), dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> dict[str, np.ndarray]:
        self.text_forward_calls += 1
        result = {}
        for text in texts:
            lowered = text.lower()
            result[text] = np.asarray((0.0, 1.0) if "flower" in lowered else (1.0, 0.0), dtype=np.float32)
        return result


def _write_image_marker(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _settings() -> SemanticPipelineSettings:
    return SemanticPipelineSettings(
        concept_prompts={"fire": "fire", "flower": "flower"},
        freshness_tau_seconds=10.0,
        clip_temperature=0.1,
    )


def _task(*, deadline: datetime | None = NOW + timedelta(seconds=30), task_id: str = "fire") -> TaskContext:
    return TaskContext(
        task_id=task_id,
        task_text="fire response" if task_id == "fire" else "flower catalog",
        concept_importance={"fire": 0.95, "flower": 0.10},
        deadline_at=deadline,
        urgency_horizon_seconds=60.0,
    )


def _pipeline() -> tuple[TransformerParameterPipeline, FakeEmbeddingBackend]:
    backend = FakeEmbeddingBackend()
    return TransformerParameterPipeline(_settings(), backend, NetworkXKnowledgeBase()), backend


def test_repeated_image_lowers_novelty_and_uses_one_embedding_per_observation(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, backend = _pipeline()

    first = pipeline.extract_from_path(image, _task(), NOW, captured_at=NOW - timedelta(seconds=5))
    second = pipeline.extract_from_path(image, _task(), NOW, captured_at=NOW - timedelta(seconds=5))

    assert first.get("novelty").value == 1.0
    assert second.get("novelty").value == 0.0
    assert backend.image_forward_calls == 2


def test_unrelated_image_raises_novelty_after_a_repeat(tmp_path: Path) -> None:
    repeated = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    unrelated = _write_image_marker(tmp_path, "other.jpeg", b"other-image")
    pipeline, _ = _pipeline()

    pipeline.extract_from_path(repeated, _task(), NOW, captured_at=NOW)
    repeat_bundle = pipeline.extract_from_path(repeated, _task(), NOW, captured_at=NOW)
    unrelated_bundle = pipeline.extract_from_path(unrelated, _task(), NOW, captured_at=NOW)

    assert unrelated_bundle.get("novelty").value > repeat_bundle.get("novelty").value


def test_changing_task_changes_relevance(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, _ = _pipeline()

    fire_bundle = pipeline.extract_from_path(image, _task(task_id="fire"), NOW, captured_at=NOW)
    flower_bundle = pipeline.extract_from_path(image, _task(task_id="flower"), NOW, captured_at=NOW)

    assert fire_bundle.get("relevance").value > flower_bundle.get("relevance").value


def test_older_timestamp_lowers_freshness(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, _ = _pipeline()

    recent = pipeline.extract_from_path(image, _task(), NOW, captured_at=NOW - timedelta(seconds=1))
    old = pipeline.extract_from_path(image, _task(), NOW, captured_at=NOW - timedelta(seconds=40))

    assert recent.get("freshness").value > old.get("freshness").value


def test_urgency_is_unavailable_without_deadline_or_context(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, _ = _pipeline()

    bundle = pipeline.extract_from_path(image, _task(deadline=None), NOW, captured_at=NOW)
    urgency = bundle.get("urgency")

    assert urgency.available is False
    assert urgency.value is None
    assert urgency.provenance["reason"] == "deadline_or_urgency_context_missing"


def test_external_oran_value_can_be_preserved_without_ue_fabrication(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, _ = _pipeline()
    oran_urgency = ParameterValue(
        value=0.8,
        raw_value={"policy_revision": "oran-42"},
        available=True,
        method="oran_deadline_policy",
        source="oran_context",
        version="oran/v1",
        provenance={"controller": "test"},
        units="normalized_score",
        producer_level=ParameterProducerLevel.ORAN,
    )
    task = replace(_task(deadline=None), provided_parameters={"urgency": oran_urgency})

    bundle = pipeline.extract_from_path(image, task, NOW, captured_at=NOW)

    assert bundle.get("urgency") == oran_urgency


def test_semantic_header_preserves_unavailability_without_fake_zero(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, _ = _pipeline()
    bundle = pipeline.extract_from_path(image, _task(deadline=None), NOW, captured_at=NOW)

    codec = SemanticHeaderCodec(_settings().catalog)
    packet = build_semantic_packet(bundle, codec=codec)
    decoded = {field.name: field for field in codec.decode(packet.header_bytes).fields}

    assert decoded["urgency"].available is False
    assert decoded["urgency"].value is None
    assert decoded["AoI_seconds"].value == 0.0


def test_packet_report_orders_semantic_header_before_payload_sections(tmp_path: Path) -> None:
    image = _write_image_marker(tmp_path, "same.jpeg", b"same-image")
    pipeline, _ = _pipeline()
    bundle = pipeline.extract_from_path(image, _task(deadline=None), NOW, captured_at=NOW)

    report = format_semantic_packet(build_semantic_packet(bundle))

    assert report.index("SEMANTIC HEADER") < report.index("NATIVE LATENT PAYLOAD")
    assert report.index("NATIVE LATENT PAYLOAD") < report.index("COMMUNICATION VQ PAYLOAD")


def test_clip_adapter_accepts_transformers_v5_pooler_output() -> None:
    class TensorLike:
        def detach(self):
            return self

    class ModelOutput:
        pooler_output = TensorLike()

    assert _feature_tensor(ModelOutput()) is ModelOutput.pooler_output
