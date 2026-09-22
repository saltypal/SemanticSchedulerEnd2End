"""CLI-facing Stage 1B orchestration; scheduler policy remains out of scope."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from CodebookParam.pipeline import CodebookParameterPipeline
from CodebookParam.semantic_codebook import SemanticParameterCodebook
from EncDecPipeline.Codebooks.minibatch_kmeans_vq import MiniBatchKMeansVQ
from EncDecPipeline.Models.SwinJSCC.model_loader import load_stage1_swinjscc
from EncDecPipeline.Preprocessing.image_loader import discover_images, pil_to_tensor, require_torch_and_pillow
from EncDecPipeline.Preprocessing.resize import center_crop_to_multiple
from INFRA.Artifacts import ImageArtifact, TaskContext
from TransformerPipeline.KnowledgeBases import NetworkXKnowledgeBase
from TransformerPipeline.catalog import SEMANTIC_PARAMETER_ORDER, SemanticPipelineSettings
from TransformerPipeline.clip_backbone import CLIPBackbone
from TransformerPipeline.observation_builder import parse_semantic_timestamp
from TransformerPipeline.packet_builder import build_semantic_packet, print_semantic_packet
from TransformerPipeline.pipeline import TransformerParameterPipeline
from TransformerPipeline.semantic_header import SemanticHeaderCodec
from TransformerPipeline.showcase import ShowcaseRecord, plot_semantic_showcase


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Stage 1B configuration requires mapping: {name}")
    return dict(value)


def task_context_from_mapping(values: dict[str, Any]) -> TaskContext:
    """Create an explicit task policy from a YAML-compatible mapping."""

    return TaskContext(
        task_id=str(values["task_id"]),
        task_text=str(values["task_text"]),
        concept_importance={str(name): float(weight) for name, weight in dict(values.get("concept_importance", {})).items()},
        deadline_at=parse_semantic_timestamp(values.get("deadline_at")),
        urgency_horizon_seconds=float(values.get("urgency_horizon_seconds", 60.0)),
        context_text=str(values["context_text"]) if values.get("context_text") else None,
        metadata=dict(values.get("metadata", {})),
    )


def settings_from_config(config: dict[str, Any]) -> SemanticPipelineSettings:
    semantic = _as_mapping(config.get("semantic"), "semantic")
    enabled = tuple(str(name) for name in semantic.get("enabled_parameters", SEMANTIC_PARAMETER_ORDER))
    return SemanticPipelineSettings(
        concept_prompts={str(name): str(prompt) for name, prompt in dict(semantic["concept_prompts"]).items()},
        enabled_parameters=enabled,
        freshness_tau_seconds=float(semantic.get("freshness_tau_seconds", 10.0)),
        clip_temperature=float(semantic.get("clip_temperature", 0.07)),
        version=str(semantic.get("version", "stage1b-transformer/v1")),
        metadata={"config_stage": config.get("stage")},
    )


def load_semantic_image_artifact(path: Path) -> ImageArtifact:
    torch, Image = require_torch_and_pillow()
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        maximum_edge = 1024
        width, height = image.size
        largest_edge = max(width, height)
        if largest_edge > maximum_edge:
            resize_scale = maximum_edge / largest_edge
            resized_width = max(128, int(width * resize_scale))
            resized_height = max(128, int(height * resize_scale))
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        # The encoder downsamples four times and uses an 8x8 Swin window in
        # its deepest stage. An input multiple of 16 satisfies the convolution
        # hierarchy but can still leave a final feature grid that cannot be
        # partitioned into windows. Multiples of 16 * 8 = 128 satisfy both.
        image = center_crop_to_multiple(image, divisor=128)
        tensor = pil_to_tensor(image).unsqueeze(0)
    return ImageArtifact(
        tensor=tensor,
        sample_ids=[path.name],
        source="stage1b_semantic_runtime",
        metadata={"source_path": str(path)},
    )


def _ordered_demo_entries(config: dict[str, Any], input_dir: Path | None) -> list[dict[str, Any]]:
    demo = _as_mapping(config.get("demo"), "demo")
    configured = demo.get("images")
    base = input_dir or Path(str(demo.get("image_root", "Test/images")))
    if configured is None:
        return [{"filename": path.name} for path in discover_images(base)]
    if not isinstance(configured, list):
        raise ValueError("demo.images must be a list.")
    entries: list[dict[str, Any]] = []
    for item in configured:
        entry = _as_mapping(item, "demo.images item")
        filename = entry.get("filename")
        if not filename:
            raise ValueError("Each demo image entry requires filename.")
        path = base / str(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Configured Stage 1B demo image is absent: {path}")
        entries.append(entry)
    return entries


def _reference_now(config: dict[str, Any], override: str | None) -> datetime:
    demo = _as_mapping(config.get("demo"), "demo")
    value = override or demo.get("reference_now")
    parsed = parse_semantic_timestamp(value)
    if parsed is None:
        raise ValueError("Stage 1B demo requires --now or demo.reference_now.")
    return parsed


def _create_vq(path: Path | None) -> MiniBatchKMeansVQ | None:
    if path is None:
        return None
    return MiniBatchKMeansVQ.load(path)


def run_semantic_demo(
    config: dict[str, Any],
    *,
    param_extractor: str,
    model_dir: Path,
    input_dir: Path | None = None,
    communication_codebook_path: Path | None = None,
    semantic_codebook_path: Path | None = None,
    device: str = "auto",
    output_plot: Path | None = None,
    now_override: str | None = None,
) -> dict[str, Any]:
    """Run the configured image series and print header then native/VQ payload summaries."""

    if param_extractor not in {"transformer", "codebook"}:
        raise ValueError(f"Unknown parameter extractor: {param_extractor}")
    settings = settings_from_config(config)
    demo = _as_mapping(config.get("demo"), "demo")
    base_task = task_context_from_mapping(_as_mapping(demo.get("task"), "demo.task"))
    reference_now = _reference_now(config, now_override)
    active_device = _resolve_device(device)
    adapter = load_stage1_swinjscc(model_dir, device=active_device)
    encoder, decoder = adapter._require_built()
    encoder.eval()
    decoder.eval()
    semantic = _as_mapping(config.get("semantic"), "semantic")
    snr_db = float(semantic.get("snr_db", 10.0))
    rate = int(semantic.get("rate", 96))
    vq = _create_vq(communication_codebook_path)
    knowledge_base = NetworkXKnowledgeBase()
    codec = SemanticHeaderCodec(settings.catalog)
    if param_extractor == "transformer":
        clip = _as_mapping(semantic.get("clip", {}), "semantic.clip")
        pipeline: Any = TransformerParameterPipeline(
            settings=settings,
            embedding_backend=CLIPBackbone(
                model_name=str(clip.get("model_name", "openai/clip-vit-base-patch32")),
                device=active_device,
                local_files_only=bool(clip.get("local_files_only", False)),
            ),
            knowledge_base=knowledge_base,
        )
    else:
        if semantic_codebook_path is None:
            raise ValueError("--semantic-codebook is required for --param-extractor codebook.")
        pipeline = CodebookParameterPipeline(
            settings=settings,
            codebook=SemanticParameterCodebook.load(semantic_codebook_path),
            knowledge_base=knowledge_base,
        )
    records: list[ShowcaseRecord] = []
    root = input_dir or Path(str(demo.get("image_root", "Test/images")))
    for entry in _ordered_demo_entries(config, input_dir):
        path = root / str(entry["filename"])
        image = load_semantic_image_artifact(path)
        deadline = parse_semantic_timestamp(entry.get("deadline_at"))
        task = replace(base_task, deadline_at=deadline if deadline is not None else base_task.deadline_at)
        captured_at = parse_semantic_timestamp(entry.get("captured_at"))
        image_metadata = {"demo_manifest": True, "label": entry.get("label", path.stem)}
        if param_extractor == "transformer":
            bundle = pipeline.extract_from_path(
                path,
                task,
                reference_now,
                captured_at=captured_at,
                metadata=image_metadata,
            )
            latent = adapter.encode(image, snr_db=snr_db, rate=rate)
        else:
            result = pipeline.extract_with_latent(
                adapter,
                image,
                path,
                task,
                reference_now,
                snr_db=snr_db,
                rate=rate,
                captured_at=captured_at,
                metadata=image_metadata,
            )
            bundle = result.bundle
            latent = result.latent
        vq_packet = vq.encode(latent) if vq is not None else None
        packet = build_semantic_packet(
            bundle,
            codec=codec,
            native_latent=latent,
            communication_vq_packet=vq_packet,
        )
        print_semantic_packet(packet, codec=codec)
        records.append(ShowcaseRecord(path, bundle, packet, str(entry.get("label", path.stem))))
    if output_plot is not None:
        plot_semantic_showcase(records, output_plot)
    return {
        "status": "semantic_demo_complete",
        "param_extractor": param_extractor,
        "image_count": len(records),
        "knowledge_observation_count": knowledge_base.observation_count,
        "semantic_header_bytes": [len(record.packet.header_bytes) for record in records],
        "parameter_bundles": [record.bundle.to_dict() for record in records],
        "output_plot": str(output_plot) if output_plot is not None else None,
        "note": "semantic header is a logical sidecar and is not included in Stage 1A VQ packet accounting",
    }
