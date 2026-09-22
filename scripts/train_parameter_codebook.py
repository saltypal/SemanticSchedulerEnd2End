"""Train and verify the demo-only Stage 1C semantic parameter codebook.

This command mirrors the Stage 1C notebook while making the training run
reproducible from the project root. CLIP is used only to form offline teacher
targets. The saved codebook contains pooled SwinJSCC representations and never
loads CLIP at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from CodebookParam.semantic_codebook import CONTENT_PARAMETER_NAMES, SemanticParameterCodebook
from CodebookParam.swin_representation import capture_pre_modulation_representation
from EncDecPipeline.Models.SwinJSCC.model_loader import load_stage1_swinjscc
from EncDecPipeline.Preprocessing.image_loader import discover_images, file_sha256
from TransformerPipeline.KnowledgeBases import NetworkXKnowledgeBase
from TransformerPipeline.clip_backbone import CLIPBackbone
from TransformerPipeline.observation_builder import parse_semantic_timestamp
from TransformerPipeline.pipeline import TransformerParameterPipeline
from TransformerPipeline.runtime import load_semantic_image_artifact, settings_from_config, task_context_from_mapping
from utils.config_loader import load_yaml_config


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_training_images(training_root: Path, test_root: Path, minimum_count: int) -> tuple[list[Path], list[str]]:
    training_images = discover_images(training_root)
    if len(training_images) < minimum_count:
        raise RuntimeError(f"Need at least {minimum_count} distinct training images; found {len(training_images)}.")
    training_hashes = [file_sha256(path) for path in training_images]
    if len(set(training_hashes)) != len(training_hashes):
        raise RuntimeError("Training images are not unique by SHA-256 hash.")
    test_hashes = {file_sha256(path) for path in discover_images(test_root)}
    overlap = sorted(set(training_hashes).intersection(test_hashes))
    if overlap:
        raise RuntimeError(f"Train/test image leakage detected by SHA-256: {overlap}")
    return training_images, training_hashes


def train(config_path: Path, model_dir: Path, device: str) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    codebook_config = dict(config["parameter_codebook"])
    training_root = PROJECT_ROOT / str(codebook_config["training_image_root"])
    test_root = PROJECT_ROOT / str(config["demo"]["image_root"])
    artifact_path = PROJECT_ROOT / str(codebook_config["artifact_path"])
    training_images, training_hashes = _validate_training_images(
        training_root,
        test_root,
        int(codebook_config["minimum_distinct_images"]),
    )

    settings = settings_from_config(config)
    task = task_context_from_mapping(dict(config["demo"]["task"]))
    reference_now = parse_semantic_timestamp(config["demo"]["reference_now"])
    if reference_now is None:
        raise RuntimeError("The codebook training configuration requires demo.reference_now.")
    active_device = _resolve_device(device)
    clip_config = dict(config["semantic"]["clip"])
    teacher = TransformerParameterPipeline(
        settings=settings,
        embedding_backend=CLIPBackbone(
            model_name=str(clip_config["model_name"]),
            device=active_device,
            local_files_only=bool(clip_config.get("local_files_only", False)),
        ),
        knowledge_base=NetworkXKnowledgeBase(),
    )
    adapter = load_stage1_swinjscc(model_dir, device=active_device)
    encoder, decoder = adapter._require_built()
    encoder.eval()
    decoder.eval()

    run_context = {
        "config": str(config_path),
        "model_dir": str(model_dir),
        "device": active_device,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "training_dataset": dict(codebook_config.get("training_dataset", {})),
        "training_image_root": str(training_root),
        "training_image_count": len(training_images),
        "training_image_hashes": training_hashes,
        "demo_only": bool(codebook_config["demo_only"]),
    }
    representations: list[Any] = []
    target_rows: list[dict[str, float]] = []
    for image_path in training_images:
        bundle = teacher.extract_from_path(
            image_path,
            task,
            reference_now,
            captured_at=reference_now,
            metadata={"parameter_codebook_training": True, "dataset": "Kodak"},
        )
        targets: dict[str, float] = {}
        for parameter_name in CONTENT_PARAMETER_NAMES:
            parameter = bundle.get(parameter_name)
            if not parameter.available or parameter.value is None:
                raise RuntimeError(f"Offline teacher did not produce {parameter_name} for {image_path}.")
            targets[parameter_name] = parameter.value
        image_artifact = load_semantic_image_artifact(image_path)
        capture = capture_pre_modulation_representation(
            adapter,
            image_artifact,
            snr_db=float(config["semantic"]["snr_db"]),
            rate=int(config["semantic"]["rate"]),
        )
        representations.append(capture.representation.detach().cpu().numpy())
        target_rows.append(targets)
        print(json.dumps({"image": image_path.name, "targets": targets, "representation_shape": list(capture.representation.shape)}))

    codebook = SemanticParameterCodebook.train(
        representations,
        target_rows,
        task_profile_id=task.task_id,
        training_image_hashes=training_hashes,
        random_state=int(codebook_config["random_state"]),
        codebook_size=int(codebook_config["codebook_size"]),
        demo_only=bool(codebook_config["demo_only"]),
        metadata={"run_context": run_context},
    )
    saved_path = codebook.save(artifact_path)

    reloaded = SemanticParameterCodebook.load(saved_path)
    prediction = reloaded.predict(representations[0], task_profile_id=task.task_id)
    if tuple(prediction.values) != CONTENT_PARAMETER_NAMES:
        raise RuntimeError("Reloaded semantic codebook returned an unexpected parameter schema.")
    if reloaded.content_hash() != codebook.content_hash():
        raise RuntimeError("Reloaded semantic codebook content hash differs from the trained artifact.")

    summary_path = PROJECT_ROOT / "results" / "stage1c_parameter_codebook_demo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        **run_context,
        "artifact_path": str(saved_path),
        "artifact_sha256": _sha256_file(saved_path),
        "artifact_content_hash": codebook.content_hash(),
        "target_fields": list(CONTENT_PARAMETER_NAMES),
        "verification_prediction": {
            "codeword_id": prediction.codeword_id,
            "values": prediction.values,
            "squared_distance": prediction.squared_distance,
        },
        "warning": "demo_only: Kodak teacher pairs provide an engineering demonstration, not a generalization result.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"artifact": str(saved_path), "summary": str(summary_path), "content_hash": codebook.content_hash()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "stage1c_parameter_codebook.yaml")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "SwinJSCC")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.config.resolve(), arguments.model_dir.resolve(), arguments.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
