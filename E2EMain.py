"""Entry point for contract checks and manifest-validated Stage 1A inference setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Channels.channel_factory import register_builtin_channels
from EncDecPipeline.Codebooks.minibatch_kmeans_vq import register_minibatch_kmeans_vq
from EncDecPipeline.Models.SwinJSCC.adapter import register_swinjscc
from EncDecPipeline.Models.SwinJSCC.checkpoint_manager import Stage1ArtifactManifest
from Evaluation.image_evaluator import register_image_quality_evaluator
from INFRA.Core.stage_controller import StageController
from INFRA.Registries import CHANNEL_REGISTRY, CODEBOOK_REGISTRY, ENCDEC_REGISTRY, EVALUATOR_REGISTRY
from utils.config_loader import load_yaml_config


def register_stage1a_components() -> None:
    register_builtin_channels()
    register_minibatch_kmeans_vq()
    register_swinjscc()
    register_image_quality_evaluator()


def validate_stage1a_skeleton(config: dict[str, object]) -> dict[str, object]:
    StageController().require_available(str(config.get("stage", "stage1a")))
    register_stage1a_components()
    expected = {
        "encdec": ("swinjscc",),
        "channels": ("awgn", "rayleigh_perfect_csi", "rician"),
        "codebooks": ("minibatch_kmeans_vq_k256",),
        "evaluators": ("image_quality",),
    }
    actual = {
        "encdec": ENCDEC_REGISTRY.names(),
        "channels": CHANNEL_REGISTRY.names(),
        "codebooks": CODEBOOK_REGISTRY.names(),
        "evaluators": EVALUATOR_REGISTRY.names(),
    }
    for category, names in expected.items():
        absent = set(names).difference(actual[category])
        if absent:
            raise RuntimeError(f"Stage 1A registry validation failed for {category}: {sorted(absent)}")
    return {"stage": "stage1a", "registry": actual, "status": "skeleton_validated"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stage1a.yaml"))
    parser.add_argument("--action", choices=("validate", "validate-manifest"), default="validate")
    parser.add_argument("--model-dir", type=Path)
    arguments = parser.parse_args()
    config = load_yaml_config(arguments.config)
    if arguments.action == "validate":
        print(json.dumps(validate_stage1a_skeleton(config), indent=2, default=list))
        return
    if arguments.model_dir is None:
        parser.error("--model-dir is required for --action validate-manifest")
    manifest = Stage1ArtifactManifest.load_and_validate(arguments.model_dir)
    print(json.dumps(manifest.raw, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
