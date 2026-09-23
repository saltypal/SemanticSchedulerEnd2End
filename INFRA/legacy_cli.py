"""Entry point for Stage 1A validation and modular Stage 1B semantic demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from INFRA.Channels.channel_factory import register_builtin_channels
from SemParam.Codebooks.semantic_codebook import register_semantic_parameter_codebook
from EncDec.Codebooks.minibatch_kmeans_vq import register_minibatch_kmeans_vq
from EncDec.Models.SwinJSCC.adapter import register_swinjscc
from EncDec.Models.SwinJSCC.checkpoint_manager import Stage1ArtifactManifest
from EncDec.Evaluation.image_evaluator import register_image_quality_evaluator
from INFRA.Core.stage_controller import StageController
from INFRA.Registries import (
    CHANNEL_REGISTRY,
    CODEBOOK_REGISTRY,
    ENCDEC_REGISTRY,
    EVALUATOR_REGISTRY,
    KNOWLEDGE_REGISTRY,
    PARAMETER_CODEBOOK_REGISTRY,
)
from SemParam.KnowledgeBases import register_networkx_knowledge_base
from SemParam.catalog import SEMANTIC_PARAMETER_ORDER
from SemParam.runtime import run_semantic_demo, settings_from_config
from INFRA.utils.config_loader import load_yaml_config


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


def register_stage1b_components() -> None:
    register_networkx_knowledge_base()
    register_semantic_parameter_codebook()


def validate_stage1b_skeleton(config: dict[str, object]) -> dict[str, object]:
    StageController().require_available(str(config.get("stage", "stage1b")))
    register_stage1b_components()
    settings = settings_from_config(config)
    if tuple(definition.name for definition in settings.catalog) != tuple(settings.enabled_parameters):
        raise RuntimeError("Stage 1B semantic parameter catalog differs from its configured order.")
    return {
        "stage": "stage1b",
        "status": "semantic_contract_validated",
        "parameters": list(settings.enabled_parameters),
        "knowledge_bases": KNOWLEDGE_REGISTRY.names(),
        "parameter_codebooks": PARAMETER_CODEBOOK_REGISTRY.names(),
        "scheduler": "not_implemented_by_design",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--action", choices=("validate", "validate-manifest", "semantic-validate", "semantic-run"), default="validate")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--param-extractor", choices=("transformer", "codebook"), default="transformer")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--communication-codebook", type=Path)
    parser.add_argument("--semantic-codebook", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--now", help="Timezone-aware ISO-8601 reference time for a reproducible semantic demo.")
    arguments = parser.parse_args()
    default_config = Path("config/stage1b.yaml") if arguments.action.startswith("semantic-") else Path("config/stage1a.yaml")
    config = load_yaml_config(arguments.config or default_config)
    if arguments.action == "validate":
        print(json.dumps(validate_stage1a_skeleton(config), indent=2, default=list))
        return
    if arguments.action == "semantic-validate":
        print(json.dumps(validate_stage1b_skeleton(config), indent=2, default=list))
        return
    if arguments.action == "semantic-run":
        if arguments.model_dir is None:
            parser.error("--model-dir is required for --action semantic-run")
        validate_stage1b_skeleton(config)
        result = run_semantic_demo(
            config,
            param_extractor=arguments.param_extractor,
            model_dir=arguments.model_dir,
            input_dir=arguments.input_dir,
            communication_codebook_path=arguments.communication_codebook,
            semantic_codebook_path=arguments.semantic_codebook,
            device=arguments.device,
            output_plot=arguments.output_plot,
            now_override=arguments.now,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if arguments.model_dir is None:
        parser.error("--model-dir is required for --action validate-manifest")
    manifest = Stage1ArtifactManifest.load_and_validate(arguments.model_dir)
    print(json.dumps(manifest.raw, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
