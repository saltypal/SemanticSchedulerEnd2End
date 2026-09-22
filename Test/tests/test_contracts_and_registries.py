import pytest

from Channels.channel_factory import create_channel, register_builtin_channels
from EncDecPipeline.Codebooks.minibatch_kmeans_vq import register_minibatch_kmeans_vq
from EncDecPipeline.Models.SwinJSCC.adapter import register_swinjscc
from Evaluation.image_evaluator import register_image_quality_evaluator
from INFRA.Core.stage_controller import StageController
from INFRA.Registries import CHANNEL_REGISTRY, CODEBOOK_REGISTRY, ENCDEC_REGISTRY, EVALUATOR_REGISTRY
from TransformerPipeline.KnowledgeBases import Stage1BKnowledgeBase


def test_stage1a_concrete_components_are_registered_without_importing_torch() -> None:
    register_builtin_channels()
    register_minibatch_kmeans_vq()
    register_swinjscc()
    register_image_quality_evaluator()
    assert "awgn" in CHANNEL_REGISTRY.names()
    assert "swinjscc" in ENCDEC_REGISTRY.names()
    assert "minibatch_kmeans_vq_k256" in CODEBOOK_REGISTRY.names()
    assert "image_quality" in EVALUATOR_REGISTRY.names()
    assert create_channel("awgn").name == "awgn"


def test_stage1b_networkx_baseline_is_available_without_touching_stage1a_registries() -> None:
    pytest.importorskip("networkx")
    knowledge_base = Stage1BKnowledgeBase()
    assert knowledge_base.name == "networkx_local"
    StageController().require_available("stage1b")
