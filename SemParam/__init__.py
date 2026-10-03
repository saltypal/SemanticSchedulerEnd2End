"""Semantic evidence APIs with lazy, identity-preserving public exports.

Cached latent scheduling does not load backbones or the earlier pipeline.
Existing public names keep their canonical class and function identities.
"""
from importlib import import_module

_EXPORTS = {
    "SEMANTIC_PARAMETER_ORDER": "SemParam.catalog",
    "SemanticPipelineSettings": "SemParam.catalog",
    "TransformerParameterPipeline": "SemParam.pipeline",
    "build_semantic_packet": "SemParam.headers.packet",
    "format_semantic_packet": "SemParam.headers.packet",
    "print_semantic_packet": "SemParam.headers.packet",
    "SemanticHeaderCodec": "SemParam.headers.semantic",
    "SemanticSchedulingHeaderV2": "SemParam.headers.semantic",
    "SemanticSchedulingHeaderV2Codec": "SemParam.headers.semantic",
    "FeatureValue": "SemParam.features.scheduler",
    "SchedulerAnnotation": "SemParam.features.scheduler",
    "SchedulerFeatureExtractor": "SemParam.features.scheduler",
    "SchedulingFeatureVectorV2": "SemParam.features.scheduler",
    "DeliveredKnowledgeSnapshot": "SemParam.history.delivered",
    "DeliveredKnowledgeState": "SemParam.history.delivered",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
