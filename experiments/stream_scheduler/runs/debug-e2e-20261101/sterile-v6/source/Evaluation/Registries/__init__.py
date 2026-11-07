"""Named component registries. Registration happens in concrete package modules."""

from Evaluation.Registries.channel_registry import CHANNEL_REGISTRY
from Evaluation.Registries.codebook_registry import CODEBOOK_REGISTRY
from Evaluation.Registries.encdec_registry import ENCDEC_REGISTRY
from Evaluation.Registries.evaluator_registry import EVALUATOR_REGISTRY
from Evaluation.Registries.knowledge_registry import KNOWLEDGE_REGISTRY
from Evaluation.Registries.parameter_codebook_registry import PARAMETER_CODEBOOK_REGISTRY
from Evaluation.Registries.scheduler_registry import SCHEDULER_REGISTRY
from Evaluation.Registries.transformer_registry import TRANSFORMER_REGISTRY

__all__ = [
    "CHANNEL_REGISTRY",
    "CODEBOOK_REGISTRY",
    "ENCDEC_REGISTRY",
    "EVALUATOR_REGISTRY",
    "KNOWLEDGE_REGISTRY",
    "PARAMETER_CODEBOOK_REGISTRY",
    "SCHEDULER_REGISTRY",
    "TRANSFORMER_REGISTRY",
]
