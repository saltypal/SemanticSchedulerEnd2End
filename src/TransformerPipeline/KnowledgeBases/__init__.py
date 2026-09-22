"""Historical/context evidence stores; separate from current-input extraction."""

from INFRA.Registries import KNOWLEDGE_REGISTRY
from TransformerPipeline.KnowledgeBases.networkx_baseline import NetworkXKnowledgeBase
from TransformerPipeline.KnowledgeBases.scalable_placeholders import (
    GlobalMemoryKnowledgeBase,
    RedisKnowledgeBase,
    RichKnowledgeGraph,
    VectorStoreKnowledgeBase,
)


def register_networkx_knowledge_base() -> None:
    if not KNOWLEDGE_REGISTRY.contains("networkx_local"):
        KNOWLEDGE_REGISTRY.register("networkx_local", NetworkXKnowledgeBase)


# Preserve the original public name while making Stage 1B genuinely usable.
Stage1BKnowledgeBase = NetworkXKnowledgeBase

__all__ = [
    "GlobalMemoryKnowledgeBase",
    "NetworkXKnowledgeBase",
    "RedisKnowledgeBase",
    "RichKnowledgeGraph",
    "Stage1BKnowledgeBase",
    "VectorStoreKnowledgeBase",
    "register_networkx_knowledge_base",
]
