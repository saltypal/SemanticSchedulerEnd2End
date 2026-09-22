"""Named future stores that fail explicitly until their infrastructure exists."""

from __future__ import annotations

from INFRA.errors import ComponentUnavailableError


class _UnavailableKnowledgeBase:
    component_name = "knowledge base"

    def __init__(self) -> None:
        raise ComponentUnavailableError(
            self.component_name,
            "Stage 1B",
            "Only the local NetworkX demonstration knowledge base is implemented.",
        )


class VectorStoreKnowledgeBase(_UnavailableKnowledgeBase):
    component_name = "vector-store knowledge base"


class RedisKnowledgeBase(_UnavailableKnowledgeBase):
    component_name = "Redis knowledge base"


class GlobalMemoryKnowledgeBase(_UnavailableKnowledgeBase):
    component_name = "global/cross-UE knowledge base"


class RichKnowledgeGraph(_UnavailableKnowledgeBase):
    component_name = "rich knowledge graph"
