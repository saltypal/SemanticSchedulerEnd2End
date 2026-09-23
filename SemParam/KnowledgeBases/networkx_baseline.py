"""Small, inspectable NetworkX knowledge base for sequential Stage 1B demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from INFRA.Artifacts import KnowledgeArtifact, ObservationArtifact, SemanticParameterBundle
from INFRA.Interfaces import KnowledgeInterface


def _normalized(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Knowledge-base descriptors must have a non-zero norm.")
    return values / norm


class NetworkXKnowledgeBase(KnowledgeInterface):
    """Local graph plus descriptor history; deliberately not a distributed store."""

    name = "networkx_local"

    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError as error:
            raise RuntimeError("NetworkXKnowledgeBase requires the optional stage1b dependencies.") from error
        self._nx = nx
        self.graph = nx.Graph()
        self._descriptors: dict[str, np.ndarray] = {}

    def query(self, observation: ObservationArtifact) -> KnowledgeArtifact:
        descriptor = _normalized(observation.image_embedding)
        if not self._descriptors:
            evidence = {
                "history_size": 0,
                "max_cosine_similarity": None,
                "nearest_observation_id": None,
                "novelty_reference": "empty_history",
            }
        else:
            similarities = {
                identifier: float(np.clip(np.dot(descriptor, previous), -1.0, 1.0))
                for identifier, previous in self._descriptors.items()
            }
            nearest_observation_id, max_similarity = max(similarities.items(), key=lambda item: item[1])
            evidence = {
                "history_size": len(self._descriptors),
                "max_cosine_similarity": max_similarity,
                "nearest_observation_id": nearest_observation_id,
                "novelty_reference": "max_cosine_similarity",
            }
        return KnowledgeArtifact(
            evidence=evidence,
            knowledge_source=self.name,
            metadata={"graph_nodes": self.graph.number_of_nodes(), "graph_edges": self.graph.number_of_edges()},
        )

    def record(self, observation: ObservationArtifact, parameters: SemanticParameterBundle) -> None:
        descriptor = _normalized(observation.image_embedding)
        self._descriptors[observation.observation_id] = descriptor
        self.graph.add_node(
            observation.observation_id,
            node_type="image_observation",
            image_hash=observation.image_hash,
            source_path=observation.source_path,
            parameter_schema=parameters.schema_version,
        )
        for concept_name, score in observation.concept_scores.items():
            concept_node = f"concept:{concept_name}"
            self.graph.add_node(concept_node, node_type="concept")
            self.graph.add_edge(
                observation.observation_id,
                concept_node,
                relation="semantic_evidence",
                confidence=float(score),
            )

    @property
    def observation_count(self) -> int:
        return len(self._descriptors)
