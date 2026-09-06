"""Contract for historical/context evidence stores."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import ImageArtifact, KnowledgeArtifact


class KnowledgeInterface(ABC):
    @abstractmethod
    def query(self, image: ImageArtifact) -> KnowledgeArtifact:
        raise NotImplementedError
