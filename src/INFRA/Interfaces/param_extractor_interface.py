"""Contract for current-input semantic parameter extraction."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import ImageArtifact, SemanticParametersArtifact


class ParameterExtractorInterface(ABC):
    @abstractmethod
    def extract(self, image: ImageArtifact) -> SemanticParametersArtifact:
        raise NotImplementedError
