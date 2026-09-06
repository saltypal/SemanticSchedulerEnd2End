"""Typed data exchanged between independent research components."""

from INFRA.Artifacts.evaluation_artifact import EvaluationArtifact
from INFRA.Artifacts.image_artifact import ImageArtifact
from INFRA.Artifacts.knowledge_artifact import KnowledgeArtifact
from INFRA.Artifacts.latent_artifact import LatentArtifact
from INFRA.Artifacts.scheduling_artifact import SchedulingArtifact
from INFRA.Artifacts.semantic_params_artifact import SemanticParametersArtifact
from INFRA.Artifacts.transmission_artifact import TransmissionArtifact

__all__ = [
    "EvaluationArtifact",
    "ImageArtifact",
    "KnowledgeArtifact",
    "LatentArtifact",
    "SchedulingArtifact",
    "SemanticParametersArtifact",
    "TransmissionArtifact",
]
