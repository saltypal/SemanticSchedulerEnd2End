"""Dependency-injected Stage 1A execution flow."""

from dataclasses import dataclass

from INFRA.Artifacts import EvaluationArtifact, ImageArtifact
from INFRA.Interfaces import ChannelInterface, EncDecInterface, EvaluatorInterface


@dataclass(slots=True)
class PipelineRun:
    reconstruction: ImageArtifact
    evaluations: list[EvaluationArtifact]


class PipelineManager:
    """Runs encode -> channel -> decode -> evaluation without importing model internals."""

    def __init__(
        self,
        encdec: EncDecInterface,
        channel: ChannelInterface,
        evaluators: list[EvaluatorInterface] | None = None,
    ) -> None:
        self.encdec = encdec
        self.channel = channel
        self.evaluators = evaluators or []

    def run_image(self, image: ImageArtifact, snr_db: float, **options: object) -> PipelineRun:
        latent = self.encdec.encode(image, **options)
        transmission = self.channel.transmit(latent, snr_db, **options)
        reconstruction = self.encdec.reconstruct_received(transmission, latent)
        evaluations = [evaluator.evaluate(image, reconstruction) for evaluator in self.evaluators]
        return PipelineRun(reconstruction=reconstruction, evaluations=evaluations)
