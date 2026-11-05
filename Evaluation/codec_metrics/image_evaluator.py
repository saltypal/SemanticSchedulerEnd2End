"""Typed image evaluator registered for dependency-injected pipeline use."""

from Evaluation.Artifacts import EvaluationArtifact, ImageArtifact
from Evaluation.Interfaces import EvaluatorInterface
from Evaluation.Registries import EVALUATOR_REGISTRY
from Evaluation.codec_metrics.Image.psnr import psnr
from Evaluation.codec_metrics.Image.ssim import ssim


class ImageQualityEvaluator(EvaluatorInterface):
    def evaluate(self, reference: ImageArtifact, reconstruction: ImageArtifact) -> EvaluationArtifact:
        return EvaluationArtifact(
            metrics={"psnr_db": psnr(reference.tensor, reconstruction.tensor), "ssim": ssim(reference.tensor, reconstruction.tensor)},
            evaluator_name="image_quality",
            samples_evaluated=reference.batch_size(),
        )


def register_image_quality_evaluator() -> None:
    if not EVALUATOR_REGISTRY.contains("image_quality"):
        EVALUATOR_REGISTRY.register("image_quality", ImageQualityEvaluator)
