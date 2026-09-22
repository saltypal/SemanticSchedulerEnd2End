"""Semantic parameter codebooks; intentionally separate from Stage 1A latent VQ."""

from CodebookParam.semantic_codebook import (
    CONTENT_PARAMETER_NAMES,
    SemanticCodebookPrediction,
    SemanticParameterCodebook,
    pool_swin_representation,
    register_semantic_parameter_codebook,
)


SemanticCodebook = SemanticParameterCodebook


def __getattr__(name: str):
    """Avoid importing the runtime pipeline while its semantic-codebook module loads."""

    if name == "CodebookParameterPipeline":
        from CodebookParam.pipeline import CodebookParameterPipeline

        return CodebookParameterPipeline
    raise AttributeError(name)

__all__ = [
    "CONTENT_PARAMETER_NAMES",
    "CodebookParameterPipeline",
    "SemanticCodebook",
    "SemanticCodebookPrediction",
    "SemanticParameterCodebook",
    "pool_swin_representation",
    "register_semantic_parameter_codebook",
]
