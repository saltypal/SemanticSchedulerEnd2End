"""Future semantic codebook contracts; distinct from Stage 1A latent VQ."""

from INFRA.errors import ComponentUnavailableError


class SemanticCodebook:
    def __init__(self) -> None:
        raise ComponentUnavailableError("SemanticCodebook", "Stage 1A", "Semantic prototype codebooks are a future stage.")
