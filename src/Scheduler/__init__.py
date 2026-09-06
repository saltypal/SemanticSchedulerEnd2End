"""Future scheduling contracts; Stage 1A deliberately makes no semantic scheduling decision."""

from INFRA.errors import ComponentUnavailableError


class SemanticScheduler:
    def __init__(self) -> None:
        raise ComponentUnavailableError("SemanticScheduler", "Stage 1A", "Scheduling begins only after Stage 1B evidence contracts exist.")
