"""Stage availability guard used by the CLI and notebooks."""

from INFRA.errors import ComponentUnavailableError


class StageController:
    AVAILABLE_STAGES = {"stage1a"}

    def require_available(self, stage: str) -> None:
        if stage.lower() not in self.AVAILABLE_STAGES:
            raise ComponentUnavailableError(
                component="Stage controller",
                stage=stage,
                reason="Only Stage 1A communication, codebook, and evaluation work is active.",
            )
