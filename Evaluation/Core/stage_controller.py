"""Stage availability guard used by the CLI and notebooks."""

from Evaluation.errors import ComponentUnavailableError


class StageController:
    AVAILABLE_STAGES = {"stage1a", "stage1b"}

    def require_available(self, stage: str) -> None:
        if stage.lower() not in self.AVAILABLE_STAGES:
            raise ComponentUnavailableError(
                component="Stage controller",
                stage=stage,
                reason="Only Stage 1A communication and Stage 1B semantic-header work are active.",
            )
