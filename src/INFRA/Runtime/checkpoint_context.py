"""Checkpoint metadata shared by training and manifest export."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CheckpointContext:
    epoch: int
    phase: str
    validation_loss: float
    source_commit: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
