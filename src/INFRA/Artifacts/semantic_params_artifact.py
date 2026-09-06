"""Current-input semantic evidence artifact for a future Stage 1B pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticParametersArtifact:
    values: dict[str, Any]
    extractor_name: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
