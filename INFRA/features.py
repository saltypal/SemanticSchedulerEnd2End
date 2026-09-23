"""Explicit boundary for methods that need codec features."""
from dataclasses import dataclass
from typing import Any, Protocol
@dataclass(slots=True)
class FeatureCapture:
    representation: Any
    latent: Any = None
class FeatureProvider(Protocol):
    def capture(self, image: Any, **options: Any) -> FeatureCapture: ...
