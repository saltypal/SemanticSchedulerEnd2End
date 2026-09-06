"""Per-run state shared by orchestration, never by concrete model modules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeContext:
    run_id: str
    output_dir: Path
    config: dict[str, Any]
    device: str
    metadata: dict[str, Any] = field(default_factory=dict)
