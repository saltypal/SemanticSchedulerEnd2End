"""Stable location and schema for aggregate metrics."""

import csv
from pathlib import Path
from typing import Any


class ResultManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append_metrics(self, filename: str, row: dict[str, Any]) -> Path:
        target = self.output_dir / filename
        new_file = not target.exists()
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        return target
