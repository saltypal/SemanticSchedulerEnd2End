"""Declarative descriptions of a pipeline execution."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    encdec_name: str
    channel_name: str
    evaluator_names: tuple[str, ...]
    codebook_name: str | None = None
    scheduler_name: str | None = None
