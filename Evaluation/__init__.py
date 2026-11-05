from project_compat import install as _install_compat
_install_compat()
"""Contracts, orchestration, registries, and runtime support for the project."""

from Evaluation.Core.pipeline_manager import PipelineManager

__all__ = ["PipelineManager"]
