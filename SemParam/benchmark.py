"""Small reproducible latency and memory measurements for semantic-path comparisons."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Callable


def _synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rss_bytes() -> int | None:
    try:
        import os
        import psutil
    except ImportError:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def benchmark_callable(operation: Callable[[], Any], warmup_runs: int = 1, measured_runs: int = 5) -> dict[str, Any]:
    """Measure cold/warm execution without claiming hardware-independent results."""

    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs must be positive.")
    rss_before = _rss_bytes()
    _synchronize_cuda()
    cold_start = time.perf_counter()
    operation()
    _synchronize_cuda()
    cold_latency_ms = (time.perf_counter() - cold_start) * 1000.0
    for _ in range(warmup_runs):
        operation()
    samples: list[float] = []
    for _ in range(measured_runs):
        _synchronize_cuda()
        start = time.perf_counter()
        operation()
        _synchronize_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
    result: dict[str, Any] = {
        "cold_latency_ms": cold_latency_ms,
        "warm_latency_median_ms": statistics.median(samples),
        "warm_latency_p95_ms": sorted(samples)[max(0, int(round(0.95 * (len(samples) - 1))))],
        "measured_runs": measured_runs,
        "cpu_rss_before_bytes": rss_before,
        "cpu_rss_after_bytes": _rss_bytes(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            result["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
            result["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return result


def artifact_size_bytes(path: str | Path) -> int:
    """Return file size for a codebook/model artifact, with explicit missing-file failure."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")
    return artifact.stat().st_size
