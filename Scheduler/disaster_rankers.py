"""Lightweight policy scoring; no neural framework, image or codec imports."""
from __future__ import annotations

from pathlib import Path
import numpy as np


class ArrayScorer:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as saved:
            self.weights = {key: saved[key] for key in saved.files}

    def __call__(self, rows):
        values = (np.asarray(rows) - self.weights["mean"]) / self.weights["scale"]
        layers = int(self.weights["layers"])
        for layer in range(layers):
            values = values @ self.weights[f"w{layer}"].T + self.weights[f"b{layer}"]
            if layer < layers - 1:
                values = np.maximum(values, 0)
        if "classes" in self.weights:
            values = np.exp(values - values.max(axis=1, keepdims=True))
            values /= values.sum(axis=1, keepdims=True)
            return values @ self.weights["classes"]
        return np.asarray(values).reshape(-1)


def candidate_first_safe_mask(service_ms, deadlines_ms, now_ms: float):
    """Only a deterministic current-ready-set timing statement, not admission.

    Returns individually feasible jobs in overload; caller must report overload.
    """
    costs, deadlines = np.asarray(service_ms, float), np.asarray(deadlines_ms, float)
    if costs.shape != deadlines.shape or costs.ndim != 1 or np.any(costs <= 0):
        raise ValueError("Invalid service/deadline vectors")
    if not np.isfinite(costs).all() or np.isnan(deadlines).any() or not np.isfinite(now_ms):
        raise ValueError("Nonfinite timing input")
    feasible = now_ms + costs <= deadlines
    order = np.argsort(deadlines, kind="stable")
    if not np.all(now_ms + np.cumsum(costs[order]) <= deadlines[order]):
        return feasible, True
    allowed = np.zeros(len(costs), dtype=bool)
    prefix, smallest_slack = 0., float("inf")
    for index in order:
        allowed[index] = costs[index] <= smallest_slack + 1e-9 and feasible[index]
        prefix += costs[index]
        smallest_slack = min(smallest_slack, deadlines[index] - now_ms - prefix)
    return allowed, False
