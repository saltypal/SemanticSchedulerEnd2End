"""Lazy LADI image-classifier adapter for an image-only scheduler baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math


LADI_REFERENCE_MODEL = "MITLL/LADI-v2-classifier-small-reference"

SEMANTIC_PRIORITY_BY_LABEL = {
    "flooding_structures": 0,
    "buildings_minor_or_greater": 1,
    "roads_damage": 1,
    "buildings_affected_or_greater": 1,
    "trees_damage": 2,
    "debris_any": 2,
    "flooding_any": 2,
    "bridges_any": 2,
    "buildings_any": 3,
    "roads_any": 3,
    "water_any": 3,
    "trees_any": 3,
}


def semantic_scene_features(probabilities: Mapping[str, float]) -> tuple[float, float]:
    """Return P1 scene importance and P4 interpretation confidence.

    P1 is the strongest classifier probability after weighting labels by the
    frozen task rubric. P4 is one minus mean normalized binary entropy across
    the multi-label outputs. Both are bounded model-derived proxies in [0, 1].
    """
    values = {str(label).lower(): min(1.0, max(0.0, float(score)))
              for label, score in probabilities.items()}
    if not values:
        raise ValueError("Classifier probabilities are empty")
    weighted = [values.get(label, 0.0) * ((4 - priority) / 4.0)
                for label, priority in SEMANTIC_PRIORITY_BY_LABEL.items()]
    scene_importance = max(weighted, default=0.0)
    entropies = []
    for probability in values.values():
        clipped = min(1.0 - 1e-7, max(1e-7, probability))
        entropy = -(clipped * math.log(clipped)
                    + (1.0 - clipped) * math.log(1.0 - clipped)) / math.log(2.0)
        entropies.append(entropy)
    confidence = 1.0 - sum(entropies) / len(entropies)
    return scene_importance, confidence


@dataclass(frozen=True, slots=True)
class LadiPrediction:
    probabilities: Mapping[str, float]
    predicted_stratum: str | None
    mapped_priority: int | None
    model_repository: str = LADI_REFERENCE_MODEL
    threshold: float = 0.5


def map_ladi_probabilities(probabilities: Mapping[str, float], threshold: float = 0.5) -> tuple[str | None, int | None]:
    """Map classifier outputs through the public prototype rubric.

    This is a policy mapping, never a claim that a native model score is a
    calibrated emergency priority probability.
    """

    scores = {str(name).lower(): float(value) for name, value in probabilities.items()}
    if scores.get("flooding_structures", 0.0) >= threshold:
        return "flooding_structures", 0
    if scores.get("buildings_affected_or_greater", 0.0) >= threshold:
        return "buildings_affected_or_greater", 1
    if scores.get("roads_damage", 0.0) >= threshold:
        return "roads_damage", 1
    if scores.get("debris_any", 0.0) >= threshold:
        return "debris_any", 2
    if scores.get("trees_damage", 0.0) >= threshold:
        return "trees_damage", 2
    return None, None


class LadiPredictionProvider:
    """Load the optional reference classifier only when image prediction is requested."""

    def __init__(self, model_name: str = LADI_REFERENCE_MODEL, *, device: str = "auto", local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as error:
            raise RuntimeError("LADI prediction requires optional torch and transformers dependencies.") from error
        device = "cuda" if self.requested_device == "auto" and torch.cuda.is_available() else self.requested_device
        if device == "auto":
            device = "cpu"
        # Pin the preprocessing path explicitly.  The checkpoint currently
        # ships a slow processor; leaving this unset would make a future
        # Transformers release silently switch to the fast implementation and
        # change feature values slightly.
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_name, local_files_only=self.local_files_only, use_fast=False)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_name, local_files_only=self.local_files_only).to(device).eval()
        self._torch = torch
        self._device = device

    def predict(self, image_path: str | Path, *, threshold: float = 0.5) -> LadiPrediction:
        self._ensure_loaded()
        from PIL import Image

        assert self._processor is not None and self._model is not None and self._torch is not None
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            values = self._torch.sigmoid(self._model(**inputs).logits[0]).cpu().tolist()
        probabilities = {self._model.config.id2label[index]: float(value) for index, value in enumerate(values)}
        stratum, priority = map_ladi_probabilities(probabilities, threshold=threshold)
        return LadiPrediction(probabilities, stratum, priority, self.model_name, threshold)
