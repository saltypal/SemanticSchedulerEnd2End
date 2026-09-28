"""Lazy LADI image-classifier adapter for an image-only scheduler baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


LADI_REFERENCE_MODEL = "MITLL/LADI-v2-classifier-small-reference"


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
        self._processor = AutoImageProcessor.from_pretrained(self.model_name, local_files_only=self.local_files_only)
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
