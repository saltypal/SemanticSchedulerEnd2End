"""Image evidence adapters for the scheduler pilot.

Models never see the human response-priority label. Their native outputs remain
available beside the explicitly mapped, provisional scheduling priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    kind: str
    role: str


MODEL_SPECS = {
    "clip": ModelSpec("openai/clip-vit-base-patch32", "zero_shot", "scene classifier"),
    "siglip": ModelSpec("google/siglip-base-patch16-224", "zero_shot", "scene classifier"),
    "ladi_small": ModelSpec("MITLL/LADI-v2-classifier-small-reference", "ladi", "aerial multi-label classifier"),
    "ladi_large": ModelSpec("MITLL/LADI-v2-classifier-large-reference", "ladi", "aerial multi-label classifier"),
    "deit_tiny": ModelSpec("facebook/deit-tiny-patch16-224", "imagenet", "generic ImageNet control"),
    "grounding_dino": ModelSpec("IDEA-Research/grounding-dino-tiny", "grounding", "prompted object detector"),
}

# These are scene descriptions, not validated response priorities. Edit the
# mapping only after writing a mission-specific human labeling rubric.
SCENE_PROMPTS = {
    "severe_fire": "an aerial photograph of a large active fire engulfing buildings",
    "fire_smoke": "an aerial photograph of an active fire or thick smoke",
    "flooded_structures": "an aerial photograph of buildings surrounded by flood water",
    "flood": "an aerial photograph of a flooded area",
    "collapsed_building": "an aerial photograph of a collapsed or destroyed building",
    "damaged_road": "an aerial photograph of a road damaged by a disaster",
    "traffic_accident": "an aerial photograph of a traffic collision",
    "normal": "an aerial photograph of an intact ordinary landscape",
}

# 0 is highest priority, matching INFRA.contracts.Job.
SCENE_TO_PRIORITY = {
    "severe_fire": 0,
    "flooded_structures": 0,
    "fire_smoke": 1,
    "collapsed_building": 1,
    "traffic_accident": 1,
    "flood": 2,
    "damaged_road": 2,
    "normal": 3,
}
PRIORITY_NAMES = {0: "Critical", 1: "Emergency", 2: "Moderate", 3: "Normal"}

GROUNDING_PROMPTS = [
    "fire", "smoke", "flood water", "collapsed building",
    "damaged road", "crashed vehicle", "person needing rescue",
]


def model_directory(root: Path, key: str) -> Path:
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {key!r}; choose from {tuple(MODEL_SPECS)}")
    return Path(root) / key


def map_ladi_scores(scores: dict[str, float], threshold: float = 0.5) -> tuple[str | None, int | None]:
    """Apply an explicit, conservative pilot mapping to LADI multi-label scores."""

    active = {name.lower(): score for name, score in scores.items() if score >= threshold}
    if active.get("flooding_structures", 0) >= threshold:
        return "flooded_structures", 0
    if active.get("buildings_minor_or_greater", 0) >= threshold:
        return "collapsed_building", 1
    if active.get("flooding_any", 0) >= threshold:
        return "flood", 2
    if active.get("roads_damage", 0) >= threshold or active.get("debris_any", 0) >= threshold:
        return "damaged_road", 2
    # No detected label is not proof of a normal scene.
    return None, None


def map_grounding_label(label: str) -> tuple[str | None, int | None]:
    value = label.lower()
    if "rescue" in value:
        return "rescue_needed", 0
    if "fire" in value or "smoke" in value:
        return "fire_smoke", 1
    if "collapsed" in value:
        return "collapsed_building", 1
    if "crashed" in value:
        return "traffic_accident", 1
    if "flood" in value:
        return "flood", 2
    if "road" in value:
        return "damaged_road", 2
    return None, None


class PriorityModel:
    """Load one local model once and infer over many images."""

    def __init__(self, key: str, model_root: Path, device: str = "auto") -> None:
        if key not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {key}")
        self.key = key
        self.spec = MODEL_SPECS[key]
        self.path = model_directory(model_root, key)
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(
                f"Model {key} is missing from {self.path}. Run scripts/download_priority_models.py first."
            )
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = device
        self._torch = torch
        self._load()

    def _load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

        if self.spec.kind == "zero_shot":
            self.runner = pipeline(
                "zero-shot-image-classification", model=str(self.path),
                device=0 if self.device == "cuda" else -1,
            )
        elif self.spec.kind == "grounding":
            self.runner = pipeline(
                "zero-shot-object-detection", model=str(self.path),
                device=0 if self.device == "cuda" else -1,
            )
        else:
            self.processor = AutoImageProcessor.from_pretrained(str(self.path), local_files_only=True)
            self.runner = AutoModelForImageClassification.from_pretrained(
                str(self.path), local_files_only=True
            ).to(self.device).eval()

    def predict(self, image: Any) -> dict[str, Any]:
        """Return native labels and a provisional priority; unknown stays null."""

        if self.spec.kind == "zero_shot":
            labels = list(SCENE_PROMPTS.values())
            predictions = self.runner(image, candidate_labels=labels)
            top = predictions[0]
            scene = next(key for key, prompt in SCENE_PROMPTS.items() if prompt == top["label"])
            scores = {key: next(float(row["score"]) for row in predictions if row["label"] == prompt)
                      for key, prompt in SCENE_PROMPTS.items()}
            priority = SCENE_TO_PRIORITY[scene]
            return self._record(scene, priority, float(top["score"]), scores)

        if self.spec.kind == "grounding":
            detections = self.runner(image, candidate_labels=GROUNDING_PROMPTS, threshold=0.30)
            top = max(detections, key=lambda row: row["score"], default=None)
            scene, priority = map_grounding_label(top["label"]) if top else (None, None)
            details = [{"label": row["label"], "score": float(row["score"]), "box": row["box"]}
                       for row in detections]
            return self._record(scene, priority, float(top["score"]) if top else None, details)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.runner(**inputs).logits[0]
        if self.spec.kind == "ladi":
            probabilities = self._torch.sigmoid(logits).cpu().tolist()
            scores = {self.runner.config.id2label[index]: float(probability)
                      for index, probability in enumerate(probabilities)}
            scene, priority = map_ladi_scores(scores)
            return self._record(scene, priority, max(scores.values()), scores)

        probabilities = self._torch.softmax(logits, dim=-1)
        top_indices = probabilities.topk(3).indices.cpu().tolist()
        scores = {self.runner.config.id2label[index]: float(probabilities[index])
                  for index in top_indices}
        # ImageNet labels do not define disaster response priority.
        return self._record(None, None, max(scores.values()), scores)

    def _record(self, scene: str | None, priority: int | None,
                score: float | None, native_output: Any) -> dict[str, Any]:
        return {
            "model_key": self.key,
            "model_repository": self.spec.repository,
            "scene_label": scene,
            "priority_class": priority,
            "priority_name": PRIORITY_NAMES.get(priority) if priority is not None else None,
            "top_score": score,
            "native_output": native_output,
            "score_note": "Raw model score; not a calibrated response-priority probability",
        }
