"""Lazy CLIP adapter used only by the Transformer semantic-parameter path."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Semantic embeddings must have a non-zero norm.")
    return values / norm


def _feature_tensor(model_output: object):
    """Accept both Transformers v4 tensor returns and v5 model-output returns."""

    if hasattr(model_output, "detach"):
        return model_output
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        candidate = getattr(model_output, attribute, None)
        if candidate is not None and hasattr(candidate, "detach"):
            return candidate
    raise TypeError(f"CLIP feature call returned an unsupported object: {type(model_output)!r}")


class SemanticEmbeddingBackend(Protocol):
    """Narrow seam that lets unit tests avoid downloading or loading CLIP."""

    def encode_image(self, image_path: Path) -> np.ndarray:
        """Return one normalized embedding for one image."""

    def encode_texts(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Return normalized embeddings keyed by the supplied text."""


class CLIPBackbone:
    """Caches text features and runs the CLIP image backbone exactly once per image."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._text_cache: dict[str, np.ndarray] = {}
        self.image_forward_calls = 0
        self.text_forward_calls = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "Transformer extraction requires the optional stage1b dependencies. "
                "Install the project with: pip install -e .[stage1b]"
            ) from error
        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device
        self._model = CLIPModel.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=self.local_files_only, use_fast=False)
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        self._device = device

    def encode_image(self, image_path: Path) -> np.ndarray:
        """Encode a source image once; parameter extractors reuse this result."""

        self._ensure_loaded()
        from PIL import Image

        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with self._torch.no_grad():
            features = _feature_tensor(self._model.get_image_features(pixel_values=pixel_values))
        self.image_forward_calls += 1
        return _normalize(features.detach().cpu().numpy()[0])

    def encode_texts(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Encode only uncached prompts, preserving stable values across images."""

        requested = list(dict.fromkeys(texts))
        absent = [text for text in requested if text not in self._text_cache]
        if absent:
            self._ensure_loaded()
            assert self._processor is not None
            assert self._model is not None
            assert self._torch is not None
            inputs = self._processor(text=absent, padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)
            with self._torch.no_grad():
                features = _feature_tensor(self._model.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
            for text, feature in zip(absent, features.detach().cpu().numpy(), strict=True):
                self._text_cache[text] = _normalize(feature)
            self.text_forward_calls += 1
        return {text: self._text_cache[text].copy() for text in requested}
