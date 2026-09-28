"""Fit small feature classifier and pairwise ranker on the training split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from INFRA.semantic_contracts import EpisodeJobSpec


def fit_supervised_models(
    train_jobs: Sequence[EpisodeJobSpec], output_dir: str | Path,
    *, provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Fit from reviewed training images only and export NumPy-only inference.

    The caller owns split validation. This function rejects duplicate image IDs
    to avoid overweighting a repeated training image.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from Scheduler.feature_adapter import static_feature_row

    if len(train_jobs) < 4:
        raise ValueError("At least four reviewed training images are required")
    image_ids = [item.candidate.image_id for item in train_jobs]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Supervised training requires one row per image")
    labels = np.asarray([item.base_priority for item in train_jobs], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError("Training split needs at least two priority classes")
    rows = np.stack([static_feature_row(item.candidate) for item in train_jobs])
    scaler = StandardScaler().fit(rows)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    standardized = (rows - scaler.mean_) / scale
    ordinal = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    ordinal.fit(standardized, labels)
    if len(ordinal.classes_) == 2:
        # sklearn's binary representation has one logit. Export two logits so
        # the runtime softmax and expected class calculation stay consistent.
        coefficients = np.vstack((np.zeros(10), ordinal.coef_[0]))
        intercepts = np.asarray((0.0, ordinal.intercept_[0]))
    else:
        coefficients = ordinal.coef_
        intercepts = ordinal.intercept_

    pair_rows: list[np.ndarray] = []
    pair_labels: list[int] = []
    for left in range(len(train_jobs)):
        for right in range(left + 1, len(train_jobs)):
            if labels[left] == labels[right]:
                continue
            difference = standardized[left] - standardized[right]
            preferred = int(labels[left] < labels[right])
            pair_rows.extend((difference, -difference))
            pair_labels.extend((preferred, 1 - preferred))
    if not pair_rows:
        raise ValueError("No cross-priority training pairs exist")
    ranker = LogisticRegression(fit_intercept=False, max_iter=2000, random_state=42)
    ranker.fit(np.stack(pair_rows), np.asarray(pair_labels))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    classifier_path = destination / "ordinal_classifier.npz"
    ranker_path = destination / "pairwise_ranker.npz"
    np.savez_compressed(
        classifier_path, coefficients=coefficients.astype(np.float32),
        intercepts=intercepts.astype(np.float32),
        mean=scaler.mean_.astype(np.float32), scale=scale.astype(np.float32),
        classes=ordinal.classes_.astype(np.float32),
    )
    np.savez_compressed(
        ranker_path, weights=ranker.coef_[0].astype(np.float32),
        mean=scaler.mean_.astype(np.float32), scale=scale.astype(np.float32),
    )
    manifest = {
        "schema": "track-a/v2:five-values-plus-five-masks",
        "training_image_ids": image_ids,
        "priority_classes": ordinal.classes_.tolist(),
        "pair_count": len(pair_labels),
        "oracle_assisted": True,
        "provenance": dict(provenance or {}),
    }
    (destination / "supervised_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"classifier": classifier_path, "ranker": ranker_path,
            "manifest": destination / "supervised_manifest.json"}
