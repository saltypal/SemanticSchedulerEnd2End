"""Fit small feature classifier and pairwise ranker on the training split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from Evaluation.scheduling.semantic_contracts import EpisodeJobSpec


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
    from Scheduler.semantic.features import static_feature_row

    if len(train_jobs) < 4:
        raise ValueError("At least four reviewed training images are required")
    image_ids = [item.candidate.image_id for item in train_jobs]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Supervised training requires one row per image")
    labels = np.asarray([item.base_priority for item in train_jobs], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError("Training split needs at least two priority classes")
    rows = np.stack([static_feature_row(item.candidate) for item in train_jobs])
    feature_width = rows.shape[1]
    scaler = StandardScaler().fit(rows)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    standardized = (rows - scaler.mean_) / scale
    ordinal = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    ordinal.fit(standardized, labels)
    if len(ordinal.classes_) == 2:
        # sklearn's binary representation has one logit. Export two logits so
        # the runtime softmax and expected class calculation stay consistent.
        coefficients = np.vstack((np.zeros(feature_width), ordinal.coef_[0]))
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
        "schema": "track-a/v2:seven-values-plus-seven-masks",
        "feature_width": feature_width,
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


def fit_image_priority_classifier(
    train_jobs: Sequence[EpisodeJobSpec], catalogue: Sequence[EpisodeJobSpec],
    output_dir: str | Path, *, provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Train a small image-only priority model on cached CLIP embeddings.

    The frozen image encoder runs before this function. Only training-split
    embeddings and labels fit the scaler and classifier; predictions for the
    other images are inference outputs, not training examples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_ids = [item.candidate.image_id for item in train_jobs]
    catalogue_ids = [item.candidate.image_id for item in catalogue]
    if len(set(train_ids)) != len(train_ids) or len(set(catalogue_ids)) != len(catalogue_ids):
        raise ValueError("Image priority training requires unique image IDs")
    if not set(train_ids).issubset(catalogue_ids):
        raise ValueError("Training images must belong to the prediction catalogue")
    labels = np.asarray([item.base_priority for item in train_jobs], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError("Image priority training requires at least two classes")

    def embeddings(items: Sequence[EpisodeJobSpec]) -> np.ndarray:
        rows = np.asarray([item.candidate.features.embedding for item in items], dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] < 2 or not np.isfinite(rows).all():
            raise ValueError("Cached image embeddings must be finite vectors of equal width")
        return rows

    train_vectors = embeddings(train_jobs)
    all_vectors = embeddings(catalogue)
    if train_vectors.shape[1] != all_vectors.shape[1]:
        raise ValueError("Training and catalogue embedding widths differ")
    scaler = StandardScaler().fit(train_vectors)
    classifier = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=2000, random_state=42,
    ).fit(scaler.transform(train_vectors), labels)
    probabilities = classifier.predict_proba(scaler.transform(all_vectors))
    classes = classifier.classes_.astype(np.int64)
    predictions = {
        image_id: {str(int(label)): float(probability)
                   for label, probability in zip(classes, row)}
        for image_id, row in zip(catalogue_ids, probabilities)
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "clip_priority_classifier.npz"
    prediction_path = destination / "clip_priority_predictions.json"
    np.savez_compressed(
        weights_path, coefficients=classifier.coef_.astype(np.float32),
        intercepts=classifier.intercept_.astype(np.float32),
        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
        classes=classes, embedding_width=np.asarray(train_vectors.shape[1]),
    )
    prediction_path.write_text(json.dumps(predictions, indent=2, sort_keys=True),
                               encoding="utf-8")
    (destination / "clip_priority_manifest.json").write_text(json.dumps({
        "model": "l2_logistic_regression_on_frozen_clip_image_embeddings",
        "training_image_ids": train_ids,
        "classes": classes.tolist(),
        "embedding_width": int(train_vectors.shape[1]),
        "regularization_C": 0.1,
        "image_only": True,
        "provenance": dict(provenance or {}),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return {"weights": weights_path, "predictions": prediction_path,
            "manifest": destination / "clip_priority_manifest.json"}
