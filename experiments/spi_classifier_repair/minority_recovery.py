"""Bounded train-only failure study; no final-test reuse or automatic promotion."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.spi_classifier_repair.medic_campaign import score
from experiments.spi_classifier_repair.resume_store import atomic_json, file_hash

LEVELS = ("Low", "Medium", "High", "Critical")
FAMILIES = ("balanced_logistic", "rf_control", "rf_prior_adjustment")


class PriorAdjustedForest(ClassifierMixin, BaseEstimator):
    """Post-hoc prior adjustment hypothesis, even though RF fitting is balanced.

    p'(y|x) proportional to p(y|x)/prior(y)**tau. Priors are fit-fold only.
    This is not calibrated operational urgency and can overpredict rare classes.
    """
    def __init__(self, min_samples_leaf=5, tau=0.0, random_state=29):
        self.min_samples_leaf = min_samples_leaf
        self.tau = tau
        self.random_state = random_state

    def fit(self, x, y):
        y = np.asarray(y)
        if set(y) != set(range(4)) or not np.isfinite(self.tau) or self.tau < 0:
            raise ValueError("four-class fit support and nonnegative finite tau required")
        self.priors_ = np.bincount(y, minlength=4) / len(y)
        self.model_ = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced", n_jobs=1, random_state=self.random_state).fit(x, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x):
        raw = self.model_.predict_proba(x)
        adjusted = raw / self.priors_[None, :] ** self.tau
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(1)]


def configs(family):
    if family == "balanced_logistic":
        return [{"C": c} for c in (0.1, 1.0, 10.0)]
    if family == "rf_control":
        return [{"min_samples_leaf": leaf, "tau": 0.0} for leaf in (5, 2)]
    if family == "rf_prior_adjustment":
        return [{"min_samples_leaf": leaf, "tau": tau}
                for tau in (0.0, 0.5, 1.0) for leaf in (5, 2)]
    raise ValueError("unknown recovery family")


def build(family, config):
    if family == "balanced_logistic":
        return LogisticRegression(class_weight="balanced", max_iter=2000,
                                  random_state=29, **config)
    if family in ("rf_control", "rf_prior_adjustment"):
        return PriorAdjustedForest(**config)
    raise ValueError("unknown recovery family")


def grouped_folds(y, groups, seed=29):
    y, groups = np.asarray(y), np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    for train, validation in folds:
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("group leakage")
        if set(y[train]) != set(range(4)) or set(y[validation]) != set(range(4)):
            raise ValueError("insufficient four-class support for grouped folds")
    return folds


def choose(x, y, groups, family):
    """No zero-recall candidate can silently win on overall macro-F1 alone."""
    folds = grouped_folds(y, groups)
    grid = []
    for config in configs(family):
        predictions = np.full(len(y), -1, dtype=int)
        fold_metrics = []
        for train, validation in folds:
            model = build(family, config).fit(x[train], y[train])
            predictions[validation] = model.predict(x[validation])
            fold_metrics.append(score(y[validation], predictions[validation]))
        metrics = score(y, predictions)
        grid.append({"config": config, "pooled_oof": metrics,
                     "fold_metrics": fold_metrics,
                     "noncollapsed": metrics["worst_class_recall"] > 0})
    supported = [r for r in grid if r["noncollapsed"]]
    if supported:
        # Stable ordering above favors stronger regularization/simpler correction on ties.
        chosen = max(supported, key=lambda r: r["pooled_oof"]["macro_f1"])
    else:
        chosen = max(grid, key=lambda r: (r["pooled_oof"]["worst_class_recall"],
                                          r["pooled_oof"]["macro_f1"]))
    return {**chosen, "no_noncollapsed_candidate": not bool(supported)}, grid


def load_development(features, records, labels):
    rows = json.loads(Path(records).read_text(encoding="utf8"))
    target_rows = json.loads(Path(labels).read_text(encoding="utf8"))
    row_map = {r["image_id"]: r for r in rows}
    label_map = {r["image_id"]: r for r in target_rows}
    if len(row_map) != len(rows) or len(label_map) != len(target_rows):
        raise ValueError("duplicate source identities")
    with np.load(features, allow_pickle=False) as cache:
        identities = cache["image_ids"].tolist()
        x = cache["embeddings"]
    if x.ndim != 2 or len(x) != len(identities) or not np.isfinite(x).all():
        raise ValueError("invalid cached features")
    if len(set(identities)) != len(identities) or set(identities) != set(row_map):
        raise ValueError("feature identity mismatch")
    result = {}
    for role in ("train", "validation"):
        indices, kept = [], []
        for i, identity in enumerate(identities):
            row = row_map[identity]
            # Deliberately filter before reading any target from the excluded old test.
            if row.get("split") != role:
                continue
            if identity not in label_map:
                raise ValueError("missing development label")
            label = label_map[identity]
            if label.get("split") != role or label.get("group") != row.get("group"):
                raise ValueError("label role/group mismatch")
            if row["image_sha256"] != label.get("image_sha256"):
                raise ValueError("label image hash mismatch")
            if label.get("available") is not True:
                continue
            if label.get("level") not in LEVELS:
                raise ValueError("invalid development target")
            indices.append(i)
            kept.append({"image_id": identity, "group": row["group"],
                         "target": LEVELS.index(label["level"])})
        result[role] = {"x": x[indices], "y": np.array([r["target"] for r in kept]),
                        "groups": np.array([r["group"] for r in kept]), "records": kept}
    if set(result["train"]["groups"]) & set(result["validation"]["groups"]):
        raise ValueError("train/validation group leakage")
    return result


def run(features, records, labels, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = load_development(features, records, labels)
    tr, dev = data["train"], data["validation"]
    manifest = {"scope": "post-v20 exploratory recovery; no fresh-test result",
                "features_sha256": file_hash(features), "records_sha256": file_hash(records),
                "labels_sha256": file_hash(labels), "script_sha256": file_hash(__file__),
                "python": sys.version, "interpreter": sys.executable, "hardware": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "joblib")},
                "seed": 29, "cpu_threads": 1, "rows": {k: len(v["y"]) for k, v in data.items()},
                "class_counts": {k: np.bincount(v["y"], minlength=4).tolist() for k, v in data.items()},
                "selection": "training-only nested grouped3x3; exclude zero pooled-recall candidates before macroF1",
                "validation": "descriptive after training-only selection; no selector reads validation targets",
                "old_test": "excluded from target parsing, fitting, selection and evaluation",
                "encoder_inference": "reused exact v20 cached features, no fresh forward measured"}
    atomic_json(output/"manifest.json", manifest)
    outer = grouped_folds(tr["y"], tr["groups"])
    results = []
    for family in FAMILIES:
        begin = time.perf_counter()
        print("RECOVERY FAMILY START", family, flush=True)
        oof = np.full(len(tr["y"]), -1, dtype=int)
        fold_rows = []
        for fold, (a, b) in enumerate(outer):
            selected, grid = choose(tr["x"][a], tr["y"][a], tr["groups"][a], family)
            model = build(family, selected["config"]).fit(tr["x"][a], tr["y"][a])
            oof[b] = model.predict(tr["x"][b])
            fold_rows.append({"fold": fold, "selected": selected, "grid": grid,
                              "train_ids": [tr["records"][i]["image_id"] for i in a],
                              "validation_ids": [tr["records"][i]["image_id"] for i in b],
                              "outer_metrics": score(tr["y"][b], oof[b])})
            atomic_json(output/(family+"_outer_progress.json"), fold_rows)
            print("RECOVERY OUTER FOLD END", family, fold, flush=True)
        selected, grid = choose(tr["x"], tr["y"], tr["groups"], family)
        final = build(family, selected["config"]).fit(tr["x"], tr["y"])
        artifact = output/(family+".joblib")
        joblib.dump(final, artifact)
        predictions = [{**r, "prediction": int(p)} for r, p in zip(tr["records"], oof)]
        result = {"family": family, "nested_training_oof": score(tr["y"], oof),
                  "outer_folds": fold_rows, "selected_on_training": selected, "full_training_grid": grid,
                  "training_oof_predictions": predictions,
                  "artifact": artifact.name, "artifact_sha256": file_hash(artifact),
                  "artifact_bytes": artifact.stat().st_size, "fit_search_seconds": time.perf_counter()-begin}
        results.append(result)
        atomic_json(output/"results.json", results)
    # Freeze the complete training-only choice before computing any validation score.
    eligible = [r for r in results if r["nested_training_oof"]["worst_class_recall"] > 0]
    chosen = max(eligible or results, key=lambda r: r["nested_training_oof"]["macro_f1"])
    atomic_json(output/"selection_frozen.json", {
        "family": chosen["family"], "model_sha256": chosen["artifact_sha256"],
        "selection_role": "train", "fresh_final_evaluation_available": False,
        "runtime_promotion_allowed": False})
    for result in results:
        final = joblib.load(output/result["artifact"])
        pred = final.predict(dev["x"])
        result["validation_descriptive_only"] = score(dev["y"], pred)
        result["validation_predictions"] = [{**r, "prediction": int(p)}
                                             for r, p in zip(dev["records"], pred)]
        for _ in range(10):
            final.predict(dev["x"][:1])
        times = []
        for _ in range(100):
            begin = time.perf_counter_ns()
            final.predict(dev["x"][:1])
            times.append((time.perf_counter_ns()-begin)/1e6)
        result["head_only_cpu_ms"] = {"p50": float(np.median(times)),
                                      "p95": float(np.percentile(times, 95)), "batch": 1,
                                      "warmup": 10, "samples": 100, "threads": 1}
        result["flops"] = {"training": None, "inference": None,
                           "reason": "sklearn solvers/trees not instrumented; cache reuse is not fresh encoder inference"}
        if hasattr(final, "coef_"):
            result["flops"]["head_dense_matmul_arithmetic_proxy"] = int(2*final.coef_.size)
    atomic_json(output/"results.json", results)
    summary = {"selected_by_training_only": chosen["family"],
               "selected_training_oof": chosen["nested_training_oof"],
               "selected_validation_descriptive_only": chosen["validation_descriptive_only"],
               "elapsed_seconds": time.perf_counter()-started,
               "runtime_promotion_allowed": False, "independent_urgency_validated": False,
               "remaining_blockers": ["teacher reliability not independently established",
                                      "validation Critical support2", "no unexposed final test",
                                      "AI audit must not become automatic ground-truth relabeling"]}
    atomic_json(output/"summary.json", summary)
    print("RECOVERY COMPLETE", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("features", "records", "labels", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.features, args.records, args.labels, args.output)
