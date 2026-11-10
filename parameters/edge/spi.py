"""SPI head: class-balanced multinomial logistic regression on CLIP embeddings with temperature scaling.

Chooses C by validation macro-F1, then fits one temperature on validation by minimising NLL. Provides
evaluation (macro-F1, per-class precision/recall, confusion, ECE, per-source), the no-normals ablation,
Cohen kappa and a Critical-recall gate. Features npz keys: ``image_path`` and ``emb``; labels JSONL rows
from ``parameters.edge.labels`` (``image_path``, ``tier``, ``source``); splits JSON maps path -> split.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TIERS = ("Critical", "High", "Medium", "Low")
N_CLASSES = len(TIERS)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SPIHead:
    """Linear head with temperature: p = softmax((X W^T + b) / T) over the four tiers."""

    def __init__(self, coef, intercept, classes, temperature=1.0):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = np.asarray(intercept, dtype=float)
        self.classes = [int(c) for c in classes]
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def logits(self, X):
        z = np.asarray(X, dtype=float) @ self.coef.T + self.intercept
        full = np.full((len(z), N_CLASSES), -1e9)
        full[:, self.classes] = z
        return full

    def predict_proba(self, X, calibrated=True):
        t = self.temperature if calibrated else 1.0
        return _softmax(self.logits(X) / t)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, prefix):
        prefix = str(prefix)
        np.savez(prefix + ".npz", coef=self.coef, intercept=self.intercept)
        Path(prefix + ".json").write_text(json.dumps(
            {"classes": self.classes, "temperature": self.temperature}), encoding="utf-8")

    @classmethod
    def load(cls, prefix):
        prefix = str(prefix)
        arr = np.load(prefix + ".npz")
        meta = json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))
        return cls(arr["coef"], arr["intercept"], meta["classes"], meta["temperature"])


def _fit_lr(X, y, C, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(X, y)
    return SPIHead(clf.coef_, clf.intercept_, clf.classes_)


def _macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", labels=sorted(set(y.tolist())), zero_division=0))


def _nll(probs, y):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(logits, y):
    """T minimising validation NLL (log-grid then local refinement)."""
    y = np.asarray(y, dtype=int)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    best = min(grid, key=lambda t: _nll(_softmax(logits / t), y))
    fine = np.linspace(best * 0.95, best * 1.05, 101)
    return float(min(fine, key=lambda t: _nll(_softmax(logits / t), y)))


def fit_spi(X_train, y_train, X_val, y_val, C_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """Train per C, pick best val macro-F1, then temperature-scale on val. Returns (head, info)."""
    X_train, X_val = np.asarray(X_train, float), np.asarray(X_val, float)
    y_train, y_val = np.asarray(y_train, int), np.asarray(y_val, int)
    if len(C_grid) == 0:
        raise ValueError("C_grid must be non-empty")
    best = None
    scores = {}
    for C in C_grid:
        head = _fit_lr(X_train, y_train, C, seed)
        f1 = _macro_f1(y_val, head.predict(X_val))
        scores[float(C)] = f1
        if best is None or f1 > best[0]:
            best = (f1, C, head)
    f1, C, head = best
    head.temperature = fit_temperature(head.logits(X_val), y_val)
    return head, {"C": float(C), "val_macro_f1": f1, "scores": scores, "temperature": head.temperature}


def ece(probs, y, n_bins=15):
    """Expected calibration error on top-label confidence."""
    probs, y = np.asarray(probs), np.asarray(y, int)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def _report(probs, y):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    pred = probs.argmax(axis=1)
    labels = list(range(N_CLASSES))
    p, r, _, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {"n": int(len(y)), "macro_f1": _macro_f1(y, pred), "ece": ece(probs, y),
            "precision": {TIERS[i]: float(p[i]) for i in labels},
            "recall": {TIERS[i]: float(r[i]) for i in labels},
            "support": {TIERS[i]: int(sup[i]) for i in labels},
            "critical_recall": float(r[0]),
            "confusion": confusion_matrix(y, pred, labels=labels).tolist()}


def evaluate_spi(head, X, y, source=None):
    """Metrics on calibrated probabilities; ``by_source`` filled when ``source`` is given."""
    y = np.asarray(y, int)
    probs = head.predict_proba(X)
    rep = _report(probs, y)
    rep["ece_uncalibrated"] = ece(head.predict_proba(X, calibrated=False), y)
    if source is not None:
        source = np.asarray(source)
        rep["by_source"] = {str(s): _report(probs[source == s], y[source == s])
                            for s in sorted(set(source.tolist()))}
    return rep


def no_normals_ablation(X_train, y_train, everyday_train, X_eval, everyday_eval, C=1.0, seed=0):
    """Retrain without everyday images; report the false-Critical rate on everyday eval images."""
    everyday_train, everyday_eval = np.asarray(everyday_train, bool), np.asarray(everyday_eval, bool)
    if not everyday_eval.any():
        raise ValueError("no everyday images in the evaluation set")
    X_train, y_train, X_eval = np.asarray(X_train, float), np.asarray(y_train, int), np.asarray(X_eval, float)

    def rate(head):
        return float((head.predict(X_eval[everyday_eval]) == 0).mean())
    without = _fit_lr(X_train[~everyday_train], y_train[~everyday_train], C, seed)
    out = {"false_critical_without_normals": rate(without)}
    if everyday_train.any():
        out["false_critical_with_normals"] = rate(_fit_lr(X_train, y_train, C, seed))
    return out


def agreement_kappa(labels_a, labels_b):
    """Cohen's kappa between two label sequences."""
    a, b = list(labels_a), list(labels_b)
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be equal-length and non-empty")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return 1.0 if pe >= 1.0 else float((po - pe) / (1.0 - pe))


def gate_critical_recall(report, min_recall):
    """True when Critical recall in the report meets the threshold."""
    return bool(report["critical_recall"] >= min_recall)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True, help="output prefix for the head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-critical-recall", type=float, default=None)
    args = ap.parse_args(argv)
    feats = np.load(args.features, allow_pickle=False)
    emb = {str(p): i for i, p in enumerate(feats["image_path"])}
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["image_path"] in emb and r["image_path"] in splits]
    tier_idx = {t: i for i, t in enumerate(TIERS)}
    data = {}
    for s in ("train", "val", "test"):
        sub = [r for r in rows if splits[r["image_path"]] == s]
        data[s] = (feats["emb"][[emb[r["image_path"]] for r in sub]],
                   np.array([tier_idx[r["tier"]] for r in sub], int),
                   np.array([r.get("source") or "" for r in sub]))
    head, info = fit_spi(data["train"][0], data["train"][1], data["val"][0], data["val"][1], seed=args.seed)
    head.save(args.out)
    report = {"fit": info, "val": evaluate_spi(head, data["val"][0], data["val"][1], data["val"][2])}
    if len(data["test"][1]):
        report["test"] = evaluate_spi(head, data["test"][0], data["test"][1], data["test"][2])
        if args.min_critical_recall is not None:
            report["critical_recall_gate"] = gate_critical_recall(report["test"], args.min_critical_recall)
    Path(str(args.out) + "_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
