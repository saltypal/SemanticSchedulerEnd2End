"""Fit the receiver-side models: the duplicate logistic (``DupModel``) and the fidelity predictor u_hat.

Reuses ``receiver.DupModel`` and ``contracts.Header``; JSON save/load for both.
"""
import json
from pathlib import Path

import numpy as np

from Evaluation.edge.receiver import DupModel


def _roc_auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def fit_dup_model(pairs, sem_keys, perc_keys, C=10.0):
    """Logistic fit on (d_sem/64, d_perc/64). ``pairs``: dicts with id_a, id_b, label; keys: id -> int.

    Returns ``(DupModel, auc)``.
    """
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for p in pairs:
        a, b = p["id_a"], p["id_b"]
        X.append(((sem_keys[a] ^ sem_keys[b]).bit_count() / 64.0,
                  (perc_keys[a] ^ perc_keys[b]).bit_count() / 64.0))
        y.append(1 if p["label"] == "duplicate" else 0)
    if len(set(y)) < 2:
        raise ValueError("need both duplicate and distinct pairs")
    X = np.asarray(X)
    clf = LogisticRegression(C=C, max_iter=1000).fit(X, y)
    model = DupModel(float(clf.intercept_[0]), float(clf.coef_[0][0]), float(clf.coef_[0][1]))
    return model, _roc_auc(clf.decision_function(X), y)


def dup_scores(model, d_sem, d_perc):
    z = model.a + model.b1 * np.asarray(d_sem) / 64.0 + model.b2 * np.asarray(d_perc) / 64.0
    return 1.0 / (1.0 + np.exp(-z))


def per_event_auc(scores, labels, events):
    """ROC-AUC per event; events with a single class map to None."""
    scores, labels, events = np.asarray(scores), np.asarray(labels), np.asarray(events)
    out = {}
    for e in sorted(set(events.tolist())):
        m = events == e
        out[e] = _roc_auc(scores[m], labels[m]) if len(set(labels[m].tolist())) == 2 else None
    return out


class UHat:
    """Logistic fidelity predictor on (sts, tier_probs, one-hot m_index); callable on a Header."""

    def __init__(self, mean, scale, coef, intercept, n_m):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept, self.n_m = np.asarray(coef, float), float(intercept), int(n_m)

    def _x(self, sts, probs, m):
        onehot = np.zeros(self.n_m)
        onehot[min(max(int(m), 0), self.n_m - 1)] = 1.0
        return np.concatenate([np.asarray(sts, float), np.asarray(probs, float), onehot])

    def __call__(self, header):
        x = (self._x(header.sts, header.tier_probs, header.m_index) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(x @ self.coef + self.intercept))))

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept, "n_m": self.n_m}


def fit_u_hat(sts_matrix, spi_probs, m_index, fidelity, C=1.0):
    """Fit u_hat(header) = P(fidelity=1 | sts, tier probs, m). Constant when fidelity has one class."""
    from sklearn.linear_model import LogisticRegression
    sts, probs = np.asarray(sts_matrix, float), np.asarray(spi_probs, float)
    m, y = np.asarray(m_index, int), np.asarray(fidelity, int)
    if not (len(sts) == len(probs) == len(m) == len(y)) or len(y) == 0:
        raise ValueError("inputs must be equal-length and non-empty")
    n_m = int(m.max()) + 1
    X = np.hstack([sts, probs, np.eye(n_m)[m]])
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    if len(set(y.tolist())) < 2:
        p = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        return UHat(mean, scale, np.zeros(X.shape[1]), float(np.log(p / (1 - p))), n_m)
    clf = LogisticRegression(C=C, max_iter=1000).fit((X - mean) / scale, y)
    return UHat(mean, scale, clf.coef_[0], clf.intercept_[0], n_m)


def save_models(path, dup=None, u_hat=None):
    obj = {}
    if dup is not None:
        obj["dup"] = {"a": dup.a, "b1": dup.b1, "b2": dup.b2}
    if u_hat is not None:
        obj["u_hat"] = u_hat.to_dict()
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_models(path):
    """Returns ``(DupModel | None, UHat | None)``."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    dup = DupModel(**obj["dup"]) if "dup" in obj else None
    u = UHat(**obj["u_hat"]) if "u_hat" in obj else None
    return dup, u
