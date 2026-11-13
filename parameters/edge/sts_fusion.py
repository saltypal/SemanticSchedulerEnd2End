"""D19: receiver-side fusion of the SPI posterior and the 13-value STS into one tier belief.

Features x = [log(clip(tier_probs)) (4), sts (13)]. Candidates: logistic, logistic with SPI x STS-histogram
interactions, HistGradientBoosting, small MLP; each temperature-scaled on val, plus an optional Critical logit
bias searched on val. Selection (``select``): highest val macro-F1 subject to val Critical recall >= SPI's and
ECE <= SPI ECE + 0.01; if none qualifies, the highest Critical recall. Header-only inputs; val only for selection.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from parameters.edge.spi import N_CLASSES, _macro_f1, _report, _softmax, fit_temperature

N_STS = 13
BIAS_GRID = tuple(float(b) for b in np.arange(0.0, 3.01, 0.25))
ECE_SLACK = 0.01


def _base(P, S):
    return np.hstack([np.log(np.clip(np.asarray(P, float), 1e-6, 1.0)), np.asarray(S, float)])


def _expand(X, mode):
    """mode 'inter': append the outer product of the 4 log-SPI with the 7 histogram values (columns 4:11)."""
    X = np.asarray(X, float)
    if mode != "inter":
        return X
    return np.hstack([X, (X[:, :4, None] * X[:, None, 4:11]).reshape(len(X), -1)])


def _features(rows):
    return _base([r["tier_probs"] for r in rows], [r["sts"] for r in rows])


def _usable(rows, split=None):
    return [r for r in rows if len(r.get("sts") or []) == N_STS and (split is None or r.get("split") == split)]


class StsFusion:
    """Belief = softmax(raw_logits / T + bias). Linear kind: standardised (optionally interaction) features;
    model kind: pickled sklearn classifier whose log-probabilities are the raw logits (stored base64 in the JSON)."""

    def __init__(self, mean, scale, coef=None, intercept=None, temperature=1.0, classes=(0, 1, 2, 3),
                 mode="plain", model=None, bias=None):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef = None if coef is None else np.asarray(coef, float)
        self.intercept = None if intercept is None else np.asarray(intercept, float)
        self.temperature, self.classes = float(temperature), [int(c) for c in classes]
        self.mode, self.model = mode, model
        self.bias = np.zeros(len(self.classes)) if bias is None else np.asarray(bias, float)

    def logits(self, X):
        Z = (_expand(X, self.mode) - self.mean) / self.scale
        if self.model is not None:
            return np.log(np.clip(self.model.predict_proba(Z), 1e-9, 1.0))
        return Z @ self.coef.T + self.intercept

    def proba(self, X):
        """(n, 4) probabilities in tier order (missing classes get 0)."""
        p = _softmax(self.logits(X) / self.temperature + self.bias)
        out = np.zeros((len(p), N_CLASSES))
        out[:, self.classes] = p
        return out

    def __call__(self, tier_probs, sts):
        if len(sts) != N_STS:
            return tuple(tier_probs)
        return tuple(float(v) for v in self.proba(_base([tier_probs], [sts]))[0])

    def to_dict(self):
        d = dict(mean=self.mean.tolist(), scale=self.scale.tolist(), temperature=self.temperature,
                 classes=self.classes, mode=self.mode, bias=self.bias.tolist())
        if self.model is not None:
            import base64
            import pickle
            d["model_b64"] = base64.b64encode(pickle.dumps(self.model)).decode("ascii")
        else:
            d.update(coef=self.coef.tolist(), intercept=self.intercept.tolist())
        return d

    @classmethod
    def from_dict(cls, d):
        model = None
        if d.get("model_b64"):
            import base64
            import pickle
            model = pickle.loads(base64.b64decode(d["model_b64"]))   # our own fitted artefact
        return cls(d["mean"], d["scale"], d.get("coef"), d.get("intercept"), d.get("temperature", 1.0),
                   d.get("classes", (0, 1, 2, 3)), d.get("mode", "plain"), model, d.get("bias"))


def _fit_candidate(name, Xt, yt, Xv, yv, seed, C_grid):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    mode = "inter" if name == "logreg_inter" else "plain"
    E = _expand(Xt, mode)
    mean = E.mean(axis=0)
    scale = np.where(E.std(axis=0) > 1e-9, E.std(axis=0), 1.0)
    Zt, Zv = (E - mean) / scale, (_expand(Xv, mode) - mean) / scale
    if name in ("logreg", "logreg_inter"):
        best = None
        for C in C_grid:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed).fit(Zt, yt)
            f1 = _macro_f1(yv, clf.predict(Zv))
            if best is None or f1 > best[0]:
                best = (f1, clf)
        clf = best[1]
        fus = StsFusion(mean, scale, clf.coef_, clf.intercept_, 1.0, clf.classes_, mode)
    else:
        if name == "hgb":
            clf = HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, learning_rate=0.08,
                                                 early_stopping=True, random_state=seed).fit(Zt, yt)
        else:
            clf = MLPClassifier((48,), early_stopping=True, validation_fraction=0.1, max_iter=300,
                                random_state=seed).fit(Zt, yt)
        fus = StsFusion(mean, scale, None, None, 1.0, clf.classes_, mode, model=clf)
    fus.temperature = fit_temperature(fus.logits(Xv), yv)
    return fus


def select(cands, spi_rep):
    """Highest val macro-F1 s.t. Critical recall >= SPI's and ECE <= SPI ECE + 0.01; else highest Critical
    recall (ties: macro-F1). Returns (best, constraint_met)."""
    ok = [c for c in cands if c["critical_recall"] >= spi_rep["critical_recall"] - 1e-12
          and c["ece"] <= spi_rep["ece"] + ECE_SLACK]
    if ok:
        return max(ok, key=lambda c: c["macro_f1"]), True
    return max(cands, key=lambda c: (c["critical_recall"], c["macro_f1"])), False


def fit(rows, seed=0, folds=5, C_grid=(0.1, 1.0, 10.0), names=("logreg", "logreg_inter", "hgb", "mlp"),
        biases=BIAS_GRID):
    """Fit all candidates on split=='train'; temperature and the Critical bias on split=='val'; pick by
    ``select``. ``folds`` is kept for API stability. Result carries .candidates/.chosen/.constraint_met/.spi_val."""
    tr, va = _usable(rows, "train"), _usable(rows, "val")
    if not tr or not va:
        raise ValueError("need rows with 13-value sts in both train and val splits")
    Xt, Xv = _features(tr), _features(va)
    yt, yv = np.asarray([r["tier"] for r in tr], int), np.asarray([r["tier"] for r in va], int)
    spi = _report(np.asarray([r["tier_probs"] for r in va], float), yv)
    cands = []
    for name in names:
        fus = _fit_candidate(name, Xt, yt, Xv, yv, seed, C_grid)
        for b in biases:
            fus.bias = np.zeros(len(fus.classes))
            if 0 in fus.classes:
                fus.bias[fus.classes.index(0)] = float(b)
            rep = _report(fus.proba(Xv), yv)
            snap = copy.copy(fus)
            snap.bias = fus.bias.copy()
            cands.append(dict(name=name, bias=float(b), fusion=snap, macro_f1=rep["macro_f1"],
                              critical_recall=rep["critical_recall"], ece=rep["ece"], recall=rep["recall"]))
    best, met = select(cands, spi)
    fus = best["fusion"]
    fus.candidates = [{k: v for k, v in c.items() if k != "fusion"} for c in cands]
    fus.spi_val, fus.constraint_met, fus.chosen = spi, met, dict(name=best["name"], bias=best["bias"])
    return fus


def report(rows, fusion):
    """Val-only: SPI alone vs fused (macro-F1, per-tier recall/precision, Critical recall, ECE) and delta."""
    va = _usable(rows, "val")
    y = np.asarray([r["tier"] for r in va], int)
    spi = _report(np.asarray([r["tier_probs"] for r in va], float), y)
    fused = _report(fusion.proba(_features(va)), y)
    delta = {k: fused[k] - spi[k] for k in ("macro_f1", "critical_recall", "ece")}
    return dict(n_val=int(len(y)), spi=spi, fused=fused, delta=delta)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit the SPI+STS receiver fusion (D19)")
    ap.add_argument("--table", required=True, help="simtable_v1.jsonl or a folder containing it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t = Path(a.table)
    if t.is_dir():
        t = t / "simtable_v1.jsonl"
    rows = [json.loads(x) for x in t.read_text(encoding="utf-8").splitlines() if x.strip()]
    fus = fit(rows, seed=a.seed)
    rep = report(rows, fus)
    rep.update(candidates=fus.candidates, chosen=fus.chosen, constraint_met=fus.constraint_met)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sts_fusion.json").write_text(json.dumps(fus.to_dict()), encoding="utf-8")
    (out / "sts_fusion_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("val n=%d      macroF1  critRecall  ECE" % rep["n_val"])
    for k in ("spi", "fused"):
        print("%-12s %.4f   %.4f     %.4f" % (k, rep[k]["macro_f1"], rep[k]["critical_recall"], rep[k]["ece"]))
    print("candidates (b=0 and chosen): name bias macroF1 critRecall ECE recall[C,H,M,N]")
    for c in fus.candidates:
        if c["bias"] == 0 or (c["name"], c["bias"]) == (fus.chosen["name"], fus.chosen["bias"]):
            print("  %-13s b=%.2f F1=%.4f crit=%.4f ece=%.4f recall=%s" % (
                c["name"], c["bias"], c["macro_f1"], c["critical_recall"], c["ece"],
                [round(v, 3) for v in c["recall"].values()]))
    print("chosen", fus.chosen, "constraint_met", fus.constraint_met)
    return rep


if __name__ == "__main__":
    main()
