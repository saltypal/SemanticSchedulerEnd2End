"""D19: receiver-side fusion of the SPI posterior and the 13-value STS into one tier belief.

Features x = [log(clip(tier_probs)) (4), sts (13)], standardised on train; multinomial logistic
regression (C by val macro-F1), temperature on val NLL. Header-only inputs. ``report`` is val only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from parameters.edge.spi import N_CLASSES, _macro_f1, _report, _softmax, fit_temperature

N_STS = 13


def _features(rows):
    p = np.log(np.clip(np.asarray([r["tier_probs"] for r in rows], float), 1e-6, 1.0))
    s = np.asarray([r["sts"] for r in rows], float)
    return np.hstack([p, s])


def _usable(rows, split=None):
    return [r for r in rows if len(r.get("sts") or []) == N_STS and (split is None or r.get("split") == split)]


class StsFusion:
    def __init__(self, mean, scale, coef, intercept, temperature=1.0, classes=(0, 1, 2, 3)):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept = np.asarray(coef, float), np.asarray(intercept, float)
        self.temperature, self.classes = float(temperature), [int(c) for c in classes]

    def logits(self, X):
        return ((np.asarray(X, float) - self.mean) / self.scale) @ self.coef.T + self.intercept

    def proba(self, X):
        """(n, 4) probabilities in tier order (missing classes get 0)."""
        p = _softmax(self.logits(X) / self.temperature)
        out = np.zeros((len(p), N_CLASSES))
        out[:, self.classes] = p
        return out

    def __call__(self, tier_probs, sts):
        if len(sts) != N_STS:
            return tuple(tier_probs)
        x = np.hstack([np.log(np.clip(np.asarray(tier_probs, float), 1e-6, 1.0)), np.asarray(sts, float)])[None]
        return tuple(float(v) for v in self.proba(x)[0])

    def to_dict(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), coef=self.coef.tolist(),
                    intercept=self.intercept.tolist(), temperature=self.temperature, classes=self.classes)

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("temperature", 1.0),
                   d.get("classes", (0, 1, 2, 3)))


def fit(rows, seed=0, folds=5, C_grid=(0.1, 1.0, 10.0)):
    """Fit on split=='train', choose C and temperature on split=='val'. ``folds`` is kept for API stability."""
    from sklearn.linear_model import LogisticRegression
    tr, va = _usable(rows, "train"), _usable(rows, "val")
    if not tr or not va:
        raise ValueError("need rows with 13-value sts in both train and val splits")
    Xt, Xv = _features(tr), _features(va)
    yt, yv = np.asarray([r["tier"] for r in tr], int), np.asarray([r["tier"] for r in va], int)
    mean = Xt.mean(axis=0)
    scale = np.where(Xt.std(axis=0) > 1e-9, Xt.std(axis=0), 1.0)
    best = None
    for C in C_grid:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit((Xt - mean) / scale, yt)
        f1 = _macro_f1(yv, clf.predict((Xv - mean) / scale))
        if best is None or f1 > best[0]:
            best = (f1, clf)
    clf = best[1]
    fus = StsFusion(mean, scale, clf.coef_, clf.intercept_, 1.0, clf.classes_)
    fus.temperature = fit_temperature(fus.logits(Xv), yv)
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sts_fusion.json").write_text(json.dumps(fus.to_dict()), encoding="utf-8")
    (out / "sts_fusion_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("val n=%d      macroF1  critRecall  ECE" % rep["n_val"])
    for k in ("spi", "fused"):
        print("%-12s %.4f   %.4f     %.4f" % (k, rep[k]["macro_f1"], rep[k]["critical_recall"], rep[k]["ece"]))
    print("delta        %+.4f   %+.4f     %+.4f" % (rep["delta"]["macro_f1"], rep["delta"]["critical_recall"], rep["delta"]["ece"]))
    return rep


if __name__ == "__main__":
    main()
