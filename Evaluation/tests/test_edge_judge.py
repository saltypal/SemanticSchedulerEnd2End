import csv
import numpy as np
from Evaluation.edge import judge as J


def blobs(n=60):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, .5, (n, 4)), rng.normal(3, .5, (n, 4))])
    return x, np.array([0] * n + [1] * n)


def test_fidelity_logic():
    x, y = blobs()
    j = J.LinearProbeJudge().fit(x, y)
    flipped = x.copy()
    flipped[:5] *= -1
    u = J.fidelity(j, x, flipped)
    assert (u[:5] == 0).all() and (u[5:] == 1).all()
    assert j.predict_proba(x).shape == (120, 2)


def test_save_load(tmp_path):
    x, y = blobs()
    j = J.LinearProbeJudge().fit(x, y)
    j.save(tmp_path / "j.pkl")
    assert (J.LinearProbeJudge.load(tmp_path / "j.pkl").predict(x) == j.predict(x)).all()


def test_audit_roundtrip(tmp_path):
    pairs = [("o%d" % i, "r%d" % i, i % 2) for i in range(20)]
    out = tmp_path / "a.csv"
    assert J.audit_sheet(pairs, 10, np.random.default_rng(1), out) == 10
    assert J.agreement(out) is None
    rows = list(csv.DictReader(open(out)))
    for r in rows:
        r["human_label"] = r["u"]
    rows[0]["human_label"] = str(1 - int(rows[0]["u"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert abs(J.agreement(out) - 0.9) < 1e-9
