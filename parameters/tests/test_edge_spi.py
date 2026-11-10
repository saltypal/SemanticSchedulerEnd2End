import numpy as np

from parameters.edge import spi


def blobs(n, seed, sep=2.5, noise=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    centers = np.eye(4, 8) * sep
    return centers[y] + rng.normal(0, noise, (n, 8)), y


def test_fit_calibration_and_roundtrip(tmp_path):
    Xt, yt = blobs(400, 0, noise=1.5)
    Xv, yv = blobs(400, 1, noise=1.5)
    head, info = spi.fit_spi(Xt, yt, Xv, yv, (0.1, 1.0, 100.0), seed=0)
    assert head.temperature > 0 and info["C"] in (0.1, 1.0, 100.0)
    rep = spi.evaluate_spi(head, Xv, yv, source=np.array(["a"] * 200 + ["b"] * 200))
    assert rep["ece"] <= rep["ece_uncalibrated"] + 1e-9
    assert set(rep["by_source"]) == {"a", "b"} and 0 <= rep["critical_recall"] <= 1
    head.save(tmp_path / "h")
    h2 = spi.SPIHead.load(tmp_path / "h")
    assert np.allclose(h2.predict_proba(Xv), head.predict_proba(Xv))
    assert np.allclose(head.predict_proba(Xv).sum(axis=1), 1.0)


def test_overconfident_head_is_cooled():
    Xt, yt = blobs(300, 2, sep=1.0, noise=2.0)
    Xv, yv = blobs(300, 3, sep=1.0, noise=2.0)
    head, _ = spi.fit_spi(Xt, yt, Xv, yv, (1000.0,), seed=0)
    assert head.temperature > 1.0


def test_kappa_gate_and_ablation():
    assert spi.agreement_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == 1.0
    assert spi.agreement_kappa([0, 1, 0, 1], [1, 0, 1, 0]) < 0
    assert spi.gate_critical_recall({"critical_recall": 0.9}, 0.8)
    assert not spi.gate_critical_recall({"critical_recall": 0.5}, 0.8)
    X, y = blobs(200, 4)
    ev = np.zeros(200, bool)
    ev[:50] = True
    y = np.where(ev, 3, y)
    out = spi.no_normals_ablation(X, y, ev, X, ev)
    assert 0 <= out["false_critical_without_normals"] <= 1 and "false_critical_with_normals" in out
