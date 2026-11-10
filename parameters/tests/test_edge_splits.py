import numpy as np
import pytest

from parameters.edge import dup_pairs as D
from parameters.edge import splits as S


def test_cross_split_dup_removed():
    rng = np.random.default_rng(0)
    rows = [{"image_path": "a%d" % i, "split": "train"} for i in range(5)]
    rows += [{"image_path": "v0", "split": "val"}, {"image_path": "v1", "split": "val"},
             {"image_path": "t0", "split": "test"}]
    sem = [int(rng.integers(0, 2 ** 62)) << 1 for _ in rows]
    perc = [int(rng.integers(0, 2 ** 62)) << 1 for _ in rows]
    sem[5] = sem[0] ^ 0b11          # v0 semantic near-dup of train a0
    perc[7] = perc[1]               # t0 perceptual dup of train a1
    kept, rep = S.drop_cross_split_near_duplicates(rows, sem, perc, 3, 2)
    names = {r["image_path"] for r in kept}
    assert "v0" not in names and "t0" not in names and "v1" in names and "a0" in names
    assert rep["dropped"] == {"val": [5], "test": [7]}
    assert S.drop_cross_split_near_duplicates(rows, sem, perc, 3, 2)[1] == rep


def test_assign_event_splits():
    rows = [{"image_path": "p%d" % i, "event": "hurricane_harvey"} for i in range(5)]
    rows += [{"image_path": "p9", "event": "california_wildfires"},
             {"image_path": "p8", "event": "hurricane_maria"}]
    rows += [{"image_path": "n%d" % i, "event": ""} for i in range(1000)]
    s, rep = S.assign_event_splits(rows)
    assert s["p0"] == "train" and s["p9"] == "val" and s["p8"] == "test"
    n = sum(1 for i in range(1000) if s["n%d" % i] == "train")
    assert 540 < n < 660
    assert S.assign_event_splits(rows)[0] == s and rep["counts"]["train"] >= 5


def test_pairs_stratified(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"id": i, "tweet_id": str(i // 2)} for i in range(300)]
    base = int(rng.integers(0, 2 ** 62))
    sem = [base ^ (((1 << int(rng.integers(0, 20))) - 1) if i % 3 else int(rng.integers(0, 2 ** 40)))
           for i in range(300)]
    perc = [int(rng.integers(0, 2 ** 62)) for _ in rows]
    pairs = D.candidate_pairs(rows, sem, perc, 30, np.random.default_rng(1))
    assert 20 <= len(pairs) <= 30
    ds = [p[2] for p in pairs]
    assert any(d <= 8 for d in ds) and any(9 <= d <= 20 for d in ds) and any(d >= 21 for d in ds)
    path = tmp_path / "p.csv"
    D.write_csv(pairs, path)
    with pytest.raises(ValueError):
        D.load_labelled(path)


def test_unlisted_event_raises_instead_of_leaking():
    import pytest
    from parameters.edge import splits as sp
    cfg = {"train_events": ["a"], "val_events": ["b"], "test_events": ["c"], "normals_split": [0.6, 0.1, 0.3]}
    with pytest.raises(ValueError):
        sp.assign_event_splits([{"image_path": "x", "event": "zzz"}], cfg)
    out, _ = sp.assign_event_splits([{"image_path": "x", "event": "a"}, {"image_path": "y", "event": None}], cfg)
    assert out["x"] == "train" and out["y"] in {"train", "val", "test"}
