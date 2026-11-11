import csv
from collections import namedtuple

import numpy as np
import pytest

from parameters.edge import embedding_grid as eg
from parameters.edge.features import FakeEmbedder

CFG = {
    "encoders": [{"name": "good", "ue_cost": 90}, {"name": "mid", "ue_cost": 20}, {"name": "bad", "ue_cost": 5}],
    "classifiers": {"logistic": {"C": [0.1, 1.0]}, "knn": {"k": [5]}},
    "selection": {"min_critical_recall": 0.6, "f1_tie_tol": 0.005},
}


def make_data(seed=0, per=40, poison=False):
    rng = np.random.default_rng(seed)
    paths, y, split, src = [], [], [], []
    for sp, n in (("train", per), ("val", per // 2), ("test", per // 2)):
        for c in range(4):
            for i in range(n):
                paths.append("%s_%d_%d" % (sp, c, i))
                y.append(c)
                split.append(sp)
                src.append("everyday" if i % 5 == 0 else "event")
    y = np.array(y)
    centers = np.eye(4, 8) * 4
    embs = {}
    for name, noise in (("good", 0.6), ("mid", 2.5), ("bad", 8.0)):
        embs[name] = (np.array(paths), centers[y] + rng.normal(0, noise, (len(y), 8)))
    labels = {p: {"tier": int(c), "source": s} for p, c, s in zip(paths, y, src)}
    splits = dict(zip(paths, split))
    if poison:
        tst = [i for i, s in enumerate(split) if s == "test"]
        for n in embs:
            embs[n][1][tst] = rng.normal(0, 50, (len(tst), 8))
        for i in tst:
            labels[paths[i]]["tier"] = 0
    return embs, labels, splits


def test_run_grid_saves_and_resumes(tmp_path):
    embs, labels, splits = make_data()
    c1 = eg.run_grid(embs, labels, splits, CFG, tmp_path)
    files = list((tmp_path / "candidates").glob("*.json"))
    assert len(files) == len(c1) == 3 * 3
    mt = {f: f.stat().st_mtime_ns for f in files}
    c2 = eg.run_grid(embs, labels, splits, CFG, tmp_path)
    assert {f: f.stat().st_mtime_ns for f in files} == mt
    assert [c["id"] for c in c1] == [c["id"] for c in c2]


def test_selection_picks_separable_and_outputs(tmp_path):
    embs, labels, splits = make_data()
    chosen, model = eg.grid_and_select(embs, labels, splits, CFG, tmp_path)
    assert chosen["encoder"] == "good"
    assert (tmp_path / "chosen_spi.json").exists() and (tmp_path / "grid_report.json").exists()
    assert (tmp_path / "chosen_head.npz").exists() or (tmp_path / "chosen_head.pkl").exists()
    with open(tmp_path / "comparison.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert {"encoder", "family", "val_macro_f1", "critical_recall", "ece", "ue_cost"} <= set(header)


def test_gate_beats_higher_f1():
    cands = [
        {"id": "a", "val_macro_f1": 0.95, "critical_recall": 0.4, "ece": 0.01, "ue_cost": 1},
        {"id": "b", "val_macro_f1": 0.80, "critical_recall": 0.7, "ece": 0.05, "ue_cost": 50},
    ]
    assert eg.select_best(cands, CFG["selection"])["id"] == "b"
    with pytest.raises(ValueError):
        eg.select_best(cands[:1], CFG["selection"])


def test_tie_break_lower_cost_then_ece():
    base = {"critical_recall": 0.9}
    cands = [dict(base, id="x", val_macro_f1=0.900, ue_cost=80, ece=0.01),
             dict(base, id="y", val_macro_f1=0.898, ue_cost=20, ece=0.09),
             dict(base, id="z", val_macro_f1=0.899, ue_cost=20, ece=0.02)]
    assert eg.select_best(cands, CFG["selection"])["id"] == "z"


def test_test_rows_never_used(tmp_path):
    e1, l1, s1 = make_data()
    e2, l2, s2 = make_data(poison=True)
    a = eg.run_grid(e1, l1, s1, CFG, tmp_path / "a")
    b = eg.run_grid(e2, l2, s2, CFG, tmp_path / "b")
    key = lambda cs: [(c["id"], c["val_macro_f1"], c["ece"]) for c in cs]
    assert key(a) == key(b)
    assert eg.select_best(a, CFG["selection"])["id"] == eg.select_best(b, CFG["selection"])["id"]


def test_final_eval_requires_confirmation(tmp_path):
    embs, labels, splits = make_data()
    _, model = eg.grid_and_select(embs, labels, splits, CFG, tmp_path)
    with pytest.raises(PermissionError):
        eg.final_eval(model, embs["good"], labels, splits)
    rep = eg.final_eval(model, embs["good"], labels, splits, confirm_test=True)
    assert rep["n"] > 0


def test_extraction_resumable(tmp_path):
    items = ["img%d" % i for i in range(10)]
    emb = FakeEmbedder(dim=6)
    calls = []

    def loader(it):
        calls.append(it)
        return it
    eg.extract_encoder("enc", items, loader, 3, "cpu", tmp_path, shard_size=4, min_free_gb=0, embedder=emb)
    with np.load(tmp_path / "emb_enc.npz") as z:
        assert z["emb"].dtype == np.float16 and len(z["image_path"]) == 10
    assert (tmp_path / "emb_enc.done").exists() and not (tmp_path / "shards_enc").exists()
    # interrupted run: shard 0 exists, others missing -> only later shards are recomputed
    (tmp_path / "emb_enc.done").unlink()
    (tmp_path / "emb_enc.npz").unlink()
    sd = tmp_path / "shards_enc"
    sd.mkdir()
    np.savez(sd / "shard_00000.npz", image_path=np.array(items[:4]), emb=emb(items[:4]).astype(np.float16))
    calls.clear()
    eg.extract_encoder("enc", items, loader, 3, "cpu", tmp_path, shard_size=4, min_free_gb=0, embedder=emb)
    assert calls == items[4:]
    calls.clear()
    eg.extract_encoder("enc", items, loader, 3, "cpu", tmp_path, shard_size=4, min_free_gb=0, embedder=emb)
    assert calls == []


def test_disk_guard(monkeypatch, tmp_path):
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(eg.shutil, "disk_usage", lambda p: U(10e9, 9.5e9, 0.5e9))
    with pytest.raises(RuntimeError, match="free disk"):
        eg.disk_guard(2.0, tmp_path)
    with pytest.raises(RuntimeError):
        eg.extract_encoder("e", ["a"], lambda x: x, 1, "cpu", tmp_path, min_free_gb=2.0, embedder=FakeEmbedder(4))
    assert not (tmp_path / "emb_e.npz").exists()


# ---------------------------------------------------------------- _image_embedding (transformers 4/5 robust)

class _Out:
    """ModelOutput-like object exposing only the given attributes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_clip(dim_pool=8, dim_proj=5):
    torch = pytest.importorskip("torch")

    class Vision:
        def __call__(self, pixel_values):
            return _Out(pooler_output=torch.ones(pixel_values.shape[0], dim_pool))

    class M:
        vision_model = Vision()
        visual_projection = torch.nn.Linear(dim_pool, dim_proj, bias=False)

        def get_image_features(self, **kw):
            raise AssertionError("clip path must not call get_image_features")
    return M(), torch.zeros(3, 3, 4, 4)


def _fake_siglip(ret):
    torch = pytest.importorskip("torch")

    class M:
        def get_image_features(self, pixel_values):
            return ret(pixel_values.shape[0], torch)
    return M(), torch.zeros(2, 3, 4, 4)


def test_clip_uses_vision_model_and_projection():
    m, px = _fake_clip()
    out = eg._image_embedding(m, "clip", px)
    assert tuple(out.shape) == (3, 5)


def test_siglip_tensor_return():
    m, px = _fake_siglip(lambda n, t: t.ones(n, 7))
    assert tuple(eg._image_embedding(m, "siglip", px).shape) == (2, 7)


def test_siglip_modeloutput_pooler_output():
    m, px = _fake_siglip(lambda n, t: _Out(pooler_output=t.ones(n, 6), image_embeds=None))
    assert tuple(eg._image_embedding(m, "siglip", px).shape) == (2, 6)


def test_siglip_modeloutput_image_embeds():
    m, px = _fake_siglip(lambda n, t: _Out(image_embeds=t.ones(n, 4)))
    assert tuple(eg._image_embedding(m, "siglip", px).shape) == (2, 4)


def test_dim_mismatch_names_encoder_dims_and_type():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError) as e:
        eg._check_dim(torch.ones(2, 10), {"name": "clip_b32", "dim": 512})
    msg = str(e.value)
    assert "clip_b32" in msg and "got 10" in msg and "expected 512" in msg and "Tensor" in msg
    eg._check_dim(torch.ones(2, 512), {"name": "clip_b32", "dim": 512})


def _only_low_train():
    embs, labels, splits = make_data()
    for p in labels:
        if splits[p] == "train":
            labels[p]["tier"] = 3
    return embs, labels, splits


def test_grid_data_error_single_class_train_and_empty_val(tmp_path):
    embs, labels, splits = _only_low_train()
    with pytest.raises(eg.GridDataError) as e:
        eg.run_grid(embs, labels, splits, CFG, tmp_path)
    msg = str(e.value)
    assert "train n=" in msg and "'Low': " in msg and "val n=" in msg and isinstance(e.value, ValueError)
    embs, labels, splits = make_data()
    no_val = {p: ("test" if s == "val" else s) for p, s in splits.items()}
    with pytest.raises(eg.GridDataError, match="val n=0"):
        eg.run_grid(embs, labels, no_val, CFG, tmp_path)


def _write_grid_inputs(tmp_path, labels, splits, embs):
    import json
    names = {"good": "emb_good"}
    d = tmp_path / "grid"
    d.mkdir()
    (d / "labels_by_uid.jsonl").write_text("\n".join(json.dumps(
        {"image_path": p, "tier": int(v["tier"]), "source": v["source"]}) for p, v in labels.items()))
    (d / "splits_by_uid.json").write_text(json.dumps(splits))
    paths, emb = embs["good"]
    np.savez(d / "emb_good.npz", image_path=paths, emb=emb)
    (d / "emb_good.done").write_text("{}")
    return d


def test_cli_degenerate_flag(tmp_path, monkeypatch):
    import json
    cfg = dict(CFG, encoders=[{"name": "good", "ue_cost": 1}])
    monkeypatch.setattr(eg, "load_config", lambda name: cfg)
    embs, labels, splits = _only_low_train()
    d = _write_grid_inputs(tmp_path, labels, splits, embs)
    base = ["grid", "--embeddings-dir", str(d), "--labels", str(d / "labels_by_uid.jsonl"),
            "--splits", str(d / "splits_by_uid.json"), "--out", str(d)]
    with pytest.raises(eg.GridDataError):
        eg.main(base)                                                   # full mode: hard failure
    assert eg.main(base + ["--allow-degenerate"]) == 0                  # pilot: plumbing only
    assert "train n=" in json.loads((d / eg.DEGENERATE_FILE).read_text())["message"]
    assert eg.main(["select", "--grid-dir", str(d), "--allow-degenerate"]) == 0


def test_grid_yaml_dims_for_clip_and_siglip():
    from Evaluation.edge.edge_config import load_config
    dims = {e["name"]: e["dim"] for e in load_config("grid")["encoders"]}
    assert dims["clip_b32"] == 512 and dims["siglip2_base"] == 768
