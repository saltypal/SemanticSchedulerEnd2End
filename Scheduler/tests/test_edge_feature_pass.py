"""Feature pass tests: CPU only, fake embedder / SQ-GAN / clocks injected, no downloads."""
import copy
import hashlib
import json

import numpy as np
import pytest

from Codec.edge import compute_profile, feature_pass as fp
from Evaluation.edge.edge_config import load_config

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ENCODERS = ["clip_b32", "dinov2_small"]
DIMS = {"clip_b32": 512, "dinov2_small": 384}


def _vec(name, im):
    seed = int(np.asarray(im).astype(np.int64).sum()) % (2 ** 31) + len(name)
    v = np.random.default_rng(seed).standard_normal(DIMS[name])
    return v / np.linalg.norm(v)


class FakeSQ:
    def __init__(self):
        self.encodes = 0

    def segment(self, arr):
        return (arr[..., 0].astype(np.int64) // 13) % 20, arr

    def encode(self, arr, handle, m):
        self.encodes += 1
        rng = np.random.default_rng(int(arr.sum()) % 1000 + int(m * 100))
        kept = np.sort(rng.choice(512, int(512 * m * 0.3), replace=False))
        return {"payload": bytes(rng.integers(0, 256, 40 + int(m * 100), dtype=np.uint8)), "kept": kept.tolist(),
                "features": rng.standard_normal((8, 16, 32)).astype(np.float32)}

    def decode(self, payload, m):
        return np.full((256, 512, 3), len(payload) % 255, dtype=np.uint8)


def _make_cfg(shard_size=5, batch=4, **over):
    cfg = copy.deepcopy(load_config("feature_pass"))
    cfg["shard"].update(size=shard_size, batch_size=batch, min_free_gb=0.0, warmup_images=2)
    cfg["embeddings"]["encoders"] = ENCODERS
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def _dataset(tmp_path, n=12, bad=()):
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        f = "images/img%02d.jpg" % i
        if i in bad:
            (root / f).write_bytes(b"not a jpeg")
        else:
            arr = np.random.default_rng(i).integers(0, 255, (60 + i, 90 + 3 * i, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / f)
        rows.append({"uid": "crisismmd:i%02d" % i, "file": f, "tier": "Low", "event": "hurricane_harvey" if i % 2 else None,
                     "source": "crisismmd", "image_path": "i%02d" % i})
    return root, rows


def _ctx(root, out, cfg, fail=(), sq=True, counter=None):
    def emb_factory(name):
        if name in fail:
            raise RuntimeError("model missing")

        def fn(images):
            if counter is not None:
                counter.append(len(images))
            return np.stack([_vec(name, im) for im in images])
        return fn
    fake = FakeSQ()
    backends = fp.Backends(emb_factory, (lambda: fake) if sq else None, clock=(lambda it=iter(__import__("itertools").count(0, 0.002)): next(it)))
    return fp.Context(cfg, load_config("grid"), root, out, backends, log=lambda *a: None), fake


def _shards(out):
    return sorted((out / "shards").glob("shard_????"))


def test_pass_aligned_and_complete(tmp_path):
    root, rows = _dataset(tmp_path)
    ctx, _ = _ctx(root, tmp_path / "o", _make_cfg())
    summ = fp.run(ctx, rows)
    assert summ["counts"]["ok"] == 12 and summ["n_failures"] == 0
    assert (tmp_path / "o/run_manifest.json").exists()
    for s in _shards(tmp_path / "o"):
        uids = json.loads((s / "uids.json").read_text())
        n = len(uids)
        assert (s / "done.json").exists()
        for f in s.glob("*.npy"):
            if f.name.startswith("offsets_"):
                assert len(np.load(f)) == n + 1
            else:
                assert len(np.load(f)) == n, f.name
        assert np.load(s / "emb_clip_b32.npy").dtype == np.float16
        assert np.load(s / "simhash64.npy").dtype == np.uint64
        assert np.load(s / "prefeat_meanmax.npy").shape == (n, 16)
        assert np.load(s / "class_grid.npy").shape == (n, 16, 32)
        off = np.load(s / "offsets_m0.25.npy")
        assert off[-1] == (s / "payloads_m0.25.bin").stat().st_size
        assert len(list((s / "recon_m0.55").glob("*.jpg"))) == n
        j = Image.open(next((s / "recon_m0.55").glob("*.jpg")))
        assert j.size == (512, 256)


def test_resumable_skips_finished_shards(tmp_path):
    root, rows = _dataset(tmp_path)
    out = tmp_path / "o"
    ctx, _ = _ctx(root, out, _make_cfg())
    fp.run(ctx, rows)
    shards = _shards(out)
    assert len(shards) == 3
    import shutil
    shutil.rmtree(shards[1])
    (out / "shards" / (shards[2].name + ".partial")).mkdir()          # stale partial must be ignored
    calls = []
    ctx2, _ = _ctx(root, out, _make_cfg(), counter=calls)
    summ = fp.run(ctx2, rows)
    assert len(_shards(out)) == 3 and summ["counts"]["ok"] == 12
    n_redone = len(json.loads((shards[1] / "uids.json").read_text()))
    assert sum(calls) == n_redone * len(ENCODERS)


def test_part_split_covers_all_shards(tmp_path):
    root, rows = _dataset(tmp_path)
    out = tmp_path / "o"
    for i in range(2):
        ctx, _ = _ctx(root, out, _make_cfg())
        fp.run(ctx, rows, part=(i, 2))
    assert fp.summarize(out)["counts"]["ok"] == 12


def test_hashes_deterministic(tmp_path):
    root, rows = _dataset(tmp_path)
    res = []
    for k in range(2):
        ctx, _ = _ctx(root, tmp_path / ("o%d" % k), _make_cfg())
        fp.run(ctx, rows)
        res.append({u: (int(h), int(p)) for s in _shards(tmp_path / ("o%d" % k))
                    for u, h, p in zip(json.loads((s / "uids.json").read_text()), np.load(s / "simhash64.npy"),
                                       np.load(s / "phash64.npy"))})
    assert res[0] == res[1] and len(res[0]) == 12
    s = _shards(tmp_path / "o0")[0]
    emb = np.load(s / "emb_clip_b32.npy")[0]
    from parameters.edge.features import simhash64
    assert simhash64(emb.astype(np.float64)) == int(np.load(s / "simhash64.npy")[0])


def test_bitmap_roundtrip_and_stored_kept(tmp_path):
    kept = [0, 5, 8, 511, 100]
    bm = fp.pack_bitmap(kept)
    assert bm.shape == (64,) and bm.dtype == np.uint8
    assert fp.unpack_bitmap(bm).tolist() == sorted(kept)
    with pytest.raises(ValueError):
        fp.pack_bitmap([512])
    root, rows = _dataset(tmp_path, n=3)
    ctx, _ = _ctx(root, tmp_path / "o", _make_cfg(shard_size=10))
    fp.run(ctx, rows)
    s = _shards(tmp_path / "o")[0]
    bms = np.load(s / "kept_bitmap_m0.25.npy")
    assert bms.shape == (3, 64) and all(len(fp.unpack_bitmap(b)) > 0 for b in bms)


def test_letterbox_params():
    img = Image.new("RGB", (100, 200), (255, 0, 0))
    arr, p = fp.letterbox(img, (256, 512))
    assert arr.shape == (256, 512, 3)
    w, h, nw, nh, pl, pt, sc = p
    assert (nh, nw) == (256, 128) and pl == (512 - 128) // 2 and pt == 0 and sc == pytest.approx(1.28)
    assert arr[:, :int(pl)].sum() == 0


def test_bad_image_skipped_and_reported(tmp_path):
    root, rows = _dataset(tmp_path, n=8, bad=(3,))
    ctx, _ = _ctx(root, tmp_path / "o", _make_cfg())
    summ = fp.run(ctx, rows)
    assert summ["counts"]["ok"] == 7 and summ["n_failures"] == 1
    assert summ["failures"][0]["uid"] == "crisismmd:i03" and summ["failures"][0]["stage"] == "load"
    assert (tmp_path / "o/summary.json").exists()


def test_missing_encoder_only_skips_that_encoder(tmp_path):
    root, rows = _dataset(tmp_path, n=6)
    ctx, _ = _ctx(root, tmp_path / "o", _make_cfg(), fail=("dinov2_small",))
    summ = fp.run(ctx, rows)
    assert "emb_dinov2_small" in summ["skipped"] and summ["counts"]["ok"] == 6
    s = _shards(tmp_path / "o")[0]
    assert (s / "emb_clip_b32.npy").exists() and not (s / "emb_dinov2_small.npy").exists()


def test_timing_rows_per_stage(tmp_path):
    root, rows = _dataset(tmp_path, n=6)
    ctx, _ = _ctx(root, tmp_path / "o", _make_cfg(shard_size=10))
    fp.run(ctx, rows)
    t = [json.loads(l) for l in (_shards(tmp_path / "o")[0] / "timings.jsonl").read_text().splitlines()]
    stages = {r["stage"] for r in t}
    expect = {"load", "embed_clip_b32", "embed_dinov2_small", "phash", "sqgan_segment", "sqgan_encode_m0.25",
              "sqgan_encode_m0.55", "decode_m0.25", "decode_m0.55", "recon_save"}
    assert expect <= stages
    per = [r for r in t if r["stage"] == "phash"]
    assert len(per) == 6 and sum(r["warmup"] for r in per) == 2 and all(r["ms"] >= 0 for r in per)
    assert fp.summarize(tmp_path / "o")["stage_mean_ms"]["phash"] >= 0


def test_sqgan_and_decode_flags_off(tmp_path):
    root, rows = _dataset(tmp_path, n=4)
    cfg = _make_cfg(shard_size=10, sqgan={"enabled": False})
    ctx, fake = _ctx(root, tmp_path / "o", cfg)
    fp.run(ctx, rows)
    s = _shards(tmp_path / "o")[0]
    assert fake.encodes == 0 and not list(s.glob("payloads_*")) and (s / "emb_clip_b32.npy").exists()


def test_limit_and_stable_sharding():
    rows = [{"uid": "u%d" % i} for i in range(50)]
    a, b = fp.plan_shards(rows, 10), fp.plan_shards(list(reversed(rows)), 10)
    assert {k: [r["uid"] for r in v] for k, v in a.items()} == {k: [r["uid"] for r in v] for k, v in b.items()}
    assert sum(len(v) for v in fp.plan_shards(rows, 500, limit=20).values()) == 20


def test_export_grid_provisional_splits(tmp_path):
    root, rows = _dataset(tmp_path)
    ctx, _ = _ctx(root, tmp_path / "o", _make_cfg())
    fp.run(ctx, rows)
    msgs = []
    lp, sp = fp.export_grid_inputs(tmp_path / "o", rows, tmp_path / "o/grid", None, log=msgs.append)
    assert any("PROVISIONAL" in m for m in msgs)
    z = np.load(tmp_path / "o/grid/emb_clip_b32.npz")
    assert z["emb"].shape == (12, 512) and (tmp_path / "o/grid/emb_clip_b32.done").exists()
    splits = json.loads(sp.read_text())
    assert set(splits) == set(z["image_path"].tolist())


def test_compute_profile_tiny_module(tmp_path):
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, 5))
    x = torch.rand(1, 3, 8, 8)
    r = compute_profile.profile_module(net, (x,), batch_sizes=(1, 2), repeats=2, warmup=1)
    assert r["params"] == sum(p.numel() for p in net.parameters()) and r["flops"] > 0
    assert len(r["latency_ms"]["2"]) == 2
    assert compute_profile.hook_flops(net, (x,), {}) > 0
    prof = compute_profile.profile_all({"batch_sizes": [1], "repeats": 1, "warmup": 0, "phash_samples": 2},
                                       {"tiny": (net, (x,), {})}, tmp_path / "compute_profile.json",
                                       phash_images=[np.random.default_rng(0).random((64, 64)) * 255] * 2)
    assert prof["tiny"]["flops"] > 0 and prof["phash"]["cpu_ms_mean"] >= 0
    assert (tmp_path / "compute_profile.json").exists()


def test_find_data_root(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "labels_raw.jsonl").write_text("")
    assert fp.find_data_root("labels_raw.jsonl", tmp_path, 3) == tmp_path / "a" / "b"
    with pytest.raises(FileNotFoundError, match="no folder"):
        fp.find_data_root("index.jsonl", tmp_path, 3)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "labels_raw.jsonl").write_text("")
    with pytest.raises(FileNotFoundError, match="several"):
        fp.find_data_root("labels_raw.jsonl", tmp_path, 3)


# ---------------------------------------------------------------- stage isolation

class RaisingSegSQ(FakeSQ):
    def segment(self, arr):
        raise RuntimeError("segmenter exploded")


class RaisingEncodeSQ(FakeSQ):
    def encode(self, arr, handle, m):
        if m == 0.55:
            raise ValueError("encode m0.55 failed")
        return super().encode(arr, handle, m)


def _ctx_sq(root, out, cfg, sq, fail_enc=None):
    def emb_factory(name):
        def fn(images):
            if name == fail_enc:
                raise RuntimeError("encoder boom")
            return np.stack([_vec(name, im) for im in images])
        return fn
    backends = fp.Backends(emb_factory, lambda: sq, clock=(lambda it=iter(__import__("itertools").count(0, 0.002)): next(it)))
    return fp.Context(cfg, load_config("grid"), root, out, backends, log=lambda *a: None)


def test_failing_encoder_keeps_image_ok_and_flags(tmp_path):
    root, rows = _dataset(tmp_path, n=6)
    out = tmp_path / "o"
    summ = fp.run(_ctx_sq(root, out, _make_cfg(shard_size=10), FakeSQ(), fail_enc="dinov2_small"), rows)
    assert summ["counts"]["ok"] == 6 and summ["n_failures"] == 0
    s = _shards(out)[0]
    assert not np.load(s / "emb_dinov2_small_ok.npy").any() and np.load(s / "emb_clip_b32_ok.npy").all()
    assert not (s / "emb_dinov2_small.npy").exists() and (s / "emb_clip_b32.npy").exists()
    assert (s / "phash64.npy").exists() and np.load(s / "phash_ok.npy").all()
    assert np.load(s / "simhash_ok.npy").all() and (s / "class_grid.npy").exists() and np.load(s / "sqgan_ok.npy").all()
    st = json.loads((s / "failures.json").read_text())
    assert any(r["stage"] == "embed_dinov2_small" and r["error_type"] == "RuntimeError" and r["count"] == 6
               and "encoder boom" in r["message"] for r in st)
    assert any(r["stage"] == "embed_dinov2_small" and r["count"] == 6 for r in summ["stage_failures"])
    assert summ["n_stage_failures"] == 6


def test_failing_simhash_encoder_sets_simhash_flag_false(tmp_path):
    root, rows = _dataset(tmp_path, n=4)
    out = tmp_path / "o"
    summ = fp.run(_ctx_sq(root, out, _make_cfg(shard_size=10), FakeSQ(), fail_enc="clip_b32"), rows)
    s = _shards(out)[0]
    n = len(json.loads((s / "uids.json").read_text()))
    assert summ["counts"]["ok"] == 4 and (s / "phash64.npy").exists()
    assert not np.load(s / "simhash_ok.npy").any() and len(np.load(s / "simhash64.npy")) == n
    assert np.load(s / "emb_dinov2_small_ok.npy").all()


def test_failing_sqgan_segment_keeps_other_stages(tmp_path):
    root, rows = _dataset(tmp_path, n=6)
    out = tmp_path / "o"
    summ = fp.run(_ctx_sq(root, out, _make_cfg(shard_size=10), RaisingSegSQ()), rows)
    assert summ["counts"]["ok"] == 6 and summ["n_failures"] == 0
    s = _shards(out)[0]
    assert (s / "done.json").exists() and (s / "emb_clip_b32.npy").exists() and (s / "phash64.npy").exists()
    assert not np.load(s / "sqgan_ok.npy").any() and np.load(s / "sqgan_ok.npy").shape == (6, 2)
    assert np.load(s / "class_grid.npy").shape == (6, 16, 32) and not np.load(s / "class_grid.npy").any()
    assert any(r["stage"] == "sqgan_segment" and r["error_type"] == "RuntimeError" and r["count"] == 6
               for r in summ["stage_failures"])
    stages = {json.loads(l)["stage"] for l in (s / "timings.jsonl").read_text().splitlines()}
    assert {"load", "phash", "embed_clip_b32"} <= stages


def test_failing_single_m_encode_flags_only_that_m(tmp_path):
    root, rows = _dataset(tmp_path, n=4)
    out = tmp_path / "o"
    summ = fp.run(_ctx_sq(root, out, _make_cfg(shard_size=10), RaisingEncodeSQ()), rows)
    s = _shards(out)[0]
    ok = np.load(s / "sqgan_ok.npy")
    assert ok[:, 0].all() and not ok[:, 1].any()
    assert any(r["stage"] == "sqgan_encode_m0.55" and r["error_type"] == "ValueError" for r in summ["stage_failures"])
    assert np.load(s / "decode_ok_m0.25.npy").all() and not np.load(s / "decode_ok_m0.55.npy").any()
    assert summ["counts"]["ok"] == 4


def test_failing_decode_is_a_stage_failure(tmp_path):
    class BadDecode(FakeSQ):
        def decode(self, payload, m):
            raise OSError("no decode")
    root, rows = _dataset(tmp_path, n=3)
    summ = fp.run(_ctx_sq(root, tmp_path / "o", _make_cfg(shard_size=10), BadDecode()), rows)
    assert summ["counts"]["ok"] == 3
    assert {r["stage"] for r in summ["stage_failures"]} == {"decode_m0.25", "decode_m0.55"}


def test_export_grid_skips_failed_rows(tmp_path):
    root, rows = _dataset(tmp_path, n=6)
    out = tmp_path / "o"
    fp.run(_ctx_sq(root, out, _make_cfg(shard_size=10), FakeSQ(), fail_enc="dinov2_small"), rows)
    fp.export_grid_inputs(out, rows, out / "grid", None, log=lambda *a: None)
    assert (out / "grid/emb_clip_b32.npz").exists() and not (out / "grid/emb_dinov2_small.npz").exists()
    assert not (out / "grid/emb_clip_b32_ok.npz").exists()


# ---------------------------------------------------------------- backends: clear errors

def test_restore_flat_bundle_missing_manifest_message(tmp_path):
    from Codec.edge import feature_pass_backends as be
    flat = tmp_path / "flat"
    flat.mkdir()
    for i in range(20):
        (flat / ("f%02d.bin" % i)).write_text("x")
    (flat / "bundle_manifest.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as e:
        be.restore_flat_bundle(flat, tmp_path / "dest")
    msg = str(e.value)
    assert str(flat) in msg and "*flat_manifest.json" in msg and "bundle_manifest.json" in msg
    assert "f00.bin" in msg and "f19.bin" not in msg                 # at most 15 names listed
    with pytest.raises(FileNotFoundError, match="does not exist"):
        be.restore_flat_bundle(tmp_path / "nope", tmp_path / "dest")


def test_restore_flat_bundle_ok(tmp_path):
    from Codec.edge import feature_pass_backends as be
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "scheduler_sqgan_flat_manifest.json").write_text(json.dumps(
        {"bundle_manifest": "bm.json", "files": {"a/b.pt": "w.pt"}}))
    (flat / "bm.json").write_text("{}")
    (flat / "w.pt").write_text("w")
    b = be.restore_flat_bundle(flat, tmp_path / "dest")
    assert (b / "bundle.json").exists() and (b / "a/b.pt").read_text() == "w"


def test_find_source_root_message(tmp_path):
    from Codec.edge import feature_pass_backends as be
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__init__.py").write_text("")
    (tmp_path / "readme.txt").write_text("")
    with pytest.raises(FileNotFoundError) as e:
        be.find_source_root(tmp_path)
    msg = str(e.value)
    assert str(tmp_path) in msg and "src/sqgan.py" in msg and "readme.txt" in msg


def test_notebook_install_cell_has_lpips():
    from scripts.edge import make_run_all_notebook as mk
    nb, _ = mk.build()
    install = ["".join(c["source"]) for c in nb["cells"] if "pip install" in "".join(c["source"])]
    assert install and "lpips" in install[0]
