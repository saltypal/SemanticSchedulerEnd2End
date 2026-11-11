"""ONE per-image feature pass that caches everything training needs later (no GPU, no re-timing).

Each image is decoded from disk once, inside one loop over batches, and the enabled stages run on it:
  1. embeddings of every encoder in config/edge/grid.yaml (float16, ``emb_<encoder>.npy``);
  2. SimHash64 of the ``hashes.simhash_encoder`` float16 embedding and pHash64 (uint64);
  3. SQ-GAN: letterbox to 256x512, teacher segmentation, payload bytes per m (``payloads_m<m>.bin`` +
     ``offsets_m<m>.npy``), kept-token bitmap (64 bytes, bit i = token i kept), 16x32 majority-vote class
     grid, 13-float STS, and the pre-mask encoder grid pooled mean+max (512 float16);
  4. decode of each payload -> JPEG q95 reconstruction at 256x512 plus decode ms;
  5. per image / per stage wall ms in ``timings.jsonl`` (first N images of a stage flagged warm-up).

Letterbox mapping: scale = min(512/w, 256/h); the image is resized to (round(w*scale), round(h*scale)) with
LANCZOS and centred on a zero canvas. ``letterbox.npy`` rows hold [orig_w, orig_h, new_w, new_h, pad_left,
pad_top, scale] so original coordinates can be recovered.

Shards: bucket = sha256(uid) % n_shards, n_shards = ceil(N / shard_size); a shard is written to
``shard_XXXX.partial`` then renamed, ``done.json`` is the last file, finished shards are skipped, and
``--part i/n`` lets several runs split the shards. One unreadable image never stops the run: it is listed in
``failures``; any other stage error is a stage-level failure (``*_ok.npy`` flags + zero rows). Models are injected through ``Backends`` (tests use fakes); the real ones are in
``feature_pass_backends``. Reuses features.simhash64/phash64, sts.compute_sts, embedding_grid.disk_guard and
splits.assign_event_splits.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from Evaluation.edge.edge_config import load_config
from parameters.edge import features, sts as sts_mod
from parameters.edge.embedding_grid import disk_guard

N_TOKENS = 512
SEGMAP_FILE = "segmap_u8.npz"     # key "seg": (N,256,512) uint8 teacher class map, 0..19, zeros where sqgan failed


# ------------------------------------------------------------------ small pure helpers

def stable_hash(text):
    """Stable 64-bit integer from sha256 (process independent, unlike hash())."""
    return int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:8], "big")


def uid_file_stem(uid):
    return hashlib.sha1(str(uid).encode()).hexdigest()[:16]


def m_tag(m):
    return "m%g" % float(m)


def pack_bitmap(kept_idx, n=N_TOKENS):
    """Kept token indices -> (n/8,) uint8 bitmap (little bit order: token i -> byte i//8, bit i%8)."""
    idx = np.unique(np.asarray(list(kept_idx), dtype=np.int64))
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError("kept index outside 0..%d" % (n - 1))
    bits = np.zeros(n, dtype=np.uint8)
    bits[idx] = 1
    return np.packbits(bits, bitorder="little")


def unpack_bitmap(bitmap, n=N_TOKENS):
    """Inverse of pack_bitmap: sorted kept indices."""
    return np.flatnonzero(np.unpackbits(np.asarray(bitmap, dtype=np.uint8), bitorder="little")[:n]).astype(np.int64)


def letterbox(img, hw=(256, 512), pad_value=0):
    """PIL RGB image -> (uint8 HxWx3, params[7]); see module docstring for the mapping."""
    from PIL import Image
    h_t, w_t = hw
    w, h = img.size
    scale = min(w_t / w, h_t / h)
    nw, nh = max(1, min(w_t, round(w * scale))), max(1, min(h_t, round(h * scale)))
    canvas = np.full((h_t, w_t, 3), pad_value, dtype=np.uint8)
    pl, pt = (w_t - nw) // 2, (h_t - nh) // 2
    canvas[pt:pt + nh, pl:pl + nw] = np.asarray(img.convert("RGB").resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
    return canvas, np.array([w, h, nw, nh, pl, pt, scale], dtype=np.float32)


def pool_prefeatures(grid):
    """(C, H, W) pre-mask encoder grid -> (2C,) float16: spatial mean then spatial max."""
    g = np.asarray(grid, dtype=np.float32)
    if g.ndim != 3:
        raise ValueError("pre-mask grid must be (C,H,W), got %r" % (g.shape,))
    return np.concatenate([g.mean(axis=(1, 2)), g.max(axis=(1, 2))]).astype(np.float16)


def u64(values):
    return np.array([int(v) for v in values], dtype=np.uint64)


def plan_shards(rows, shard_size, limit=None):
    """{shard_idx: rows} by stable hash of ``uid``; ``limit`` keeps the first N rows in hash order."""
    rows = sorted(rows, key=lambda r: (stable_hash(r["uid"]), r["uid"]))
    if limit:
        rows = rows[:int(limit)]
    n_shards = max(1, math.ceil(len(rows) / int(shard_size)))
    plan = {}
    for r in rows:
        plan.setdefault(stable_hash(r["uid"]) % n_shards, []).append(r)
    return plan


def find_data_root(marker, base="/kaggle/input", depth=3):
    """Folder under ``base`` (up to ``depth`` levels) that contains ``marker``; error if none or several."""
    base = Path(base)
    hits = sorted({p.parent for d in range(depth + 1)
                   for p in base.glob("/".join(["*"] * d + [marker]))}) if base.is_dir() else []
    if not hits:
        raise FileNotFoundError("no folder containing %s under %s (depth %d); pass the root explicitly" % (marker, base, depth))
    if len(hits) > 1:
        raise FileNotFoundError("several folders contain %s under %s: %s; pass the root explicitly"
                                % (marker, base, [str(h) for h in hits]))
    return hits[0]


def read_labels_raw(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "uid" not in r or "file" not in r:
                raise ValueError("labels row lacks uid/file: %r" % (list(r)[:6],))
            rows.append(r)
    return rows


# ------------------------------------------------------------------ timing

class StageTimer:
    """Per-stage wall-clock rows. ``sync`` is torch.cuda.synchronize for GPU stages; both injectable."""

    def __init__(self, clock=time.perf_counter, sync=None, warmup=3):
        self.clock, self.sync, self.warmup = clock, sync or (lambda: None), int(warmup)
        self.rows, self.seen = [], Counter()

    def run(self, stage, uids, fn, gpu=False):
        """Run ``fn()``; log its wall ms split evenly over ``uids``. Re-raises fn's exception untimed."""
        if gpu:
            self.sync()
        t0 = self.clock()
        out = fn()
        if gpu:
            self.sync()
        per = (self.clock() - t0) * 1000.0 / max(len(uids), 1)
        for u in uids:
            self.rows.append({"uid": u, "stage": stage, "ms": per, "batch_size": len(uids),
                              "warmup": bool(self.seen[stage] < self.warmup)})
            self.seen[stage] += 1
        return out

    def drain(self):
        rows, self.rows = self.rows, []
        return rows


# ------------------------------------------------------------------ backends / context

@dataclass
class Backends:
    """Injected model access. ``embedder_factory(name)`` -> fn(list[PIL]) -> (n,dim); ``sqgan_factory()`` ->
    object with segment(arr)->(class_map, handle), encode(arr, handle, m)->{payload, kept, features},
    decode(payload, m)->uint8 HxWx3. Either may raise; the encoder/stage is then skipped and logged."""
    embedder_factory: Optional[Callable] = None
    sqgan_factory: Optional[Callable] = None
    clock: Callable = time.perf_counter
    sync: Optional[Callable] = None
    gpu_name: Optional[Callable] = None     # real runs pass feature_pass_backends.gpu_name


@dataclass
class Context:
    cfg: dict
    grid_cfg: dict
    dataset_root: Path          # images root: row['file'] is relative to it
    out_dir: Path
    backends: Backends
    log: Callable = print
    timer: StageTimer = None
    skipped: dict = field(default_factory=dict)
    _emb: dict = field(default_factory=dict)
    _sq: object = None
    _sq_tried: bool = False

    def __post_init__(self):
        self.timer = self.timer or StageTimer(self.backends.clock, self.backends.sync,
                                              self.cfg["shard"]["warmup_images"])

    def encoder_names(self):
        sel = self.cfg["embeddings"]["encoders"]
        names = [e["name"] for e in self.grid_cfg["encoders"]]
        if sel != "all":
            unknown = [n for n in sel if n not in names]
            if unknown:
                raise ValueError("unknown encoders in feature_pass.yaml: %r" % unknown)
            names = [n for n in names if n in sel]
        return names if self.cfg["embeddings"]["enabled"] else []

    def embedder(self, name):
        """Lazy; a failing load skips only that encoder (logged once, listed in summary)."""
        if name not in self._emb:
            try:
                self._emb[name] = self.backends.embedder_factory(name)
            except Exception as exc:
                self._emb[name] = None
                self.skipped["emb_" + name] = repr(exc)
                self.log("[skip] encoder %s: %r" % (name, exc))
        return self._emb[name]

    def sqgan(self):
        if not self.cfg["sqgan"]["enabled"]:
            return None
        if not self._sq_tried:
            self._sq_tried = True
            try:
                self._sq = self.backends.sqgan_factory()
            except Exception as exc:
                self.skipped["sqgan"] = repr(exc)
                self.log("[skip] SQ-GAN stage: %r" % (exc,))
        return self._sq


# ------------------------------------------------------------------ shard processing

def _load_image(root, row):
    from PIL import Image
    with Image.open(Path(root) / row["file"]) as im:
        im.load()
        return im.convert("RGB")


def _stage_fail(sfail, stage, uid, exc):
    """Record a stage-level failure (image stays ok): aggregated by (stage, error type, message)."""
    key = (stage, type(exc).__name__, str(exc)[:300])
    e = sfail.setdefault(key, {"stage": stage, "error_type": key[1], "message": key[2], "count": 0, "example_uids": []})
    e["count"] += 1
    if len(e["example_uids"]) < 5:
        e["example_uids"].append(uid)


def merge_stage_failures(*lists):
    """Merge stage-failure rows (as written to failures.json) by (stage, error_type, message), summing counts."""
    out = {}
    for rows in lists:
        for r in rows:
            k = (r["stage"], r["error_type"], r["message"])
            e = out.setdefault(k, dict(r, count=0, example_uids=[]))
            e["count"] += r["count"]
            e["example_uids"] = (e["example_uids"] + list(r.get("example_uids", [])))[:5]
    return list(out.values())


def _embed_batch(ctx, name, fn, items):
    """Embeddings for ``items`` [(uid, img)]; on a batch error retry per image.

    Returns ({uid: vec|None}, {uid: exception}) - a failed image only misses this encoder's embedding."""
    uids = [u for u, _ in items]
    try:
        arr = ctx.timer.run("embed_" + name, uids, lambda: np.asarray(fn([im for _, im in items])), gpu=True)
        return {u: arr[i] for i, u in enumerate(uids)}, {}
    except Exception:
        out, errs = {}, {}
        for u, im in items:
            try:
                out[u] = ctx.timer.run("embed_" + name, [u], lambda im=im: np.asarray(fn([im]))[0], gpu=True)
            except Exception as exc:
                out[u], errs[u] = None, exc
        return out, errs


def _gray_thumb(img, size):
    from PIL import Image
    return np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)


def _sq_one(ctx, sq, uid, img, m_levels, hw, grid, sfail):
    """SQ-GAN segment + encode (per m) for one image. Never raises: each stage failure is recorded in ``sfail``.

    Returns None when segmentation failed, else a dict whose ``m`` holds only the levels that encoded."""
    try:
        arr, lb = letterbox(img, hw, ctx.cfg["sqgan"]["pad_value"])
        cmap, handle = ctx.timer.run("sqgan_segment", [uid], lambda: sq.segment(arr), gpu=True)
        cls = sts_mod.majority_vote_grid(np.asarray(cmap), hw[0] // grid[0])
    except Exception as exc:
        _stage_fail(sfail, "sqgan_segment", uid, exc)
        return None
    res = {"letterbox": lb, "class_grid": cls.astype(np.uint8), "segmap": np.asarray(cmap).astype(np.uint8),
           "m": {}, "prefeat": None}
    for m in m_levels:
        try:
            enc = ctx.timer.run("sqgan_encode_" + m_tag(m), [uid], lambda m=m: sq.encode(arr, handle, m), gpu=True)
            kept = np.unique(np.asarray(enc["kept"], dtype=np.int64))
            bitmap = pack_bitmap(kept, grid[0] * grid[1])
            sts13 = np.asarray(sts_mod.compute_sts(cls, kept, tuple(grid)), dtype=np.float32)
            prefeat = pool_prefeatures(enc["features"]) if res["prefeat"] is None and enc.get("features") is not None else None
            entry = {"payload": bytes(enc["payload"]), "bitmap": bitmap, "sts": sts13, "recon": None, "orig": arr}
        except Exception as exc:
            _stage_fail(sfail, "sqgan_encode_" + m_tag(m), uid, exc)
            continue
        if prefeat is not None:                       # encoder is m-independent: first m's grid
            res["prefeat"] = prefeat
        res["m"][m] = entry
    return res


def _flush_arrays(shard_dir, uids, per_uid, cfg, enc_names, sq_enabled):
    """Write all aligned arrays for the readable images (uids order). Failed stages get a False flag + zero rows."""
    n = len(uids)
    info = {}
    if n == 0:
        return info
    for name in enc_names:
        ok = np.array([name in per_uid[u]["emb"] for u in uids])
        np.save(shard_dir / ("emb_%s_ok.npy" % name), ok)
        if ok.any():
            first = next(per_uid[u]["emb"][name] for u in uids if name in per_uid[u]["emb"])
            zero = np.zeros(first.shape, np.float16)
            np.save(shard_dir / ("emb_%s.npy" % name),
                    np.stack([per_uid[u]["emb"][name] if name in per_uid[u]["emb"] else zero for u in uids]).astype(np.float16))
    hashes_on = cfg["hashes"].get("enabled", True)
    if hashes_on:
        ph = [per_uid[u]["phash"] for u in uids]
        np.save(shard_dir / "phash64.npy", u64([0 if p is None else p for p in ph]))
        np.save(shard_dir / "phash_ok.npy", np.array([p is not None for p in ph]))
    sim = cfg["hashes"]["simhash_encoder"]
    if hashes_on and sim in enc_names:
        sh = [per_uid[u]["simhash"] for u in uids]
        np.save(shard_dir / "simhash64.npy", u64([0 if p is None else p for p in sh]))
        np.save(shard_dir / "simhash_ok.npy", np.array([p is not None for p in sh]))
    if not sq_enabled:
        return info
    m_levels = cfg["sqgan"]["m_levels"]
    ok = np.array([[bool(per_uid[u]["sq"] and m in per_uid[u]["sq"]["m"]) for m in m_levels] for u in uids])
    np.save(shard_dir / "sqgan_ok.npy", ok.reshape(n, len(m_levels)))
    grid = cfg["sqgan"]["token_grid"]
    z = lambda u: per_uid[u]["sq"]
    np.save(shard_dir / "letterbox.npy", np.stack([z(u)["letterbox"] if z(u) else np.zeros(7, np.float32) for u in uids]))
    np.save(shard_dir / "class_grid.npy", np.stack(
        [z(u)["class_grid"] if z(u) else np.zeros(tuple(grid), np.uint8) for u in uids]))
    seg = np.zeros((n,) + tuple(cfg["sqgan"]["input_hw"]), np.uint8)       # rows of failed segmentation stay zero
    for i, u in enumerate(uids):
        if z(u):
            seg[i] = z(u)["segmap"]
    np.savez_compressed(shard_dir / SEGMAP_FILE, seg=seg)
    info["segmap_shape"] = list(seg.shape)
    info["segmap_mb"] = round((shard_dir / SEGMAP_FILE).stat().st_size / 1e6, 4)
    dim = next((z(u)["prefeat"].size for u in uids if z(u) and z(u)["prefeat"] is not None), 0)
    np.save(shard_dir / "prefeat_meanmax.npy", np.stack(
        [z(u)["prefeat"] if z(u) and z(u)["prefeat"] is not None else np.zeros(dim, np.float16) for u in uids])
        if dim else np.zeros((n, 0), np.float16))
    for m in m_levels:
        tag, payloads, offsets = m_tag(m), [], [0]
        for u in uids:
            p = z(u)["m"][m]["payload"] if z(u) and m in z(u)["m"] else b""
            payloads.append(p)
            offsets.append(offsets[-1] + len(p))
        (shard_dir / ("payloads_%s.bin" % tag)).write_bytes(b"".join(payloads))
        np.save(shard_dir / ("offsets_%s.npy" % tag), np.array(offsets, dtype=np.int64))
        np.save(shard_dir / ("kept_bitmap_%s.npy" % tag), np.stack(
            [z(u)["m"][m]["bitmap"] if z(u) and m in z(u)["m"] else np.zeros(grid[0] * grid[1] // 8, np.uint8)
             for u in uids]))
        np.save(shard_dir / ("sts_%s.npy" % tag), np.stack(
            [z(u)["m"][m]["sts"] if z(u) and m in z(u)["m"] else np.zeros(13, np.float32) for u in uids]))
        if cfg["decode"]["enabled"]:
            np.save(shard_dir / ("decode_ok_%s.npy" % tag), np.array(
                [bool(z(u) and m in z(u)["m"] and z(u)["m"][m]["recon"]) for u in uids]))
            if cfg["decode"].get("quality"):          # NaN where decode or the quality stage failed
                for key, name in (("lpips", "lpips"), ("miou", "miou")):
                    np.save(shard_dir / ("%s_%s.npy" % (name, tag)), np.array(
                        [z(u)["m"][m].get(key, np.nan) if z(u) and m in z(u)["m"] else np.nan for u in uids], np.float32))
    return info


def stages_enabled(cfg):
    """Stage names this run is configured to execute (embeddings, hashes, sqgan, decode)."""
    out = []
    if cfg["embeddings"]["enabled"]:
        out.append("embeddings")
    if cfg["hashes"].get("enabled", True):
        out.append("hashes")
    if cfg["sqgan"]["enabled"]:
        out.append("sqgan")
        if cfg["decode"]["enabled"]:
            out.append("decode")
    return out


def stages_skipped(ctx, enc_names, enc_ok):
    """Stages that could not run at all: SQ-GAN whose backend failed to build, embeddings with no encoder loaded."""
    out = []
    if enc_names and not enc_ok and all(("emb_" + n) in ctx.skipped for n in enc_names):
        out.append("embeddings")
    if "sqgan" in ctx.skipped:
        out.append("sqgan")
        if ctx.cfg["decode"]["enabled"]:
            out.append("decode")
    return out


def process_shard(ctx, idx, rows):
    """Run every enabled stage over one shard and write it atomically. Returns the done.json dict.

    An image fails only if it cannot be decoded. Every other stage runs in its own try/except: an error becomes a
    stage-level failure (failures.json / summary.json) and a False flag with zero placeholder rows."""
    cfg = ctx.cfg
    final = ctx.out_dir / "shards" / ("shard_%04d" % idx)
    partial = final.with_name(final.name + ".partial")
    if (final / "done.json").exists():
        return json.loads((final / "done.json").read_text(encoding="utf-8"))
    shutil.rmtree(partial, ignore_errors=True)
    disk_guard(cfg["shard"]["min_free_gb"], ctx.out_dir)
    partial.mkdir(parents=True)
    enc_names = ctx.encoder_names()
    sim_name = cfg["hashes"]["simhash_encoder"]
    hashes_on = cfg["hashes"].get("enabled", True)
    m_levels, hw, grid = cfg["sqgan"]["m_levels"], cfg["sqgan"]["input_hw"], cfg["sqgan"]["token_grid"]
    t_shard = time.time()
    per_uid, order, failures, sfail = {}, [], [], {}
    bs = int(cfg["shard"]["batch_size"])
    for b0 in range(0, len(rows), bs):
        items = []
        for r in rows[b0:b0 + bs]:                       # decode each image from disk exactly once
            try:
                items.append((r["uid"], ctx.timer.run("load", [r["uid"]], lambda r=r: _load_image(ctx.dataset_root, r))))
            except Exception as exc:
                failures.append({"uid": r["uid"], "file": r.get("file"), "stage": "load", "error": repr(exc)})
        if not items:
            continue
        rec = {u: {"emb": {}, "phash": None, "simhash": None, "sq": None} for u, _ in items}
        for name in enc_names:
            fn = ctx.embedder(name)
            if fn is None:
                continue
            vecs, errs = _embed_batch(ctx, name, fn, items)
            for u, vec in vecs.items():
                if vec is not None:
                    rec[u]["emb"][name] = np.asarray(vec, dtype=np.float32).astype(np.float16)
            for u, exc in errs.items():
                _stage_fail(sfail, "embed_" + name, u, exc)
        for u, im in items if hashes_on else ():
            try:
                thumb = _gray_thumb(im, cfg["hashes"]["phash_thumb"])
                rec[u]["phash"] = ctx.timer.run("phash", [u], lambda t=thumb: features.phash64(t))
            except Exception as exc:
                _stage_fail(sfail, "phash", u, exc)
            if sim_name in rec[u]["emb"]:
                try:
                    rec[u]["simhash"] = ctx.timer.run(
                        "simhash", [u], lambda u=u: features.simhash64(rec[u]["emb"][sim_name].astype(np.float64)))
                except Exception as exc:
                    _stage_fail(sfail, "simhash", u, exc)
        sq = ctx.sqgan()
        if sq is not None:
            for u, im in items:
                rec[u]["sq"] = _sq_one(ctx, sq, u, im, m_levels, hw, grid, sfail)
                if rec[u]["sq"] is not None and cfg["decode"]["enabled"]:
                    for m in m_levels:
                        if m in rec[u]["sq"]["m"]:
                            _decode_one(ctx, sq, u, m, rec[u]["sq"]["m"][m], partial, sfail)
        for u, _ in items:
            per_uid[u] = rec[u]
            order.append(u)
    sq = ctx.sqgan()
    if sq is not None and cfg["decode"]["enabled"] and cfg["decode"].get("latency"):
        m0 = m_levels[0]
        _latency_shard(ctx, sq, partial, [per_uid[u]["sq"]["m"][m0]["payload"] for u in order
                                          if per_uid[u]["sq"] and m0 in per_uid[u]["sq"]["m"]][:8], m0, sfail)
    info = _flush_arrays(partial, order, per_uid, cfg, enc_names, bool(cfg["sqgan"]["enabled"]))
    (partial / "uids.json").write_text(json.dumps(order), encoding="utf-8")
    with open(partial / "timings.jsonl", "w", encoding="utf-8") as fh:
        for row in ctx.timer.drain():
            fh.write(json.dumps(row) + "\n")
    stage_rows = list(sfail.values())
    (partial / "failures.json").write_text(json.dumps(failures + stage_rows, indent=1), encoding="utf-8")
    enc_ok = [n for n in enc_names if any(n in per_uid[u]["emb"] for u in order)]
    done = {"shard": idx, "n_requested": len(rows), "n_ok": len(order), "n_failed": len(failures),
            "n_stage_failures": sum(r["count"] for r in stage_rows), "encoders": enc_ok,
            "skipped": dict(ctx.skipped), "m_levels": m_levels if ctx.sqgan() is not None else [],
            "stages_enabled": stages_enabled(cfg), "stages_skipped": stages_skipped(ctx, enc_names, enc_ok),
            "seconds": time.time() - t_shard}
    if "segmap_shape" in info:
        done.update(segmap_saved=True, segmap_shape=info["segmap_shape"], segmap_mb=info["segmap_mb"])
    (partial / "done.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
    os.replace(partial, final)
    return done


def _decode_one(ctx, sq, uid, m, entry, shard_dir, sfail):
    """Decode a payload, save the recon JPEG q95, mark entry['recon'] with the relative path (stages isolated)."""
    from PIL import Image
    try:
        arr = ctx.timer.run("decode_" + m_tag(m), [uid], lambda: sq.decode(entry["payload"], m), gpu=True)
    except Exception as exc:
        _stage_fail(sfail, "decode_" + m_tag(m), uid, exc)
        entry.pop("orig", None)
        return
    orig = entry.pop("orig", None)
    if ctx.cfg["decode"].get("quality") and orig is not None and hasattr(sq, "quality"):   # LPIPS + recon mIoU (D6)
        try:
            q = ctx.timer.run("quality_" + m_tag(m), [uid], lambda: sq.quality(orig, arr), gpu=True)
            entry["lpips"], entry["miou"] = float(q["lpips"]), float(q["miou"])
        except Exception as exc:
            _stage_fail(sfail, "quality_" + m_tag(m), uid, exc)
    try:
        d = shard_dir / ("recon_%s" % m_tag(m))
        d.mkdir(exist_ok=True)
        rel = "recon_%s/%s.jpg" % (m_tag(m), uid_file_stem(uid))
        ctx.timer.run("recon_save", [uid], lambda: Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(
            shard_dir / rel, format="JPEG", quality=int(ctx.cfg["decode"]["jpeg_quality"])))
        entry["recon"] = rel
    except Exception as exc:
        _stage_fail(sfail, "recon_save", uid, exc)


def _latency_shard(ctx, sq, shard_dir, payloads, m, sfail):
    """Decode latency samples (seconds) per batch size, written to ``latency_samples.json`` in the shard.

    The SQ-GAN decoder takes one packet per call, so a batch of B is B sequential decodes in one timed window
    (same emulation as Codec.edge.sqgan_profile). Payloads cycle when fewer than B are available."""
    prof, clock, sync = ctx.cfg["profile"], ctx.backends.clock, ctx.backends.sync or (lambda: None)
    try:
        if not payloads:
            return
        samples = {}
        for b in sorted(set(int(x) for x in prof["batch_sizes"])):
            batch = [payloads[i % len(payloads)] for i in range(b)]
            for _ in range(int(prof["warmup"])):
                for p in batch:
                    sq.decode(p, m)
            out = []
            for _ in range(int(prof["repeats"])):
                sync()
                t0 = clock()
                for p in batch:
                    sq.decode(p, m)
                sync()
                out.append(float(clock() - t0))
            samples[str(b)] = out
        (shard_dir / "latency_samples.json").write_text(
            json.dumps({"samples": samples, "m": m, "emulated_batching": True}), encoding="utf-8")
    except Exception as exc:
        _stage_fail(sfail, "latency", "shard", exc)


# ------------------------------------------------------------------ manifest / summary

def config_hash(*cfgs):
    return hashlib.sha256(json.dumps(cfgs, sort_keys=True, default=str).encode()).hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL, cwd=str(Path(__file__).parent)).strip()
    except Exception:
        return None


def _versions(pkgs=("torch", "numpy", "transformers", "pillow", "scikit-learn", "pyyaml")):
    out = {}
    for p in pkgs:
        try:
            out[p] = importlib.metadata.version(p)
        except Exception:
            out[p] = None
    return out


def write_manifest(ctx, args):
    man = {"step": "feature_pass", "git_sha": _git_sha(), "config_hash": config_hash(ctx.cfg, ctx.grid_cfg),
           "packages": _versions(), "gpu": ctx.backends.gpu_name() if ctx.backends.gpu_name else None, "args": args, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (ctx.out_dir / "run_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def summarize(out_dir):
    """Aggregate every finished shard: counts, failures (bad images listed), mean non-warm-up ms per stage."""
    shards = sorted(Path(out_dir, "shards").glob("shard_????"))
    tot, fails, stage_rows, stage_ms, skipped = Counter(), [], [], {}, {}
    seg_flags, seg_shapes, seg_mb = [], set(), 0.0
    enabled, skipped_stages = [], []
    for s in shards:
        if not (s / "done.json").exists():
            continue
        d = json.loads((s / "done.json").read_text(encoding="utf-8"))
        tot["shards"] += 1
        tot["requested"] += d["n_requested"]
        tot["ok"] += d["n_ok"]
        skipped.update(d.get("skipped", {}))
        enabled += [x for x in d.get("stages_enabled", []) if x not in enabled]
        skipped_stages += [x for x in d.get("stages_skipped", []) if x not in skipped_stages]
        if d.get("m_levels"):
            seg_flags.append(bool(d.get("segmap_saved")))
            seg_mb += float(d.get("segmap_mb", 0.0))
            if d.get("segmap_shape"):
                seg_shapes.add(tuple(d["segmap_shape"][1:]))
        for r in json.loads((s / "failures.json").read_text(encoding="utf-8")):
            (stage_rows if "count" in r else fails).append(r)      # stage-level rows carry a count; image rows do not
        for line in (s / "timings.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r["warmup"]:
                stage_ms.setdefault(r["stage"], []).append(r["ms"])
    stage_failures = merge_stage_failures(stage_rows)
    summary = {"counts": dict(tot), "n_failures": len(fails), "failures": fails, "skipped": skipped,
               "stages_enabled": enabled, "stages_skipped": skipped_stages,
               "n_stage_failures": sum(r["count"] for r in stage_failures), "stage_failures": stage_failures,
               "stage_mean_ms": {k: float(np.mean(v)) for k, v in sorted(stage_ms.items())}}
    if seg_flags:
        summary.update(segmap_saved=all(seg_flags), segmap_shape=["N"] + list(next(iter(seg_shapes), ())),
                       segmap_mb=round(seg_mb, 3))
    Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def cache_index(out_dir):
    """{uid: (shard_name, row)} over finished shards: the join key of an embeddings-only and an SQ-GAN-only cache."""
    idx = {}
    for s in sorted(Path(out_dir, "shards").glob("shard_????")):
        if (s / "done.json").exists():
            for i, u in enumerate(json.loads((s / "uids.json").read_text(encoding="utf-8"))):
                idx[u] = (s.name, i)
    return idx


def join_caches(dir_a, dir_b):
    """{uid: ((shard, row) in A, (shard, row) in B)} for the uids present in both caches."""
    a, b = cache_index(dir_a), cache_index(dir_b)
    return {u: (a[u], b[u]) for u in sorted(set(a) & set(b))}


def sqgan_required_failure(summary, require):
    """Message when ``require`` and SQ-GAN was skipped as a whole (it ran for no image), else None."""
    if require and "sqgan" in summary.get("stages_skipped", []):
        return ("--require-sqgan: the SQ-GAN stage was skipped as a whole and ran for no image (%s)"
                % summary.get("skipped", {}).get("sqgan", "no reason recorded"))
    return None


def run(ctx, rows, limit=None, part=(0, 1)):
    """Plan stable shards, process the ones of this ``part``, write manifest + summary."""
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_shards(rows, ctx.cfg["shard"]["size"], limit)
    write_manifest(ctx, {"limit": limit, "part": list(part), "n_rows": len(rows), "n_shards": len(plan)})
    for idx in sorted(plan):
        if idx % part[1] != part[0]:
            continue
        done = process_shard(ctx, idx, plan[idx])
        ctx.log("shard %04d: %d ok, %d failures" % (idx, done["n_ok"], done["n_failed"]))
        summarize(ctx.out_dir)
    return summarize(ctx.out_dir)


# ------------------------------------------------------------------ grid hand-off

def export_grid_inputs(out_dir, labels_rows, grid_dir, splits_path=None, log=print):
    """Concatenate shard embeddings into ``emb_<enc>.npz`` (+ .done), labels/splits keyed by uid.

    Uses ``splits_path`` when it exists (mapping uid|image_path -> split); otherwise builds PROVISIONAL event
    splits with parameters.edge.splits.assign_event_splits and says so. Returns (labels_jsonl, splits_json).
    """
    from parameters.edge.splits import assign_event_splits
    out_dir, grid_dir = Path(out_dir), Path(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    per_enc = {}
    for s in sorted((out_dir / "shards").glob("shard_????")):
        if not (s / "done.json").exists():
            continue
        uids = json.loads((s / "uids.json").read_text(encoding="utf-8"))
        for f in s.glob("emb_*.npy"):
            if f.stem.endswith("_ok"):
                continue
            ok_f = s / (f.stem + "_ok.npy")
            ok = np.load(ok_f) if ok_f.exists() else np.ones(len(uids), bool)   # drop zero-placeholder rows
            e = per_enc.setdefault(f.stem[4:], ([], []))
            e[0].extend(u for u, k in zip(uids, ok) if k)
            e[1].append(np.load(f)[ok])
    for name, (uids, arrs) in per_enc.items():
        np.savez(grid_dir / ("emb_%s.npz" % name), image_path=np.array(uids), emb=np.concatenate(arrs))
        (grid_dir / ("emb_%s.done" % name)).write_text(json.dumps({"n": len(uids)}), encoding="utf-8")
    have = {u for uids, _ in per_enc.values() for u in uids}
    rows = [{"image_path": r["uid"], "event": r.get("event"), "tier": r["tier"], "source": r.get("source")}
            for r in labels_rows if r["uid"] in have]
    labels_p, splits_p = grid_dir / "labels_by_uid.jsonl", grid_dir / "splits_by_uid.json"
    labels_p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if splits_path and Path(splits_path).exists():
        raw = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        by_row = {r["uid"]: r for r in labels_rows}
        splits = {u: raw.get(u) or raw.get(by_row[u].get("image_path") or "") or raw.get(str(by_row[u].get("image_id")))
                  for u in have}
        splits = {u: s for u, s in splits.items() if s}
        log("using splits file %s (%d rows)" % (splits_path, len(splits)))
    else:
        splits, rep = assign_event_splits(rows)
        log("splits_v1.json not present: using PROVISIONAL event splits from assign_event_splits %r" % (rep["counts"],))
    splits_p.write_text(json.dumps(splits), encoding="utf-8")
    return labels_p, splits_p


# ------------------------------------------------------------------ CLI

def _parse_part(text):
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise ValueError("--part must be i/n with 0<=i<n")
    return i, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--images-root", default=None, help="folder with index.jsonl + images/ (auto-discovered)")
    r.add_argument("--labels-root", default=None, help="folder with labels_raw.jsonl (auto-discovered)")
    r.add_argument("--out", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--smoke", action="store_true", help="20-image smoke test (limit 20)")
    r.add_argument("--part", default="0/1", help="i/n: process shards with idx %% n == i")
    r.add_argument("--device", default="cuda")
    r.add_argument("--no-sqgan", action="store_true", help="skip the SQ-GAN stages")
    r.add_argument("--only-sqgan", action="store_true",
                   help="run only the SQ-GAN stages (no embeddings/hashes); same shard layout, joinable by uid")
    r.add_argument("--require-sqgan", action=argparse.BooleanOptionalAction, default=None,
                   help="exit non-zero if the SQ-GAN stage was skipped as a whole (default: on when SQ-GAN is enabled and not --smoke)")
    r.add_argument("--no-decode", action="store_true")
    r.add_argument("--codec", choices=("v0", "v1"), default=None,
                   help="v1: fine-tuned bundle from config/edge/codec.yaml; also records LPIPS, recon mIoU, decode latency")
    r.add_argument("--encoders", nargs="*", default=None, help="subset of grid.yaml encoders")
    g = sub.add_parser("export-grid")
    g.add_argument("--labels-root", default=None)
    g.add_argument("--out", default=None)
    g.add_argument("--splits", default=None)
    pb = sub.add_parser("publish", help="publish the cache folder as a Kaggle dataset (skips without credentials)")
    pb.add_argument("--out", default=None)
    pb.add_argument("--slug", default=None)
    a = ap.parse_args(argv)
    cfg, grid_cfg = load_config("feature_pass"), load_config("grid")
    pth = cfg["paths"]
    out = Path(a.out or pth["out"])
    if a.cmd == "publish":
        try:
            from parameters.edge.kaggle_publish import publish_dataset
            slug = a.slug or pth["cache_slug"]
            print(publish_dataset(str(out), slug, "Edge S2 feature cache (" + slug + ")"))
        except Exception as exc:                       # publishing must never fail the run
            print({"status": "skipped", "reason": repr(exc)})
        return
    labels_root = Path(a.labels_root or find_data_root(pth["labels_file"], pth["input_base"], pth["discover_depth"]))
    labels = read_labels_raw(labels_root / pth["labels_file"])
    if a.cmd == "export-grid":
        splits = a.splits or str(labels_root / "splits_v1.json")
        lp, sp = export_grid_inputs(out, labels, out / "grid", splits)
        print(lp, sp)
        return
    cfg = json.loads(json.dumps(cfg))
    if a.no_sqgan and a.only_sqgan:
        raise SystemExit("--no-sqgan and --only-sqgan are mutually exclusive")
    if a.only_sqgan:
        cfg["embeddings"]["enabled"] = False
        cfg["hashes"]["enabled"] = False
    if a.no_sqgan:
        cfg["sqgan"]["enabled"] = False
    if a.no_decode:
        cfg["decode"]["enabled"] = False
    if a.codec:
        cfg["sqgan"]["codec"] = a.codec
    if cfg["sqgan"].get("codec") == "v1":
        cfg["decode"].update(quality=True, latency=True)
    if a.encoders is not None:
        cfg["embeddings"]["encoders"] = a.encoders
    from Codec.edge import feature_pass_backends as real
    backends = Backends(real.embedder_factory(a.device, grid_cfg),
                        lambda: real.RealSQGAN(cfg, a.device), sync=real.cuda_sync,
                        gpu_name=real.gpu_name)
    root = Path(a.images_root or find_data_root(pth["images_marker"], pth["input_base"], pth["discover_depth"]))
    ctx = Context(cfg, grid_cfg, root, out, backends)
    limit = 20 if a.smoke else a.limit
    summ = run(ctx, labels, limit, _parse_part(a.part))
    print(json.dumps({k: summ[k] for k in ("counts", "n_failures", "skipped", "stages_enabled", "stages_skipped")}, indent=2))
    require = (cfg["sqgan"]["enabled"] and not a.smoke) if a.require_sqgan is None else (a.require_sqgan and cfg["sqgan"]["enabled"])
    msg = sqgan_required_failure(summ, require)
    if msg:
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
