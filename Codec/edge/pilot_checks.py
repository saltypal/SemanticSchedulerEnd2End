"""Pilot checks: every item earlier workers reported as UNVERIFIED (needs GPU / network / real models).

    python -m Codec.edge.pilot_checks <check_name|all|list> --out <dir>

Runs ONE check inside try/except and writes ``<dir>/checks/<check_name>.json``:
``{name, status: pass|fail|skip, seconds, details, error, covers, items}``. ``items`` optionally holds a
per-item ``{item_id: {status, detail}}`` (an item without an entry inherits the check status), so one check that
exercises several things (e.g. five encoders) reports each separately. ``UNVERIFIED_ITEMS`` maps item id ->
(description, source report, check name); ``parameters.edge.pilot`` renders it as the final table.

Nothing here re-implements repo logic: each check calls the public API under test and records ``fail`` if it
breaks (that is the point of a pilot). Heavy libraries are imported lazily. Tests inject fakes through
``Ctx.inject`` (keys: make_codec, load_embedder, ue_probe, publish, runner, images).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

INPUT_BASE = "/kaggle/input"
BUNDLE_ROOT = "/kaggle/working/bundle_v0"
FULL_DISK_GB = 20.0
PUBLISH_SLUG = "edge-pilot-publish-test"
SKIP_PUBLISH_HOWTO = ("Add Kaggle secrets KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) under Add-ons > Secrets "
                      "and attach them to this notebook, then re-run.")


class Skip(Exception):
    """Raised inside a check to record status skip with the message as reason."""


@dataclass
class Ctx:
    out: Path
    input_base: str = INPUT_BASE
    bundle_root: str = BUNDLE_ROOT
    inject: dict = field(default_factory=dict)

    @property
    def work(self):
        return Path(self.out)

    def hook(self, name, default):
        return self.inject.get(name, default)


# ------------------------------------------------------------------ shared helpers

def _tail(text, n=1500):
    return (text or "")[-n:]


def item(status, detail):
    return {"status": status, "detail": str(detail)[:300]}


DISCOVER_DEPTH = 5
DATASET_KEYWORDS = ("sqgan-bundle", "sqgan-source", "sqgan-assets")
DATASET_FILES = ("bundle_manifest.json", "*flat_manifest.json", "sqgan-pinned-source.zip")


def _walk(base, depth):
    """Yield (path, depth) for every entry under ``base`` down to ``depth`` levels (1 = direct children)."""
    base = Path(base)
    if depth < 1 or not base.is_dir():
        return
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return
    for p in entries:
        yield p, 1
        if p.is_dir():
            for q, d in _walk(p, depth - 1):
                yield q, d + 1


def input_tree(base, depth=3, limit=200):
    """Names of /kaggle/input down to ``depth`` levels (relative paths, at most ``limit``) so the mount layout is visible."""
    base = Path(base)
    out = []
    for p, _ in _walk(base, depth):
        out.append(p.relative_to(base).as_posix() + ("/" if p.is_dir() else ""))
        if len(out) >= limit:
            break
    return out


def discover_datasets(input_base, depth=DISCOVER_DEPTH):
    """{'dirs': {keyword: [dir, ...]}, 'files': {pattern: [file, ...]}} found under ``input_base`` (recursive, shallowest first)."""
    import fnmatch
    dirs = {k: [] for k in DATASET_KEYWORDS}
    files = {k: [] for k in DATASET_FILES}
    for p, _ in _walk(input_base, depth):
        low = p.name.lower()
        if p.is_dir():
            for k in DATASET_KEYWORDS:
                if k in low:
                    dirs[k].append(str(p))
        else:
            for k in DATASET_FILES:
                if fnmatch.fnmatch(low, k.lower()):
                    files[k].append(str(p))
    key = lambda x: (len(Path(x).parts), x)
    return {"dirs": {k: sorted(v, key=key) for k, v in dirs.items()},
            "files": {k: sorted(v, key=key) for k, v in files.items()}}


def find_dataset(ctx, pattern):
    """Attached dataset directory whose lowercase name contains the keyword in ``pattern`` (``*sqgan-bundle*``).

    Searches recursively to depth 5 (layouts /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>). Falls
    back to the folder holding a flat/bundle manifest (bundle) or the pinned source zip (source/assets).
    """
    from Codec.edge.feature_pass_backends import find_datasets, find_upstream_sqgan
    keyword = pattern.strip("*").lower()
    found = discover_datasets(ctx.input_base)
    hits = [str(p) for p in find_datasets(keyword, ctx.input_base, DISCOVER_DEPTH)] or found["dirs"].get(keyword, [])
    if not hits and keyword == "sqgan-bundle":
        hits = [str(Path(f).parent) for k in ("*flat_manifest.json", "bundle_manifest.json") for f in found["files"][k]]
    if not hits and keyword == "sqgan-source":
        kind, up = find_upstream_sqgan(ctx.input_base, DISCOVER_DEPTH)
        hits = [str(up.parent.parent)] if kind == "dir" else []
        hits = hits or [str(Path(f).parent) for f in found["files"]["sqgan-pinned-source.zip"]]
    if not hits:
        raise Skip("no dataset matching %s mounted under %s (searched depth %d; tree: %s)"
                   % (pattern, ctx.input_base, DISCOVER_DEPTH, input_tree(ctx.input_base, 3, 20)))
    return Path(hits[0])


def ensure_bundle(ctx):
    """Restore the flat v0 bundle (reuses feature_pass_backends.restore_flat_bundle). Returns bundle dir."""
    from Codec.edge.feature_pass_backends import restore_flat_bundle
    flat = find_dataset(ctx, "*sqgan-bundle*")
    return restore_flat_bundle(flat, ctx.bundle_root), flat


def resolve_source(ctx):
    """(root, mode): mounted dataset with src/, else a clone of the pinned commit (``clone`` hook for tests)."""
    from Codec.edge.feature_pass_backends import resolve_source_root
    from Evaluation.edge.edge_config import load_config
    cfg = load_config("feature_pass")
    cfg["paths"].update(input_base=ctx.input_base, restore_dir=str(ctx.work / "restore" / "bundle_dir"))
    return resolve_source_root(cfg, log=print, clone=ctx.inject.get("clone"), base=ctx.input_base)


def source_root(ctx):
    return resolve_source(ctx)[0]


def make_codec(ctx, m=0.25):
    """Real SQGANDigitalCodec on cuda for the restored bundle (injectable)."""
    fake = ctx.inject.get("make_codec")
    if fake:
        return fake(m)
    from Codec.integration.sqgan_payload import SQGANDigitalCodec
    bundle, _ = ensure_bundle(ctx)
    return SQGANDigitalCodec(bundle=bundle, source_root=source_root(ctx), device="cuda",
                             image_fraction=float(m), semantic_fraction=float(m))


def pil_images(ctx, n):
    """n PIL RGB images: pilot dataset A images when present, else seeded random noise images."""
    if "images" in ctx.inject:
        return list(ctx.inject["images"])[:n]
    from PIL import Image
    import numpy as np
    paths = []
    root = ctx.work / "dataset-a" / "images"
    if root.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += sorted(root.rglob(ext))
    out = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        if len(out) == n:
            return out
    rng = np.random.default_rng(0)
    while len(out) < n:
        out.append(Image.fromarray(rng.integers(0, 255, (300, 520, 3), dtype=np.uint8)))
    return out


def image_tensor(img, device="cuda"):
    """PIL -> (1,3,256,512) float tensor in 0..1 via feature_pass.letterbox."""
    import numpy as np
    import torch
    from Codec.edge.feature_pass import letterbox
    arr, _ = letterbox(img)
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float().div(255.0).to(device)


def describe(obj, depth=0):
    """Type/keys/shape/dtype summary of a (possibly nested) masker output."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {"type": type(obj).__name__, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if depth > 3:
        return {"type": type(obj).__name__}
    if isinstance(obj, dict):
        return {"type": "dict", "keys": sorted(map(str, obj)), "fields": {str(k): describe(v, depth + 1) for k, v in obj.items()}}
    if isinstance(obj, (tuple, list)):
        return {"type": type(obj).__name__, "len": len(obj), "items": [describe(v, depth + 1) for v in obj[:6]]}
    attrs = getattr(obj, "__dict__", None)
    if attrs:
        return {"type": type(obj).__name__, "attrs": sorted(attrs), "fields": {k: describe(v, depth + 1) for k, v in list(attrs.items())[:12]}}
    return {"type": type(obj).__name__, "repr": repr(obj)[:80]}


def _peak_vram():
    try:
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except Exception:
        return None


def _reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cfg(name):
    from Evaluation.edge.edge_config import load_config
    return load_config(name)


# ------------------------------------------------------------------ 1 env_gpu

def check_env_gpu(ctx):
    import platform
    import torch
    d = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda_version": torch.version.cuda}
    count = torch.cuda.device_count()
    d["gpu_count"], d["cuda_available"] = count, bool(torch.cuda.is_available())
    d["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
    d["free_disk_gb"] = round(shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else ".").free / 1e9, 1)
    libs = {}
    for lib in ("transformers", "timm", "lpips", "sklearn", "PIL", "numpy", "yaml", "pytorch_lightning", "torchmetrics"):
        try:
            libs[lib] = getattr(__import__(lib), "__version__", "?")
        except Exception as exc:
            libs[lib] = "missing: %r" % exc
    d["libraries"] = libs
    d["input_tree"] = input_tree(ctx.input_base, 3, 200)
    d["discovered_datasets"] = discover_datasets(ctx.input_base)
    items = {"gpu_count": item("pass" if count else "fail", "%d GPU(s)" % count),
             "gpu_names": item("pass" if d["gpu_names"] else "fail", ", ".join(d["gpu_names"]) or "none"),
             "torch_cuda": item("pass" if d["cuda_available"] else "fail", "cuda=%s torch %s" % (d["cuda_available"], torch.__version__)),
             "free_disk": item("pass" if d["free_disk_gb"] >= 10 else "fail", "%.1f GB free" % d["free_disk_gb"]),
             "library_versions": item("pass" if not any(str(v).startswith("missing") for v in libs.values()) else "fail",
                                      "missing: " + ", ".join(k for k, v in libs.items() if str(v).startswith("missing")))}
    if count >= 2:
        try:
            net = torch.nn.DataParallel(torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU()).cuda())
            net(torch.randn(8, 16, device="cuda")).sum().backward()
            torch.cuda.synchronize()
            items["dataparallel_t4x2"] = item("pass", "DataParallel fwd/bwd on %d GPUs" % count)
        except Exception as exc:
            items["dataparallel_t4x2"] = item("fail", repr(exc))
    else:
        items["dataparallel_t4x2"] = item("skip", "only %d GPU(s); need 2" % count)
    return {"details": d, "items": items}


# ------------------------------------------------------------------ 2, 11 bundle

def check_bundle_restore(ctx):
    from Codec.edge import bundle_check
    d, items = {}, {}
    flat = find_dataset(ctx, "*sqgan-bundle*")
    d["flat_dir"], d["flat_files"] = str(flat), sorted(p.name for p in flat.iterdir())[:40]
    try:
        bundle, _ = ensure_bundle(ctx)
        res = bundle_check.verify_bundle(bundle)
        d["bundle_check"] = {"ok": res["ok"], "errors": res["errors"]}
        items["bundle_flat_restore"] = item("pass" if res["ok"] else "fail", "restored to %s; bundle_check ok=%s %s" % (bundle, res["ok"], res["errors"][:2]))
        items["feature_pass_restore_flat_bundle_pattern"] = item("pass" if res["ok"] else "fail", "flat names %s restored via restore_flat_bundle" % d["flat_files"][:3])
    except Exception as exc:
        d["restore_error"] = _tail(traceback.format_exc())
        items["bundle_flat_restore"] = item("fail", "%s: %s (flat files: %s)" % (type(exc).__name__, exc, d["flat_files"][:4]))
        items["feature_pass_restore_flat_bundle_pattern"] = item("fail", "restore_flat_bundle raised %s (flat files %s)" % (type(exc).__name__, d["flat_files"][:4]))
        items["sqgan_loads_strict"] = item("fail", "bundle not restored")
        return {"details": d, "items": items}
    try:
        _reset_vram()
        t0 = time.time()
        codec = make_codec(ctx)
        d["load_seconds"], d["peak_vram_bytes"] = round(time.time() - t0, 2), _peak_vram()
        d["validation"] = getattr(codec, "validation", None)
        items["sqgan_loads_strict"] = item("pass", "loaded in %.1fs, peak VRAM %s B" % (d["load_seconds"], d["peak_vram_bytes"]))
    except Exception as exc:
        d["load_error"] = _tail(traceback.format_exc())
        items["sqgan_loads_strict"] = item("fail", "%s: %s" % (type(exc).__name__, exc))
    return {"details": d, "items": items}


def check_bundle_deep_scan(ctx):
    from Codec.edge import bundle_check
    bundle, _ = ensure_bundle(ctx)
    res = bundle_check.verify_bundle(bundle, deep=True)
    return {"details": {"ok": res["ok"], "errors": res["errors"], "bundle": str(bundle)},
            "items": {"bundle_deep_key_scan": item("pass" if res["ok"] else "fail", "ok=%s %s" % (res["ok"], res["errors"][:2]))}}


def check_source_root(ctx):
    """feature_pass source-root resolution: mounted dataset with src/, else the pinned git clone (plus .upstream/sqgan)."""
    root, mode = resolve_source(ctx)
    return {"details": {"mode": mode, "source_root": str(root), "has_sqgan_py": (root / "src/sqgan.py").exists(),
                        "has_bitstream_py": (root / "src/bitstream.py").exists(),
                        "has_upstream_sqgan": (root / ".upstream" / "sqgan").is_dir()},
            "items": {"feature_pass_find_source_root": item("pass", "source root %s (%s)" % (root, mode))}}


# ------------------------------------------------------------------ 3 masker_format

def check_masker_format(ctx):
    from Codec.edge.codec_table import encode_with_kept_tokens
    from parameters.edge import sts as sts_mod
    m = 0.25
    codec = make_codec(ctx, m)
    img = pil_images(ctx, 2)
    seen = {}

    def record(module, inputs, output):
        seen.setdefault("v", describe(output))
    handle = codec.model.masker.register_forward_hook(record)
    rows, sts_ok = [], True
    try:
        for im in img:
            payload, kept, seg = encode_with_kept_tokens(codec, image_tensor(im, ctx.hook("device", "cuda")))
            row = {"payload_bytes": len(payload), "kept_count": len(kept), "expected_m_x_512": round(m * 512)}
            try:
                row["sts_len"] = len(sts_mod.compute_sts(seg, kept))
            except Exception as exc:
                sts_ok, row["sts_error"] = False, repr(exc)
            rows.append(row)
    finally:
        handle.remove()
    out = seen.get("v", {})
    text = json.dumps(out)
    d = {"masker_output": out, "sample_index_present": "sample_index" in text, "images": rows,
         "alpha_exposed": any(k in text.lower() for k in ("alpha", "relevance", "score"))}
    kept_ok = all(abs(r["kept_count"] - r["expected_m_x_512"]) <= 0.1 * 512 for r in rows)
    status = "pass" if d["sample_index_present"] and sts_ok and kept_ok else "fail"
    d["kept_count_matches"] = kept_ok
    d["sts_ok"] = sts_ok
    return {"details": d, "items": {"real_masker_output": item(status, "sample_index=%s kept=%s sts_ok=%s alpha=%s" % (
        d["sample_index_present"], [r["kept_count"] for r in rows], sts_ok, d["alpha_exposed"]))}}


# ------------------------------------------------------------------ feature_pass semantics (m mapping, ranges, grid)

def check_feature_pass_codec_semantics(ctx):
    from Codec.edge.codec_table import masker_capture
    from Codec.integration.sqgan_prefeatures import prefeature_capture
    from Codec.edge.feature_pass import pool_prefeatures
    import numpy as np
    m_levels = list(_cfg("feature_pass")["sqgan"]["m_levels"])
    im = image_tensor(pil_images(ctx, 1)[0], ctx.hook("device", "cuda"))
    per, grids = {}, {}
    for m in m_levels:
        codec = make_codec(ctx, m)
        sem = codec.semantic_map(im)
        with prefeature_capture(codec) as (feat, calls), masker_capture(codec) as mk:
            payload = codec.encode(im, sem)
        kept = mk.get("kept", [])
        grid = feat["features"]
        grids[m] = grid.float().cpu().numpy()
        per[m] = {"payload_bytes": len(payload), "kept": len(kept), "expected": m * 512, "index_min": min(kept) if kept else None,
                  "index_max": max(kept) if kept else None, "encoder_calls": len(calls), "grid_shape": list(grid.shape),
                  "pooled_len": int(np.asarray(pool_prefeatures(grid[0])).size)}
    ms = sorted(per)
    growing = all(per[a]["payload_bytes"] < per[b]["payload_bytes"] for a, b in zip(ms, ms[1:]))
    ratio = all(abs(per[m]["kept"] - m * 512) <= 0.1 * 512 for m in ms)
    rng = all(per[m]["index_min"] is not None and 0 <= per[m]["index_min"] and per[m]["index_max"] <= 511 for m in ms)
    shape = all(per[m]["grid_shape"][-3:] == [256, 16, 32] and per[m]["pooled_len"] == 512 for m in ms)
    once = all(per[m]["encoder_calls"] == 1 for m in ms)
    diff = float(np.abs(grids[ms[0]] - grids[ms[-1]]).max()) if len(ms) > 1 else 0.0
    st = lambda ok: "pass" if ok else "fail"
    return {"details": {"per_m": {str(k): v for k, v in per.items()}, "max_abs_grid_diff_between_m": diff},
            "items": {"feature_pass_m_mapping": item(st(growing and ratio), "payload grows with m=%s, kept~m*512=%s" % (growing, ratio)),
                      "feature_pass_sample_index_range": item(st(rng), "indices within 0..511 = %s" % rng),
                      "feature_pass_prefeat_grid_shape": item(st(shape), "grid shapes %s, pooled len 512 = %s" % ([per[m]["grid_shape"] for m in ms][:1], shape)),
                      "feature_pass_encoder_once_per_m": item(st(once), "one encoder call per encode = %s" % once),
                      "feature_pass_grid_m_independence": item(
                          "pass" if diff < 1e-4 else "fail",
                          "max |grid(m0)-grid(m1)| = %.3g (%s)" % (diff, "identical: first-m cache is safe" if diff < 1e-4
                                                                  else "DIFFERS: caching only the first m is wrong"))}}


# ------------------------------------------------------------------ 4 sqgan_latency

def codec_flops(codec, image):
    """GFLOPs of encoder + decoder for one image via compute_profile forward-pre-hook capture."""
    from Codec.edge.compute_profile import _capture_inputs, count_flops
    boxes = {n: _capture_inputs(getattr(codec.model, n)) for n in ("encoder", "decoder")}
    codec.decode(codec.encode(image))
    total = 0
    for n, (box, h) in boxes.items():
        h.remove()
        if "v" in box:
            total += count_flops(getattr(codec.model, n), box["v"][0], box["v"][1])[0]
    return total / 1e9


def check_sqgan_latency(ctx):
    from Codec.edge.sqgan_profile import profile_codec
    import numpy as np
    codec = make_codec(ctx)
    imgs = [image_tensor(i, ctx.hook("device", "cuda")) for i in pil_images(ctx, 3)]
    _reset_vram()
    prof = profile_codec(codec, imgs, [1, 2, 4, 8], repeats=3, device="cuda",
                         flops_fn=lambda c: codec_flops(c, imgs[0]))
    ms = {str(b): {"mean": float(np.mean(v) * 1e3), "min": float(np.min(v) * 1e3), "max": float(np.max(v) * 1e3)}
          for b, v in prof["latency_samples"].items()}
    d = {"decode_ms": ms, "encoder_ms": prof["encoder_ms"], "peak_vram_bytes": prof["peak_vram_bytes"], "gflops": prof["gflops"],
         "emulated_batching": prof["emulated_batching"], "payload_bytes_mean": prof["mean_payload_bytes"],
         "payload_sizes": None}
    finite = all(np.isfinite(v["mean"]) and v["mean"] > 0 for v in ms.values())
    return {"details": d, "items": {
        "decode_latency_gpu": item("pass" if finite else "fail", "decode ms mean by B: " + ", ".join("%s:%.0f" % (b, v["mean"]) for b, v in ms.items())),
        "vram_peak": item("pass" if prof["peak_vram_bytes"] else "fail", "peak %s B" % prof["peak_vram_bytes"]),
        "gflops_sqgan": item("pass" if prof["gflops"] and prof["gflops"] > 0 else "fail", "%s GFLOPs (encoder+decoder)" % prof["gflops"]),
        "feature_pass_batch_latency_irregular": item("pass" if finite else "fail", "%d images with different payload sizes; emulated=%s" % (len(imgs), prof["emulated_batching"]))}}


# ------------------------------------------------------------------ 5 embedders

ENCODER_ITEM = {"clip_b32": "load_embedder_clip", "siglip2_base": "load_embedder_siglip2", "dinov2_small": "load_embedder_dinov2_small",
                "dinov2_base": "load_embedder_dinov2_base", "convnextv2_tiny": "load_embedder_convnextv2"}


def vision_wrapper(model, family):
    import torch.nn as nn

    class Vision(nn.Module):
        def forward(self, pixel_values):
            if family in ("clip", "siglip"):
                return model.get_image_features(pixel_values=pixel_values)
            return model(pixel_values=pixel_values).last_hidden_state
    w = Vision()
    w.model = model
    return w


def _probe_encoder(ctx, enc, images, grid_cfg):
    import numpy as np
    name = enc["name"]
    loader = ctx.hook("load_embedder", None)
    if loader is None:
        from parameters.edge.embedding_grid import load_embedder
        loader = lambda n: load_embedder(n, "cuda", grid_cfg)
    probe = ctx.hook("ue_probe", None)
    if probe is None:
        from parameters.edge.embedding_grid import ue_cost_probe
        probe = lambda n, imgs: ue_cost_probe(n, lambda x: x, n=len(imgs), device="cuda", items=imgs, grid_cfg=grid_cfg)
    _reset_vram()
    embed = loader(name)
    embed(images[:1])
    t0 = time.perf_counter()
    emb = np.asarray(embed(images))
    ms = (time.perf_counter() - t0) * 1000.0 / len(images)
    rec = {"encoder": name, "dim": int(emb.shape[-1]), "yaml_dim": enc["dim"], "ms_per_image": round(ms, 2),
           "peak_vram_bytes": _peak_vram(), "shape": list(emb.shape), "finite": bool(np.isfinite(emb).all())}
    model = getattr(embed, "model", None)
    if model is not None:
        rec["params"] = int(sum(p.numel() for p in model.parameters()))
        try:
            import torch
            from Codec.edge.compute_profile import count_flops
            wrapper = vision_wrapper(model, enc["family"]).eval()
            rec["flops"] = count_flops(wrapper, (torch.rand(1, 3, 224, 224, device=next(model.parameters()).device),))[0]
        except Exception as exc:
            rec["flops_error"] = repr(exc)
    rec["ue_cost_probe"] = probe(name, images)
    rec["dim_matches_yaml"] = rec["dim"] == enc["dim"]
    return rec


def check_embedders(ctx):
    grid = ctx.hook("grid_cfg", None) or _cfg("grid")
    images = pil_images(ctx, 8)
    d, items, failed = {}, {}, []
    for enc in grid["encoders"]:
        iid = ENCODER_ITEM.get(enc["name"], "load_embedder_" + enc["name"])
        try:
            rec = _probe_encoder(ctx, enc, images, grid)
            d[enc["name"]] = rec
            ok = rec["dim_matches_yaml"] and rec["finite"]
            items[iid] = item("pass" if ok else "fail", "dim %s (yaml %s), %.1f ms/img, params %s" % (
                rec["dim"], rec["yaml_dim"], rec["ms_per_image"], rec.get("params")))
            if not ok:
                failed.append(enc["name"])
        except Exception as exc:
            d[enc["name"]] = {"error": _tail(traceback.format_exc())}
            items[iid] = item("fail", "%s: %s" % (type(exc).__name__, exc))
            failed.append(enc["name"])
        _free()
    return {"status": "fail" if failed else "pass", "details": dict(d, failed=failed), "items": items}


def _free():
    try:
        from parameters.edge.embedding_grid import _free_gpu
        _free_gpu()
    except Exception:
        pass


# ------------------------------------------------------------------ 6 judge_dino

def check_judge_dino(ctx):
    import numpy as np
    from Evaluation.edge.judge import LinearProbeJudge, extract_dino_embeddings, fidelity
    images = pil_images(ctx, 8)
    extract = ctx.hook("extract_dino", extract_dino_embeddings)
    emb = np.asarray(extract(images, device=ctx.hook("device", "cuda")) if "extract_dino" not in ctx.inject else extract(images))
    labels = np.array([i % 2 for i in range(len(images))])
    judge = LinearProbeJudge().fit(emb, labels)
    fid = fidelity(judge, emb, emb)
    ok = emb.shape[0] == 8 and bool(np.isfinite(emb).all()) and int(np.sum(fid)) == len(fid)
    return {"details": {"shape": list(emb.shape), "fidelity_self": fid.tolist()},
            "items": {"judge_dino_embeddings": item("pass" if ok else "fail", "embeddings %s, self-fidelity %s" % (list(emb.shape), fid.tolist()))}}


# ------------------------------------------------------------------ 7 ade_teacher

def check_ade_teacher(ctx):
    import yaml
    from Codec.edge import seg_map
    cfg_ft = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/edge/sqgan_finetune.yaml").read_text())
    tc = cfg_ft["teacher"]
    seg_cfg = seg_map.load_class_map()
    loader = ctx.hook("teacher_loader", None)
    if loader is None:
        def loader(model, revision):
            from transformers import SegformerForSemanticSegmentation
            m = SegformerForSemanticSegmentation.from_pretrained(model, revision=revision).eval()
            return m.config.id2label, m
    id2label, model = loader(tc["model"], tc["revision"])
    want = {r["id"]: r["ade"] for r in seg_cfg["ade_to_project"]}
    got = {int(k): str(v) for k, v in id2label.items()}
    mism = [(i, want.get(i), got.get(i)) for i in sorted(set(want) | set(got))
            if (got.get(i) or "").split(",")[0].strip().lower() != want.get(i)]
    d = {"teacher": tc["model"], "n_labels_teacher": len(got), "n_labels_yaml": len(want), "mismatches": mism}
    items = {"ade_id2label_match": item("pass" if not mism else "fail", "%d mismatches; first %s" % (len(mism), mism[:3]))}
    try:
        import torch
        index = seg_map.build_index(seg_cfg)
        x = torch.rand(2, 3, 256, 512)
        with torch.no_grad():
            logits = model(pixel_values=x).logits
        out = seg_map.project_logits(logits, index)
        d["logits_shape"], d["projected_shape"] = list(logits.shape), list(out.shape)
        ok = out.shape[1] == 20 and bool(torch.isfinite(out).all())
        items["seg_class_map_real"] = item("pass" if ok else "fail", "logits %s -> %s" % (list(logits.shape), list(out.shape)))
    except Exception as exc:
        d["project_error"] = _tail(traceback.format_exc())
        items["seg_class_map_real"] = item("fail", "%s: %s" % (type(exc).__name__, exc))
    return {"details": d, "items": items}


# ------------------------------------------------------------------ 8 finetune_source, 9 finetune_pilot

def _ft_cfg(ctx):
    from Codec.edge import sqgan_finetune
    cfg = sqgan_finetune.load_config(Path(__file__).resolve().parents[2] / "config/edge/sqgan_finetune.yaml")
    cfg["source"]["local_dir"] = str(ctx.work / "ft_source" / "Semantic_Lossless")
    return cfg


def check_finetune_source(ctx):
    from Codec.edge import sqgan_finetune as ft
    cfg = _ft_cfg(ctx)
    d, items = {}, {}
    try:                                           # path 1: pinned commit only (fallback disabled so it cannot hide a failure)
        strict = json.loads(json.dumps(cfg))
        strict["source"]["fallback_search_root"] = str(ctx.work / "no_such_fallback_dir")
        root, banner = ft.obtain_source(strict)
        d["git_root"], d["git_src_present"] = str(root), (Path(root) / "src" / "sqgan_train.py").exists()
        items["pinned_sha_fetch"] = item("pass" if d["git_src_present"] else "fail", "fetched pin; src/ present=%s" % d["git_src_present"])
    except Exception as exc:
        d["git_error"] = _tail(str(exc))
        items["pinned_sha_fetch"] = item("fail", "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    target_root = ctx.work / "ft_source" / "zip_only"
    cfg["source"]["fallback_search_root"] = ctx.input_base
    try:                                           # path 2: provide .upstream/sqgan (extracted folder or zip) into an empty dir
        target_root.mkdir(parents=True, exist_ok=True)
        target = ft.ensure_upstream_sqgan(target_root, cfg)
        d["zip_target"], d["zip_src_present"] = str(target), (target_root / "src").is_dir()
        items["upstream_zip_extract"] = item("pass" if (target / "config" / "sqgan_cityscapes.yml").is_file() else "fail",
                                             "provided %s; SemanticLossless src/ present=%s" % (target, d["zip_src_present"]))
    except Exception as exc:
        d["zip_error"] = _tail(str(exc))
        items["upstream_zip_extract"] = item("fail", "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    d["working_path"] = [k for k, v in (("pinned_sha_fetch", "git_src_present"), ("upstream_zip_extract", "zip_src_present")) if d.get(v)]
    return {"details": d, "items": items}


def _run_driver(ctx, sl_root, options, run, seg, timeout, runner):
    run.mkdir(parents=True, exist_ok=True)
    opt = run / "trainer_options.json"
    opt.write_text(json.dumps(options), encoding="utf-8")
    driver = Path(__file__).with_name("sqgan_driver.py")
    return runner([sys.executable, "-u", str(driver), "pilot", "--sl-root", str(sl_root), "--options", str(opt),
                   "--run", str(run), "--seg-classes", str(seg)], capture_output=True, text=True, cwd=str(sl_root), timeout=timeout)


def check_finetune_pilot(ctx):
    from PIL import Image
    from Codec.edge import sqgan_finetune as ft
    import numpy as np
    cfg = _ft_cfg(ctx)
    cfg["source"]["fallback_search_root"] = ctx.input_base
    cfg["inputs"]["checkpoint_search_root"] = ctx.input_base
    runner = ctx.hook("runner", subprocess.run)
    index = ctx.work / "dataset-a" / "index.jsonl"
    rows = [json.loads(x) for x in index.read_text().splitlines() if x] if index.exists() else []
    big = [r for r in rows if r.get("width", 0) >= 512 and r.get("height", 0) >= 256][:4]
    if len(big) < 4:
        raise Skip("need 4 pilot images >= 256x512 in dataset-a/index.jsonl, found %d" % len(big))
    sl_root, _ = ft.obtain_source(cfg)
    ft.ensure_upstream_sqgan(sl_root, cfg)
    ckpt = ft.find_checkpoint_dir(cfg["inputs"]["checkpoint_search_root"])
    seg = Path(__file__).resolve().parents[2] / cfg["teacher"]["seg_classes"]
    train = [dict(r, uid=r["uid"], source=r["source"], file=r["file"]) for r in big]
    work = ctx.work / "ft_pilot"
    manifest = ft.build_manifest(train, ctx.work / "dataset-a", cfg["data"], work)
    mpath = work / "train_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    options = ft.trainer_options(cfg, mpath, ckpt, int(ctx.hook("n_gpus", 0) or _gpu_count()))
    options.update(max_seconds=120.0, epochs=1)
    d, items = {"manifest_counts": manifest["counts"], "options": options}, {}
    t0 = time.time()
    done = _run_driver(ctx, sl_root, options, work / "run", seg, 900, runner)
    d["pilot_returncode"], d["pilot_seconds"] = done.returncode, round(time.time() - t0, 1)
    d["pilot_stderr_tail"] = _tail(done.stderr)
    items["trainer_launch"] = item("pass" if done.returncode == 0 else "fail", "pilot rc=%s in %.0fs %s" % (
        done.returncode, d["pilot_seconds"], "" if done.returncode == 0 else _tail(done.stderr, 200)))
    # small image: 256x256 must be rejected by the trainer and handled by small_image_policy in the entry point
    small_dir = work / "small"
    small_dir.mkdir(parents=True, exist_ok=True)
    small = small_dir / "small.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(small)
    rec = dict(manifest["images"][0], source="smallroot", relative_path="small.png", sha256=ft.sha256(small), width=256, height=256, uid="pilot:small")
    bad = dict(manifest, images=[dict(rec, split="train"), dict(rec, split="val")], roots={"smallroot": str(small_dir)},
               counts={"train": 1, "val": 1, "test": 0})
    bpath = work / "small_manifest.json"
    bpath.write_text(json.dumps(bad), encoding="utf-8")
    done2 = _run_driver(ctx, sl_root, dict(options, manifest=str(bpath)), work / "run_small", seg, 600, runner)
    policy = cfg["data"]["small_image_policy"]
    try:
        small_rows = [{"uid": "pilot:small", "source": "smallroot", "file": "images/smallroot/small.png"}]
        (ctx.work / "ft_pilot" / "small_root" / "images" / "smallroot").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(small, ctx.work / "ft_pilot" / "small_root" / "images" / "smallroot" / "small.png")
        ft.build_manifest(small_rows, ctx.work / "ft_pilot" / "small_root", cfg["data"], work / "policy")
        policy_msg = "build_manifest accepted it (policy %s)" % policy
        policy_ok = policy == "upscale"
    except ValueError as exc:
        policy_msg, policy_ok = "build_manifest raised: %s" % str(exc)[:120], policy == "skip"
    d.update(small_returncode=done2.returncode, small_stderr_tail=_tail(done2.stderr), small_policy=policy, small_policy_result=policy_msg)
    rejected = done2.returncode != 0
    items["trainer_small_image_rejection"] = item("pass" if rejected and policy_ok else "fail",
                                                  "trainer rejected 256x256 = %s; %s" % (rejected, policy_msg))
    return {"details": d, "items": items}


def _gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


# ------------------------------------------------------------------ 10 payload_semantic_map

def check_payload_semantic_map(ctx):
    import inspect
    codec = make_codec(ctx)
    sem = codec.semantic_map(image_tensor(pil_images(ctx, 1)[0], ctx.hook("device", "cuda")))
    d = {"shape": list(sem.shape), "dtype": str(sem.dtype), "min": float(sem.min()), "max": float(sem.max())}
    items = {}
    shape_ok = len(sem.shape) == 4 and sem.shape[1] == 20 and tuple(sem.shape[-2:]) == (256, 512)
    bundle = Path(getattr(codec, "bundle", "."))
    has_seg = any(bundle.rglob("seg_classes.yaml")) if bundle.exists() else False
    d["bundle_has_seg_classes"] = has_seg
    try:
        src = inspect.getsource(codec.semantic_map)
    except (OSError, TypeError):
        src = ""
    d["reader_calls_project_logits"] = "project_logits" in src
    if not has_seg:
        items["payload_reader_gap"] = item("skip", "bundle has no seg_classes.yaml (v0); sub-item not applicable")
    elif not d["reader_calls_project_logits"]:
        items["payload_reader_gap"] = item("fail", "KNOWN GAP: bundle has seg_classes.yaml but SQGANDigitalCodec.semantic_map argmaxes raw logits instead "
                                                   "of seg_map.project_logits (docs/SQGAN_FINETUNE_HANDOFF.md)")
    else:
        items["payload_reader_gap"] = item("pass", "reader applies project_logits")
    bad = not shape_ok or any(v["status"] == "fail" for v in items.values())
    return {"status": "fail" if bad else "pass", "details": d, "items": items}


# ------------------------------------------------------------------ feature_pass compute profile on real models

def check_compute_profile_real(ctx):
    from Codec.edge.compute_profile import collect_real_models, count_flops
    cfg, grid = _cfg("feature_pass"), _cfg("grid")
    models = ctx.hook("models", None) or collect_real_models(cfg, "cuda", grid)
    wanted = {"sqgan_encoder": "feature_pass_flops_sqgan_encoder", "sqgan_masker": "feature_pass_flops_sqgan_masker",
              "sqgan_decoder": "feature_pass_flops_sqgan_decoder", "segformer_teacher": "feature_pass_flops_segformer"}
    d, items = {}, {}
    for key, iid in wanted.items():
        if key not in models:
            items[iid] = item("fail", "%s not captured by the forward-pre-hook" % key)
            continue
        try:
            flops, method = count_flops(*models[key])
            d[key] = {"flops": flops, "method": method}
            items[iid] = item("pass" if flops > 0 else "fail", "%s flops %s via %s" % (key, flops, method))
        except Exception as exc:
            items[iid] = item("fail", "%s: %s" % (type(exc).__name__, exc))
    emb = [k for k in models if k.startswith("embed_")]
    bad = []
    for k in emb:
        try:
            flops, method = count_flops(*models[k])
            d[k] = {"flops": flops, "method": method}
            if flops <= 0:
                bad.append(k)
        except Exception as exc:
            bad.append("%s(%s)" % (k, exc))
    items["feature_pass_flops_embed_wrapper"] = item("pass" if emb and not bad else "fail", "%d wrappers, failing: %s" % (len(emb), bad))
    return {"details": d, "items": items}


# ------------------------------------------------------------------ 12 dataset_scale

def check_dataset_scale(ctx):
    import yaml
    from collections import defaultdict
    index = ctx.work / "dataset-a" / "index.jsonl"
    if not index.exists():
        raise Skip("pilot dataset A index.jsonl missing at %s" % index)
    rows = [json.loads(x) for x in index.read_text().splitlines() if x]
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/edge/data.yaml").read_text())
    nm = (cfg or {}).get("normals") or {}
    # defaults: 17 images/class (places365 has 365 classes, blacklist ignored), 3000 COCO people images
    full = {"crisismmd": 18082, "places365": int(nm.get("places365_per_class", 17)) * 365,
            "coco_people": int(nm.get("people_count", 3000))}
    acc = defaultdict(list)
    for r in rows:
        if r.get("bytes"):
            acc[r["source"]].append(r["bytes"])
    mean = {s: sum(v) / len(v) for s, v in acc.items()}
    est = 0.0
    for s, n in full.items():
        m = mean.get(s)
        if m is None:
            continue
        est += n * m
    gb = est / 1e9
    d = {"mean_bytes_per_source": mean, "images_per_source": {s: len(v) for s, v in acc.items()}, "full_counts": full,
         "estimate_gb": round(gb, 2), "limit_gb": FULL_DISK_GB,
         "headroom_gb": round(FULL_DISK_GB - gb, 2)}
    missing = [s for s in full if s not in mean]
    items = {"dataset_a_size_estimate": item("pass" if not missing and gb <= 0.9 * FULL_DISK_GB else "fail",
                                             "%.1f GB of %.0f GB; no pilot images for %s" % (gb, FULL_DISK_GB, missing))}
    stages = {}
    sr = ctx.work / "stage_results.json"
    if sr.exists():
        stages = {s["name"]: s for s in json.loads(sr.read_text())}
    a = stages.get("a_download_images")
    state = ctx.work / "state" / "A.done.json"           # parameters/edge/run_all.py writes <work>/state/<stage>.done.json
    if not (a and a.get("seconds")) and state.exists():
        try:
            rec = json.loads(state.read_text(encoding="utf-8"))
            a = {"seconds": rec.get("seconds"), "peak_rss_mb": rec.get("peak_rss_mb")}
        except (OSError, ValueError):
            a = None
    if a and a.get("seconds"):
        rate = len(rows) / max(a["seconds"], 1e-6)
        total = sum(full.values())
        d.update(pilot_images=len(rows), pilot_seconds=a["seconds"], images_per_second=round(rate, 3), peak_rss_mb=a.get("peak_rss_mb"),
                 projected_full_hours=round(total / rate / 3600, 2) if rate else None)
        rss = a.get("peak_rss_mb")
        items["workers16_memory"] = item("pass" if rss is not None else "skip",
                                         "peak child RSS %s MB with 8 workers; %.2f img/s; full download ~%.1f h" % (rss, rate, d["projected_full_hours"] or 0))
    else:
        items["workers16_memory"] = item("fail", "no a_download_images entry in stage_results.json and no state/A.done.json under %s (pilot stage did not run first)" % ctx.work)
    return {"details": d, "items": items}


# ------------------------------------------------------------------ 13 publish_roundtrip

def _tiny_dataset(root, n_files=300, n_dirs=5):
    root = Path(root)
    for i in range(n_files):
        folder = root / ("part%d" % (i % n_dirs))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ("f%04d.txt" % i)).write_text("pilot %d\n" % i)
    return root


def check_publish_roundtrip(ctx):
    from parameters.edge import kaggle_publish as kp
    publish = ctx.hook("publish", kp.publish_dataset)
    creds = ctx.hook("creds", None)
    if creds is None and "publish" not in ctx.inject:
        creds = kp.get_credentials()
    elif creds is None:
        creds = {"KAGGLE_API_TOKEN": "x"}
    present = sorted(creds or [])
    if not creds:
        raise Skip("no Kaggle secrets. " + SKIP_PUBLISH_HOWTO)
    d = {"secret_names_present": present}
    items = {"kaggle_secrets_present": item("pass", "secrets present: %s (values never printed)" % present),
             "api_token_env": item("pass" if "KAGGLE_API_TOKEN" in present else "fail",
                                   "env names used: %s" % present if "KAGGLE_API_TOKEN" in present else "legacy USERNAME/KEY pair only; token env path untested")}
    folder = _tiny_dataset(ctx.work / "publish_test")
    first = publish(str(folder), PUBLISH_SLUG, "Edge pilot publish test")
    (folder / "extra.txt").write_text("v2\n")
    second = publish(str(folder), PUBLISH_SLUG, "Edge pilot publish test v2", message="pilot version 2")
    d.update(first=first, second=second, files=300, folders=5)
    ok_first = first.get("status") in ("created", "versioned")
    items["dir_mode_zip_many_files"] = item("pass" if ok_first else "fail", "first publish: %s %s" % (first.get("status"), first.get("reason", "")))
    items["datasets_status_parse"] = item("pass" if second.get("status") == "versioned" else "fail",
                                          "second publish status %s (expected versioned) %s" % (second.get("status"), second.get("reason", "")))
    cli = ctx.hook("cli_publish", None)
    try:
        if cli is None:
            from Codec.edge import feature_pass
            small = _tiny_dataset(ctx.work / "publish_test_cli", 5, 1)
            feature_pass.main(["publish", "--out", str(small), "--slug", PUBLISH_SLUG])
        items["feature_pass_publish_cli"] = item("pass", "feature_pass publish ran without raising (see notebook log for its status dict)")
    except SystemExit:
        items["feature_pass_publish_cli"] = item("fail", "SystemExit from feature_pass publish")
    except Exception as exc:
        items["feature_pass_publish_cli"] = item("fail", repr(exc))
    return {"details": d, "items": items}


# ------------------------------------------------------------------ 14 discovery_note

def check_discovery_note(ctx):
    from Codec.edge import feature_pass
    from parameters.edge import build_label_dataset as bld
    base = Path(ctx.input_base)
    found = {}
    if base.is_dir():
        for marker in ("index.jsonl", "labels_raw.jsonl"):
            found[marker] = [str(p.parent) for p in base.rglob(marker)][:5]
    if not any(found.values()):
        raise Skip("no attached dataset with index.jsonl / labels_raw.jsonl under %s; the pilot builds A and B itself" % base)
    d, items = {"attached": found}, {}
    try:
        d["discover_images_root"] = str(bld.discover_images_root(str(base)))
        d["find_data_root_images"] = str(feature_pass.find_data_root("index.jsonl", str(base)))
        d["find_data_root_labels"] = str(feature_pass.find_data_root("labels_raw.jsonl", str(base)))
        items["kaggle_input_discovery"] = item("pass", "images %s labels %s" % (d["discover_images_root"], d["find_data_root_labels"]))
    except Exception as exc:
        items["kaggle_input_discovery"] = item("fail", "%s: %s" % (type(exc).__name__, exc))
    return {"details": d, "items": items}


# ------------------------------------------------------------------ feature_pass_smoke_verify (no re-run)

def check_feature_pass_smoke_verify(ctx):
    import numpy as np
    cache = ctx.work / "cache"
    shards = sorted((cache / "shards").glob("shard_????")) if (cache / "shards").is_dir() else []
    if not shards:
        raise Skip("no smoke-run shards under %s (stage c_feature_pass_smoke did not produce output)" % cache)
    cfg = _cfg("feature_pass")
    from Codec.edge.feature_pass import m_tag
    tags = [m_tag(m) for m in cfg["sqgan"]["m_levels"]]
    items, d = {}, {"shards": len(shards)}
    miss, bad_ms, stages, jpg_bad, jpg_n = [], [], {}, [], 0
    for s in shards:
        for name in ["done.json", "failures.json", "timings.jsonl", "phash64.npy", "sqgan_ok.npy", "class_grid.npy", "prefeat_meanmax.npy"] + \
                ["%s_%s" % (a, t) for t in tags for a in ("offsets", "kept_bitmap", "sts")]:
            candidates = [s / name, s / (name + ".npy")]
            if not any(c.exists() for c in candidates):
                miss.append("%s/%s" % (s.name, name))
        for line in (s / "timings.jsonl").read_text().splitlines() if (s / "timings.jsonl").exists() else []:
            r = json.loads(line)
            stages.setdefault(r["stage"], []).append(r["ms"])
            if not (r["ms"] >= 0):
                bad_ms.append((r["stage"], r["ms"]))
        for jpg in list(s.glob("recon_*/*.jpg")):
            jpg_n += 1
            try:
                from PIL import Image
                with Image.open(jpg) as im:
                    im.load()
            except Exception:
                jpg_bad.append(str(jpg))
    hashes = all((s / "phash64.npy").exists() and (s / "simhash64.npy").exists() for s in shards)
    items["feature_pass_shard_files"] = item("pass" if not miss else "fail", "missing: %s" % miss[:4])
    items["feature_pass_hashes_present"] = item("pass" if hashes else "fail", "phash64+simhash64 in every shard = %s" % hashes)
    need = ["load", "phash", "sqgan_segment"] + ["sqgan_encode_" + t for t in tags] + ["decode_" + t for t in tags] + ["recon_save"]
    absent = [n for n in need if n not in stages]
    items["feature_pass_timings_stages"] = item("pass" if not absent and not bad_ms else "fail", "absent stages %s, negative %s" % (absent, bad_ms[:3]))
    mean = {k: float(np.mean(v)) for k, v in stages.items()}
    d["stage_mean_ms"] = mean
    dec = [mean[k] for k in mean if k.startswith("decode_")]
    sane = bool(dec) and "phash" in mean and min(dec) > mean["phash"]
    items["feature_pass_timings_sane"] = item("pass" if sane else "fail", "decode mean ms %s vs phash %.2f ms" % ([round(x, 1) for x in dec], mean.get("phash", -1)))
    items["feature_pass_recon_jpegs"] = item("pass" if jpg_n and not jpg_bad else "fail", "%d recon jpegs, %d undecodable" % (jpg_n, len(jpg_bad)))
    prof = cache / "compute_profile.json"
    if prof.exists():
        p = json.loads(prof.read_text())
        zero = [k for k, v in p.items() if isinstance(v, dict) and "flops" in v and not (v["flops"] and v["flops"] > 0)]
        d["compute_profile_models"] = sorted(p)
        items["feature_pass_compute_profile_flops"] = item("pass" if not zero and p else "fail", "flops<=0 for %s" % zero)
    else:
        items["feature_pass_compute_profile_flops"] = item("fail", "compute_profile.json missing")
    seg_missing, seg_bad, seg_mismatch, seg_checked, seg_mb = [], [], 0, 0, 0.0
    for s in shards:
        if not (s / "sqgan_ok.npy").exists():
            continue                                    # SQ-GAN stage did not run in this shard
        sf = s / "segmap_u8.npz"
        if not sf.exists():
            seg_missing.append(s.name)
            continue
        seg_mb += sf.stat().st_size / 1e6
        try:
            from parameters.edge.sts import majority_vote_grid
            seg = np.load(sf)["seg"]
            grid = np.load(s / "class_grid.npy")
            ok = np.load(s / "sqgan_ok.npy").reshape(len(seg), -1).any(axis=1)
            if seg.shape != (len(grid), 256, 512) or seg.dtype != np.uint8 or (seg.size and int(seg.max()) > 19):
                seg_bad.append("%s shape %s dtype %s" % (s.name, seg.shape, seg.dtype))
                continue
            for i in np.flatnonzero(ok):
                seg_checked += 1
                if not np.array_equal(majority_vote_grid(seg[i], 256 // grid.shape[1]), grid[i]):
                    seg_mismatch += 1
        except Exception as exc:
            seg_bad.append("%s: %r" % (s.name, exc))
    d.update(segmap_mb=round(seg_mb, 3), segmap_images_checked=seg_checked, segmap_grid_mismatches=seg_mismatch)
    items["feature_pass_segmap_saved"] = item("pass" if not seg_missing and not seg_bad else "fail",
                                              "segmap_u8.npz missing in %s; bad (shape/dtype/values) %s; %.2f MB" % (seg_missing, seg_bad[:3], seg_mb))
    items["feature_pass_segmap_matches_grid"] = item("pass" if not seg_missing and not seg_bad and not seg_mismatch else "fail",
                                                     "%d of %d sqgan_ok images mismatch the saved 16x32 grid" % (seg_mismatch, seg_checked))
    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    n_ok =json.loads((cache / "summary.json").read_text())["counts"].get("ok", 0) if (cache / "summary.json").exists() else 0
    labels = ctx.work / "dataset-b" / "labels_raw.jsonl"
    full_n = sum(1 for _ in open(labels, encoding="utf-8")) if labels.exists() else 27082
    proj = total / max(n_ok, 1) * full_n / 1e9
    d.update(cache_bytes=total, n_ok=n_ok, full_images=full_n, projected_gb=round(proj, 1))
    items["feature_pass_disk_projection"] = item("pass" if n_ok and proj <= 0.9 * FULL_DISK_GB else "fail",
                                                 "smoke %.1f MB / %d images -> %.1f GB for %d images (limit %.0f GB)" % (total / 1e6, n_ok, proj, full_n, FULL_DISK_GB))
    return {"details": d, "items": items}


# ------------------------------------------------------------------ registry

CHECKS = {
    "env_gpu": check_env_gpu, "bundle_restore": check_bundle_restore, "source_root": check_source_root,
    "masker_format": check_masker_format, "feature_pass_codec_semantics": check_feature_pass_codec_semantics,
    "sqgan_latency": check_sqgan_latency, "embedders": check_embedders, "judge_dino": check_judge_dino,
    "ade_teacher": check_ade_teacher, "finetune_source": check_finetune_source, "finetune_pilot": check_finetune_pilot,
    "payload_semantic_map": check_payload_semantic_map, "bundle_deep_scan": check_bundle_deep_scan,
    "compute_profile_real": check_compute_profile_real, "dataset_scale": check_dataset_scale,
    "publish_roundtrip": check_publish_roundtrip, "discovery_note": check_discovery_note,
    "feature_pass_smoke_verify": check_feature_pass_smoke_verify,
}

# item id -> (description, source report, check name)
UNVERIFIED_ITEMS = {
    "gpu_count": ("GPU count visible to torch", "pilot", "env_gpu"),
    "gpu_names": ("GPU model names", "pilot", "env_gpu"),
    "torch_cuda": ("torch.cuda available", "pilot", "env_gpu"),
    "free_disk": ("free disk on the notebook", "pilot", "env_gpu"),
    "library_versions": ("library versions importable", "pilot", "env_gpu"),
    "dataparallel_t4x2": ("DataParallel works on 2 GPUs", "SQ-GAN fine-tune", "env_gpu"),
    "bundle_flat_restore": ("flat v0 bundle restores and passes bundle_check", "feature-pass WB", "bundle_restore"),
    "sqgan_loads_strict": ("SQGANDigitalCodec(bundle) loads strictly on cuda", "codec worker", "bundle_restore"),
    "real_masker_output": ("real masker output format (sample_index, alpha, STS)", "codec table", "masker_format"),
    "decode_latency_gpu": ("SQ-GAN batch decode latency on GPU", "sqgan_profile", "sqgan_latency"),
    "vram_peak": ("SQ-GAN peak VRAM", "sqgan_profile", "sqgan_latency"),
    "gflops_sqgan": ("SQ-GAN encoder+decoder GFLOPs", "compute_profile", "sqgan_latency"),
    "load_embedder_clip": ("clip_b32 loads and matches yaml dim", "embedding grid", "embedders"),
    "load_embedder_siglip2": ("siglip2_base loads and matches yaml dim", "embedding grid", "embedders"),
    "load_embedder_dinov2_small": ("dinov2_small loads and matches yaml dim", "embedding grid", "embedders"),
    "load_embedder_dinov2_base": ("dinov2_base loads and matches yaml dim", "embedding grid", "embedders"),
    "load_embedder_convnextv2": ("convnextv2_tiny loads and matches yaml dim", "embedding grid", "embedders"),
    "judge_dino_embeddings": ("judge extract_dino_embeddings + probe + fidelity", "judge", "judge_dino"),
    "ade_id2label_match": ("ADE20K teacher id2label equals seg_classes.yaml names", "SQ-GAN fine-tune", "ade_teacher"),
    "seg_class_map_real": ("project_logits on real SegFormer logits", "SQ-GAN fine-tune", "ade_teacher"),
    "pinned_sha_fetch": ("git fetch of pinned SemanticLossless commit", "SQ-GAN fine-tune", "finetune_source"),
    "upstream_zip_extract": ("extract .upstream/sqgan from source dataset zip", "SQ-GAN fine-tune", "finetune_source"),
    "trainer_launch": ("real trainer pilot stage launches", "SQ-GAN fine-tune", "finetune_pilot"),
    "trainer_small_image_rejection": ("trainer rejects 256x256, small_image_policy handles it", "SQ-GAN fine-tune", "finetune_pilot"),
    "payload_reader_gap": ("payload reader vs seg_map.project_logits (known gap)", "SQGAN_FINETUNE_HANDOFF", "payload_semantic_map"),
    "bundle_deep_key_scan": ("bundle_check --deep key scan of last.pt", "bundle_check", "bundle_deep_scan"),
    "dataset_a_size_estimate": ("dataset A full size fits 20 GB", "dataset A", "dataset_scale"),
    "workers16_memory": ("download memory / throughput at pilot scale", "dataset A", "dataset_scale"),
    "kaggle_secrets_present": ("Kaggle secrets available in the notebook", "kaggle_publish", "publish_roundtrip"),
    "datasets_status_parse": ("`datasets status` parse picks the versioned branch", "kaggle_publish", "publish_roundtrip"),
    "dir_mode_zip_many_files": ("--dir-mode zip uploads 300 files in 5 folders", "kaggle_publish", "publish_roundtrip"),
    "api_token_env": ("KAGGLE_API_TOKEN env var accepted by the CLI", "kaggle_publish", "publish_roundtrip"),
    "kaggle_input_discovery": ("auto-discovery of attached datasets", "label builder / feature pass", "discovery_note"),
    "feature_pass_restore_flat_bundle_pattern": ("restore_flat_bundle filename pattern vs real flat bundle", "feature-pass WB", "bundle_restore"),
    "feature_pass_find_source_root": ("find_source_root heuristic on real source dataset", "feature-pass WB", "source_root"),
    "feature_pass_m_mapping": ("image_fraction=semantic_fraction=m is the right m mapping", "feature-pass WB", "feature_pass_codec_semantics"),
    "feature_pass_sample_index_range": ("masker sample_index within 0..511", "feature-pass WB", "feature_pass_codec_semantics"),
    "feature_pass_prefeat_grid_shape": ("pre-mask grid is 256x16x32 (pooled 512)", "feature-pass WB", "feature_pass_codec_semantics"),
    "feature_pass_encoder_once_per_m": ("encoder runs once per m", "feature-pass WB", "feature_pass_codec_semantics"),
    "feature_pass_grid_m_independence": ("pre-mask grid identical across m levels", "feature-pass WB", "feature_pass_codec_semantics"),
    "feature_pass_flops_sqgan_encoder": ("FLOPs hook on real SQ-GAN encoder", "feature-pass WB", "compute_profile_real"),
    "feature_pass_flops_sqgan_masker": ("FLOPs hook on real SQ-GAN masker", "feature-pass WB", "compute_profile_real"),
    "feature_pass_flops_sqgan_decoder": ("FLOPs hook on real SQ-GAN decoder", "feature-pass WB", "compute_profile_real"),
    "feature_pass_flops_segformer": ("FLOPs hook on SegFormer call", "feature-pass WB", "compute_profile_real"),
    "feature_pass_flops_embed_wrapper": ("FLOPs on embedding-encoder Vision wrapper", "feature-pass WB", "compute_profile_real"),
    "feature_pass_batch_latency_irregular": ("batch latency replication on irregular inputs", "feature-pass WB", "sqgan_latency"),
    "feature_pass_publish_cli": ("`feature_pass publish` calls publish_dataset", "feature-pass WB", "publish_roundtrip"),
    "feature_pass_shard_files": ("smoke shards contain every expected file", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_hashes_present": ("pHash/SimHash arrays present", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_timings_stages": ("timings.jsonl has every stage, non-negative ms", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_timings_sane": ("CUDA-synced decode ms >> pHash ms", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_recon_jpegs": ("recon JPEGs decode", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_compute_profile_flops": ("compute_profile.json flops > 0 per model", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_disk_projection": ("full-dataset cache fits /kaggle/working", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_segmap_saved": ("full 256x512 uint8 teacher class map saved per shard (shape, dtype, <=19)", "feature-pass WB", "feature_pass_smoke_verify"),
    "feature_pass_segmap_matches_grid": ("majority-vote grid recomputed from the saved map equals class_grid.npy", "feature-pass WB", "feature_pass_smoke_verify"),
}

# order matters: checks that read pilot outputs run after the stages that make them
CHECK_ORDER = ["env_gpu", "bundle_restore", "source_root", "bundle_deep_scan", "payload_semantic_map", "masker_format",
               "feature_pass_codec_semantics", "sqgan_latency", "compute_profile_real", "embedders", "judge_dino",
               "ade_teacher", "finetune_source", "finetune_pilot", "publish_roundtrip", "discovery_note",
               "dataset_scale", "feature_pass_smoke_verify"]


def covers(name):
    return sorted(i for i, (_, _, c) in UNVERIFIED_ITEMS.items() if c == name)


def run_check(name, ctx):
    """Run one check; never raises. Returns (and writes) the result dict."""
    t0 = time.time()
    res = {"name": name, "status": "pass", "details": {}, "error": "", "covers": covers(name), "items": {}}
    try:
        out = CHECKS[name](ctx) or {}
        res["details"], res["items"] = out.get("details", {}), out.get("items", {})
        if "status" in out:
            res["status"] = out["status"]
        elif any(v["status"] == "fail" for v in res["items"].values()):
            res["status"] = "fail"
        elif res["items"] and all(v["status"] == "skip" for v in res["items"].values()):
            res["status"] = "skip"
    except Skip as exc:
        res["status"], res["error"] = "skip", str(exc)
        res["items"] = {i: item("skip", exc) for i in res["covers"]}
    except BaseException as exc:                         # incl. SystemExit from a CLI-style helper
        res["status"], res["error"] = "fail", _tail(traceback.format_exc())
        res["items"] = {i: item("fail", "%s: %s" % (type(exc).__name__, str(exc)[:200])) for i in res["covers"]}
    res["seconds"] = round(time.time() - t0, 2)
    folder = Path(ctx.out) / "checks"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + ".json")).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def reason_line(res):
    """Why a check is not a plain pass: full first line of its error, else the first failing/skipped item detail."""
    if res["status"] == "pass":
        return ""
    err = (res.get("error") or "").strip()
    if err and res["status"] == "skip":
        return err.splitlines()[0]
    if err:                                      # a failure carries a traceback: its last line names the exception
        lines = [x for x in err.splitlines() if x.strip()]
        return lines[-1] if lines else ""
    for name, it in res.get("items", {}).items():
        if it["status"] == res["status"]:
            return "%s: %s" % (name, str(it["detail"]).splitlines()[0] if it["detail"] else "")
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", help="check name, 'all' or 'list'")
    ap.add_argument("--out", default="/kaggle/working/pilot")
    a = ap.parse_args(argv)
    if a.check == "list":
        print("\n".join(CHECK_ORDER))
        return 0
    ctx = Ctx(out=Path(a.out))
    names = CHECK_ORDER if a.check == "all" else [a.check]
    for n in names:
        if n not in CHECKS:
            print("unknown check", n, file=sys.stderr)
            return 2
        r = run_check(n, ctx)
        print("[check] %-30s %s (%.1fs) %s" % (n, r["status"].upper(), r["seconds"], reason_line(r)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
