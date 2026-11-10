"""Compute profile, measured ONCE per model: parameters, FLOPs of one forward at the real input size, peak VRAM
and batch-size latency samples. Writes ``compute_profile.json``.

FLOPs use ``torch.utils.flop_counter.FlopCounterMode``; if that fails a hook-based estimate (2*MACs of
Conv/Linear) is used and labelled ``flops_method: "hooks"``. pHash is timed on CPU (ms). Real models covered by
``main``: SegFormer teacher, SQ-GAN encoder / masker / decoder (inputs captured with forward pre-hooks during one
real encode/decode) and every embedding encoder of grid.yaml. Heavy imports are lazy; the generic helpers work
on any ``torch.nn.Module`` (used by the CPU tests).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from Evaluation.edge.edge_config import load_config


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def _call(module, args, kwargs):
    import torch
    with torch.no_grad():
        return module(*args, **kwargs)


def hook_flops(module, args, kwargs):
    """Fallback estimate: 2*MACs of Conv/Linear layers via forward hooks."""
    import torch.nn as nn
    total = [0]

    def conv(m, i, o):
        total[0] += 2 * o.numel() * (m.in_channels // m.groups) * int(np.prod(m.kernel_size))

    def lin(m, i, o):
        total[0] += 2 * o.numel() * m.in_features
    hs = []
    for m in module.modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            hs.append(m.register_forward_hook(conv))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin))
    try:
        _call(module, args, kwargs)
    finally:
        for h in hs:
            h.remove()
    return int(total[0])


def count_flops(module, args=(), kwargs=None):
    """(flops, method) for one forward."""
    kwargs = kwargs or {}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            _call(module, args, kwargs)
        return int(fc.get_total_flops()), "FlopCounterMode"
    except Exception:
        return hook_flops(module, args, kwargs), "hooks"


def _sync(device):
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _batched(args, b):
    """Repeat the leading dim of every tensor positional arg to batch size ``b``."""
    import torch
    return tuple(a.repeat(b, *([1] * (a.dim() - 1))) if torch.is_tensor(a) and a.dim() > 0 else a for a in args)


def profile_module(module, args=(), kwargs=None, device="cpu", batch_sizes=(1,), repeats=5, warmup=2,
                   clock=time.perf_counter):
    """Profile one module on ``args`` (already on ``device``). Returns a JSON-ready dict."""
    import torch
    kwargs = kwargs or {}
    module.eval()
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    flops, method = count_flops(module, args, kwargs)
    lat = {}
    for b in batch_sizes:
        try:
            a = _batched(args, int(b))
            for _ in range(warmup):
                _call(module, a, kwargs)
            samples = []
            for _ in range(repeats):
                _sync(device)
                t0 = clock()
                _call(module, a, kwargs)
                _sync(device)
                samples.append((clock() - t0) * 1000.0)
            lat[str(b)] = samples
        except Exception as exc:                      # irregular inputs: record, do not abort the profile
            lat[str(b)] = {"error": repr(exc)}
    return {"params": count_params(module), "flops": flops, "flops_method": method,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None,
            "latency_ms": lat, "device": str(device)}


def profile_phash(images_gray, n=50, clock=time.perf_counter):
    """CPU ms per pHash64 over up to ``n`` grayscale arrays."""
    from parameters.edge.features import phash64
    ms = []
    for img in list(images_gray)[:n]:
        t0 = clock()
        phash64(img)
        ms.append((clock() - t0) * 1000.0)
    return {"cpu_ms_mean": float(np.mean(ms)), "cpu_ms_samples": ms}


def save_profile(profile, out_path):
    Path(out_path).write_text(json.dumps(profile, indent=2), encoding="utf-8")


def profile_all(profile_cfg, models, out_path, force=False, phash_images=None):
    """``models``: {name: (module, args, kwargs)}. Computes each missing entry once and rewrites the JSON."""
    out_path = Path(out_path)
    prof = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
    for name, (module, args, kwargs) in models.items():
        if name in prof:
            continue
        prof[name] = profile_module(module, args, kwargs, profile_cfg.get("device", "cpu"),
                                    profile_cfg["batch_sizes"], profile_cfg["repeats"], profile_cfg["warmup"])
        save_profile(prof, out_path)
    if phash_images is not None and "phash" not in prof:
        prof["phash"] = profile_phash(phash_images, profile_cfg.get("phash_samples", 50))
        save_profile(prof, out_path)
    return prof


# ------------------------------------------------------------------ real models (GPU; not tested)

def _capture_inputs(module):
    """Forward pre-hook that stores the (args, kwargs) of the module's first call."""
    box = {}

    def pre(m, args, kwargs):
        box.setdefault("v", (tuple(a.detach() if hasattr(a, "detach") else a for a in args),
                             {k: (v.detach() if hasattr(v, "detach") else v) for k, v in kwargs.items()}))
    h = module.register_forward_pre_hook(pre, with_kwargs=True)
    return box, h


def collect_real_models(cfg, device, grid_cfg):
    """Build the real model dict. Heavy: loads SQ-GAN, SegFormer and every encoder that loads."""
    import torch
    import torch.nn as nn
    from Codec.edge.feature_pass_backends import RealSQGAN
    from parameters.edge.embedding_grid import load_embedder
    models = {}
    sq = RealSQGAN(cfg, device)
    codec = sq._codec(sq.m_levels[0])
    img = torch.rand(1, 3, 256, 512, device=device)
    sem = codec.semantic_map(img)
    boxes = {n: _capture_inputs(getattr(codec.model, n)) for n in ("encoder", "masker")}
    payload = codec.encode(img, sem)
    dbox = _capture_inputs(codec.model.decoder)
    codec.decode(payload)
    for n, (box, h) in list(boxes.items()) + [("decoder", dbox)]:
        h.remove()
        if "v" in box:
            models["sqgan_" + n] = (getattr(codec.model, n), box["v"][0], box["v"][1])
    models["segformer_teacher"] = (codec._teacher, (), {"pixel_values": (img - 0.45) / 0.225})

    class Vision(nn.Module):
        def __init__(self, model, family):
            super().__init__()
            self.model, self.family = model, family

        def forward(self, pixel_values):
            if self.family in ("clip", "siglip"):
                return self.model.get_image_features(pixel_values=pixel_values)
            return self.model(pixel_values=pixel_values).last_hidden_state
    for e in grid_cfg["encoders"]:
        try:
            emb = load_embedder(e["name"], device, grid_cfg)
            models["embed_" + e["name"]] = (Vision(emb.model, e["family"]).eval(),
                                            (torch.rand(1, 3, 224, 224, device=device),), {})
        except Exception as exc:
            models["embed_" + e["name"] + "_SKIPPED"] = None
            print("[skip] profile of %s: %r" % (e["name"], exc))
    return {k: v for k, v in models.items() if v is not None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="folder; compute_profile.json is written inside")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg, grid_cfg = load_config("feature_pass"), load_config("grid")
    out = Path(a.out or cfg["paths"]["out"])
    out.mkdir(parents=True, exist_ok=True)
    pc = dict(cfg["profile"], device=a.device)
    rng = np.random.default_rng(0)
    gray = [rng.random((128, 128)) * 255 for _ in range(pc["phash_samples"])]
    models = collect_real_models(cfg, a.device, grid_cfg)
    prof = profile_all(pc, models, out / "compute_profile.json", a.force, gray)
    print(json.dumps({k: {"params": v.get("params"), "flops": v.get("flops")} for k, v in prof.items()}, indent=1))


if __name__ == "__main__":
    main()
