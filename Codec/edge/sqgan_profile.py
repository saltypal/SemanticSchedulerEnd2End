"""Measure SQ-GAN decode latency / encoder time / peak VRAM for the edge simulator.

Output ``latency_samples`` {B: [seconds,...]} is exactly what
``Evaluation.edge.contracts.LatencyProfile`` takes. The real SQ-GAN decoder handles ONE packet per call,
so, absent codec.decode_batch, batch size B is EMULATED by ``decode_batch_sequential`` (B sequential decodes inside one timed window).
Replace that named function with a true batched decoder when one exists; nothing else changes.
torch is imported lazily; with no CUDA the timing falls back to wall-clock.
"""
import argparse
import json
import time
from pathlib import Path


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def decode_batch_sequential(codec, payloads):
    """Emulated batch decode: decode each packet in turn (single-image decoder)."""
    return [codec.decode(p) for p in payloads]


def _timed(fn):
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    return time.perf_counter() - t0, out


def profile_codec(codec, images, batch_sizes, repeats, device="cpu", flops_fn=None, decode_batch=None):
    """Return dict: latency_samples, encoder_ms, peak_vram_bytes, gflops, masker_keys, emulated_batching."""
    if not images:
        raise ValueError("images required")
    if repeats < 1 or not batch_sizes:
        raise ValueError("repeats>=1 and batch_sizes required")
    if decode_batch is None and hasattr(codec, "decode_batch"):
        decode_batch = lambda c, ps: c.decode_batch(ps)   # true batching
        emulated = False
    else:
        emulated = decode_batch is None
        decode_batch = decode_batch or decode_batch_sequential
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available()) and str(device).startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    enc_times, payloads, masker_keys = [], [], set()
    for img in images:
        dt, enc = _timed(lambda img=img: codec.encode(img))
        enc_times.append(dt * 1e3)
        payloads.append(enc["payload"] if isinstance(enc, dict) else enc)
        if isinstance(enc, dict):
            masker_keys.update(k for k in enc if k != "payload")
    samples = {}
    for b in sorted(set(int(x) for x in batch_sizes)):
        batch = [payloads[i % len(payloads)] for i in range(b)]
        decode_batch(codec, batch)  # warm-up, not recorded
        samples[b] = [_timed(lambda: decode_batch(codec, batch))[0] for _ in range(repeats)]
    peak = int(torch.cuda.max_memory_allocated()) if cuda else None
    return {"latency_samples": samples, "encoder_ms": enc_times,
            "peak_vram_bytes": peak, "gflops": float(flops_fn(codec)) if flops_fn else None,
            "masker_keys": sorted(masker_keys), "emulated_batching": emulated,
            "mean_payload_bytes": sum(len(p) for p in payloads) / len(payloads)}


def save_profile(profile, path):
    out = dict(profile)
    out["latency_samples"] = {str(k): list(v) for k, v in profile["latency_samples"].items()}
    Path(path).write_text(json.dumps(out, indent=1))


def load_profile(path):
    d = json.loads(Path(path).read_text())
    d["latency_samples"] = {int(k): v for k, v in d["latency_samples"].items()}
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Profile the SQ-GAN decoder")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="sqgan_v0_profile.json")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image
    from Codec.integration.sqgan_payload import SQGANDigitalCodec
    codec = SQGANDigitalCodec(device=a.device)
    paths = sorted(p for p in Path(a.images_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[:a.n_images]
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((512, 256)), dtype=np.float32) / 255.0)
            .permute(2, 0, 1)[None] for p in paths]
    save_profile(profile_codec(codec, imgs, a.batch_sizes, a.repeats, a.device), a.out)


if __name__ == "__main__":
    main()
