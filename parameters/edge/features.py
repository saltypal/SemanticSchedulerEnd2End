"""UE-side perceptual keys and relevance (plan v3.1 sec. 4, steps S2/S2b).

Provides the 64-bit SimHash of a CLIP embedding (secret random hyperplanes), a numpy-only DCT pHash,
Hamming distance and mission relevance. CLIP extraction imports torch lazily. Seed comes from
``config/edge/header.yaml``.
"""
import hashlib
from functools import lru_cache

import numpy as np

from Evaluation.edge.edge_config import load_config


def default_seed():
    return int(load_config("header")["simhash_seed"])


@lru_cache(maxsize=16)
def _planes(seed, dim):
    return np.random.default_rng(int(seed)).standard_normal((64, int(dim)))


def _bits_to_int(bits):
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def simhash64(embedding, seed_secret=None):
    """64-bit SimHash: sign of 64 seeded random hyperplane projections. Returns a Python int."""
    e = np.asarray(embedding, dtype=np.float64).ravel()
    if e.size == 0:
        raise ValueError("empty embedding")
    seed = default_seed() if seed_secret is None else seed_secret
    return _bits_to_int(_planes(seed, e.size) @ e > 0)


@lru_cache(maxsize=4)
def _dct_matrix(n):
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n)[None, :] + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


def _resize_area(img, size):
    """Block-average downsample onto a size x size grid."""
    h, w = img.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    out = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            out[i, j] = img[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
    return out


def phash64(image_array_gray):
    """Standard pHash: 32x32 downsample, 2-D DCT-II, top-left 8x8 minus DC, threshold at median."""
    img = np.asarray(image_array_gray, dtype=np.float64)
    if img.ndim != 2 or min(img.shape) < 1:
        raise ValueError("phash64 expects a 2-D grayscale array")
    d = _dct_matrix(32)
    coef = (d @ _resize_area(img, 32) @ d.T)[:8, :8].ravel()[1:]   # 63 AC values
    bits = np.concatenate([[0], coef > np.median(coef)])           # DC slot fixed to 0 -> 64 bits
    return _bits_to_int(bits)


def hamming(a, b):
    """Number of differing bits between two non-negative ints."""
    return bin(int(a) ^ int(b)).count("1")


def relevance(image_emb, mission_emb):
    """Cosine similarity mapped to [0, 1] via (x + 1) / 2."""
    a = np.asarray(image_emb, dtype=np.float64).ravel()
    b = np.asarray(mission_emb, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("zero-norm embedding")
    return float(np.clip((float(a @ b) / (na * nb) + 1.0) / 2.0, 0.0, 1.0))


def extract_clip_embeddings(paths, model_name="openai/clip-vit-base-patch32", batch_size=64,
                            device="cpu"):
    """L2-normalised CLIP image embeddings, shape (len(paths), dim), float32.

    Heavy: imports torch/transformers/PIL lazily and loads the model; run on Kaggle, never in tests.
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            f = model.get_image_features(**inputs)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class FakeEmbedder:
    """Deterministic stand-in for CLIP in tests: key -> seeded unit vector."""

    def __init__(self, dim=512):
        self.dim = dim

    def __call__(self, keys):
        rows = []
        for k in keys:
            seed = int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32)
