"""Real (GPU) model access for Codec.edge.feature_pass. Heavy imports are lazy; nothing here runs in tests.

* ``embedder_factory`` wraps parameters.edge.embedding_grid.load_embedder (one lazy model per encoder).
* ``restore_flat_bundle`` rebuilds the verified SQ-GAN bundle folder from the flat Kaggle dataset
  (same scheme as Scheduler/training/study/build_paired_notebooks.py: scheduler_sqgan_flat_manifest.json).
* ``RealSQGAN`` drives Codec.integration.sqgan_payload.SQGANDigitalCodec: teacher segmentation once per image,
  then per m one encode with the pre-mask grid (sqgan_prefeatures.prefeature_capture) and kept tokens
  (codec_table.masker_capture) captured in the SAME encode call. One codec object per m (the keep fraction is a
  constructor option); the encoder grid does not depend on m so the first m's grid is the one cached.
"""
import json
import shutil
from pathlib import Path

import numpy as np


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def embedder_factory(device, grid_cfg):
    from parameters.edge.embedding_grid import load_embedder
    return lambda name: load_embedder(name, device, grid_cfg)


def not_found(directory, pattern, what, limit=15):
    """FileNotFoundError naming the directory searched, the pattern and up to ``limit`` names found there."""
    d = Path(directory)
    try:
        names = sorted(p.name for p in d.iterdir())[:limit] if d.is_dir() else []
    except OSError:
        names = []
    state = "" if d.is_dir() else " (directory does not exist)"
    return FileNotFoundError("%s: nothing matching %r in %s%s; files found there (up to %d): %s"
                             % (what, pattern, d, state, limit, names))


def find_source_root(source_dir, max_depth=5):
    """Directory under ``source_dir`` holding src/__init__.py + src/sqgan.py + src/bitstream.py."""
    base = Path(source_dir)
    for init in sorted(base.rglob("src/__init__.py"), key=lambda p: len(p.parts)):
        root = init.parent.parent
        if len(root.relative_to(base).parts) <= max_depth and (root / "src/sqgan.py").exists() \
                and (root / "src/bitstream.py").exists():
            return root
    raise not_found(base, "src/__init__.py + src/sqgan.py + src/bitstream.py", "no SQ-GAN source root")


def restore_flat_bundle(flat_dir, dest):
    """Copy the flat dataset members into ``dest/bundle`` with their original relative paths."""
    flat, dest = Path(flat_dir), Path(dest)
    manifests = sorted(flat.glob("*flat_manifest.json")) if flat.is_dir() else []
    if not manifests:
        extra = " (a bundle_manifest.json alone is not the flat mapping)" if (flat / "bundle_manifest.json").exists() else ""
        raise not_found(flat, "*flat_manifest.json", "cannot restore flat SQ-GAN bundle" + extra)
    mapping = json.loads(manifests[0].read_text())
    bundle = dest / "bundle"
    if (bundle / "bundle.json").exists():
        return bundle
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(flat / mapping["bundle_manifest"], bundle / "bundle.json")
    for rel, member in mapping["files"].items():
        name = member["file"] if isinstance(member, dict) else member
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(flat / name, target)
    return bundle


class RealSQGAN:
    """Backend with segment / encode / decode (see feature_pass.Backends)."""

    def __init__(self, cfg, device):
        p = cfg["paths"]
        self.device = device
        self.bundle = restore_flat_bundle(p["bundle_flat"], p["restore_dir"])
        self.source = find_source_root(p["sqgan_source"])
        self.m_levels = list(cfg["sqgan"]["m_levels"])
        self._codecs = {}

    def _codec(self, m):
        if m not in self._codecs:
            from Codec.integration.sqgan_payload import SQGANDigitalCodec
            self._codecs[m] = SQGANDigitalCodec(bundle=self.bundle, source_root=self.source, device=self.device,
                                                image_fraction=float(m), semantic_fraction=float(m))
        return self._codecs[m]

    def _tensor(self, arr):
        import torch
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float().div(255.0).to(self.device)

    def segment(self, arr):
        x = self._tensor(arr)
        sem = self._codec(self.m_levels[0]).semantic_map(x)
        return sem.argmax(1)[0].cpu().numpy().astype(np.int64), (x, sem)

    def encode(self, arr, handle, m):
        from Codec.edge.codec_table import masker_capture
        from Codec.integration.sqgan_prefeatures import prefeature_capture
        x, sem = handle
        codec = self._codec(m)
        with prefeature_capture(codec) as (feat, calls), masker_capture(codec) as mk:
            payload = codec.encode(x, sem)
        if len(calls) != 1 or "kept" not in mk:
            raise RuntimeError("expected one encoder call and a masker sample_index")
        return {"payload": payload, "kept": mk["kept"], "features": feat["features"][0].float().cpu().numpy()}

    def decode(self, payload, m):
        t = self._codec(m).decode(payload)
        return (t[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy())
