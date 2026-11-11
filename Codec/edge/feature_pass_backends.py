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
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np

from Evaluation.edge.edge_config import load_config


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


KAGGLE_INPUT = "/kaggle/input"
DATASET_DEPTH = 5


def _dirs_to_depth(base, depth):
    """Yield (dir_path, depth) for every directory under ``base`` down to ``depth`` levels (1 = direct children)."""
    base = Path(base)
    if depth < 1 or not base.is_dir():
        return
    try:
        children = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return
    for p in children:
        yield p, 1
        for q, d in _dirs_to_depth(p, depth - 1):
            yield q, d + 1


def find_datasets(slug_fragment, base=KAGGLE_INPUT, depth=DATASET_DEPTH):
    """Mounted dataset directories whose lowercase name contains ``slug_fragment``, shallowest first.

    Works for both Kaggle mount layouts (/kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>)."""
    frag = str(slug_fragment).strip("*").lower()
    hits = [p for p, _ in _dirs_to_depth(base, depth) if frag in p.name.lower()]
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))


def find_dataset(slug_fragment, base=KAGGLE_INPUT, depth=DATASET_DEPTH):
    """The shallowest mounted dataset directory whose lowercase name contains ``slug_fragment`` (FileNotFoundError if none)."""
    hits = find_datasets(slug_fragment, base, depth)
    if hits:
        return hits[0]
    seen = [p.relative_to(base).as_posix() for p, _ in _dirs_to_depth(base, 3)][:20] if Path(base).is_dir() else []
    raise FileNotFoundError("no dataset directory matching %r under %s (searched depth %d%s; top of tree: %s)"
                            % (slug_fragment, base, depth, "" if Path(base).is_dir() else ", base does not exist", seen))


def resolve_dataset(configured, slug_fragment, base=KAGGLE_INPUT):
    """``configured`` when that folder exists, else the mounted dataset found by ``slug_fragment``."""
    if configured and Path(configured).is_dir():
        return Path(configured)
    return find_dataset(slug_fragment, base)


def is_source_root(root):
    root = Path(root)
    return all((root / f).exists() for f in ("src/__init__.py", "src/sqgan.py", "src/bitstream.py"))


def find_mounted_source_root(base=KAGGLE_INPUT, depth=DATASET_DEPTH):
    """First directory under ``base`` (depth ``depth``, either layout) that is an SQ-GAN source root, else None."""
    if is_source_root(base):
        return Path(base)
    for p, _ in _dirs_to_depth(base, depth):
        if (p / "src").is_dir() and is_source_root(p):
            return p
    return None


def find_upstream_sqgan(base=KAGGLE_INPUT, depth=DATASET_DEPTH):
    """(kind, path): ('dir', <.upstream/sqgan>) for an extracted folder, ('zip', <sqgan-pinned-source.zip>), or (None, None)."""
    for p, _ in _dirs_to_depth(base, depth):
        if p.name == "sqgan" and p.parent.name == ".upstream":
            return "dir", p
    if Path(base).is_dir():
        for dirpath, dirs, files in os.walk(base):
            if len(Path(dirpath).relative_to(base).parts) > depth:
                dirs[:] = []
                continue
            if "sqgan-pinned-source.zip" in files:
                return "zip", Path(dirpath) / "sqgan-pinned-source.zip"
    return None, None


def provide_upstream_sqgan(root, base=KAGGLE_INPUT, log=print):
    """Make ``<root>/.upstream/sqgan`` exist: from an extracted ``.upstream/sqgan`` folder (copied) or the pinned zip."""
    root = Path(root)
    target = root / ".upstream" / "sqgan"
    if target.is_dir() and any(target.iterdir()):
        return target
    kind, found = find_upstream_sqgan(base)
    if kind == "dir":
        if found.resolve() != target.resolve():
            shutil.copytree(found, target, dirs_exist_ok=True)
        log("upstream SQ-GAN: copied extracted folder %s -> %s" % (found, target))
    elif kind == "zip":
        with zipfile.ZipFile(found) as z:
            for name in z.namelist():
                if not (root / name).resolve().is_relative_to(root.resolve()):
                    raise ValueError("zip path escapes target: " + name)
            z.extractall(root)
        log("upstream SQ-GAN: extracted %s -> %s" % (found, root))
    else:
        raise FileNotFoundError("upstream SQ-GAN code not found: no .upstream/sqgan folder and no sqgan-pinned-source.zip "
                                "under %s (depth %d)" % (base, DATASET_DEPTH))
    return target


def resolve_source_root(cfg, log=print, clone=None, base=None):
    """(root, mode). (i) a mounted dataset holding the SemanticLossless src/; (ii) otherwise a clone of the pinned commit.

    Either way ``<root>/.upstream/sqgan`` is provided from the mounted source dataset when possible."""
    p = cfg["paths"]
    base = base or p.get("input_base", KAGGLE_INPUT)
    root = find_mounted_source_root(base)
    if root is not None:
        mode = "mounted dataset"
        try:
            provide_upstream_sqgan(root, base, log)
        except (OSError, ValueError) as exc:      # read-only mount without .upstream: the root may still work
            log("[warn] upstream SQ-GAN not provided for %s: %r" % (root, exc))
    else:
        from Codec.edge import sqgan_finetune as ft
        dest = Path(p["restore_dir"]).parent / "semlossless"
        ft_cfg = ft.load_config(Path(__file__).resolve().parents[2] / "config/edge/sqgan_finetune.yaml")
        try:
            root = Path((clone or ft.fetch_pinned)(ft_cfg["source"], dest))
        except Exception as exc:
            raise FileNotFoundError(
                "no SQ-GAN source root: no mounted dataset under %s has src/__init__.py + src/sqgan.py + src/bitstream.py, "
                "and the pinned git clone into %s failed: %s: %s" % (base, dest, type(exc).__name__, str(exc)[:300])) from exc
        mode = "git clone of pinned commit"
        provide_upstream_sqgan(root, base, log)
    log("SQ-GAN source root: %s (%s)" % (root, mode))
    return root, mode


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


def v1_bundle_dir(base=KAGGLE_INPUT, codec_cfg=None):
    """Mounted fine-tuned v1 bundle folder from ``codecs.v1`` of config/edge/codec.yaml (dataset slug + bundle_subdir)."""
    v1 = (codec_cfg or load_config("codec"))["codecs"]["v1"]
    slug = str(v1["kaggle_dataset"]).split("/")[-1]
    if not slug:
        raise ValueError("codecs.v1.kaggle_dataset is empty in config/edge/codec.yaml")
    folder = find_dataset(slug, base) / v1.get("bundle_subdir", "")
    if not (folder / "bundle.json").is_file():
        raise FileNotFoundError("no bundle.json in %s (codecs.v1.bundle_subdir=%r)" % (folder, v1.get("bundle_subdir")))
    return folder


class RealSQGAN:
    """Backend with segment / encode / decode (see feature_pass.Backends)."""

    def __init__(self, cfg, device):
        p = cfg["paths"]
        self.device = device
        base = p.get("input_base", KAGGLE_INPUT)
        if cfg["sqgan"].get("codec") == "v1":
            self.bundle = v1_bundle_dir(base)
        else:
            self.bundle = restore_flat_bundle(resolve_dataset(p.get("bundle_flat"), "sqgan-bundle", base), p["restore_dir"])
        self.source, self.source_mode = resolve_source_root(cfg, base=base)
        self.m_levels = list(cfg["sqgan"]["m_levels"])
        self._codecs = {}
        self._codec(self.m_levels[0])          # load codec + teacher now: a missing teacher skips the whole stage

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

    def quality(self, orig_arr, recon_arr):
        """LPIPS (alex) and SegFormer mIoU of the reconstruction against the original, both HxWx3 uint8."""
        from Codec.edge.codec_table import codec_seg_fn, recon_miou
        import torch
        if not hasattr(self, "_lpips"):
            import lpips
            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        x, y = self._tensor(orig_arr), self._tensor(recon_arr)
        with torch.no_grad():
            lp = float(self._lpips(x * 2 - 1, y * 2 - 1).mean())
        seg = codec_seg_fn(self._codec(self.m_levels[0]))
        return {"lpips": lp, "miou": recon_miou(seg, x, y)}

    def decode(self, payload, m):
        t = self._codec(m).decode(payload)
        return (t[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy())
