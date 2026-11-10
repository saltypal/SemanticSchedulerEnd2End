"""SQ-GAN codec v1 fine-tune: config -> train-only manifest -> trainer -> verified bundle.

Entry point: ``python -m Codec.edge.sqgan_finetune --config config/edge/sqgan_finetune.yaml``.
The trainer is the teammate's Semantic_Lossless ``src/sqgan_train.py`` (pinned commit, read-only); it is
run by ``Codec/edge/sqgan_driver.py`` in a subprocess because its top-level package is also called ``src``.
Every stage here is a small function; the trainer is an injectable callable ``trainer(options, run_dir, ctx)``
so the plumbing is testable without GPU or weights. Reuses ``parameters.edge.splits.assign_event_splits``
(provisional splits), ``Codec.edge.seg_map`` (class map) and ``Codec.edge.bundle_check`` (verification).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from Codec.edge import bundle_check
from Codec.integration.sqgan_payload import COMPONENTS, sha256

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- config
def load_config(path, overrides=()):
    """Load YAML; ``overrides`` are 'a.b.c=value' strings (value parsed as YAML)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError("unknown config key: " + key)
        node[parts[-1]] = yaml.safe_load(value)
    for section in ("source", "inputs", "data", "teacher", "train", "output", "codec"):
        if section not in cfg:
            raise ValueError("config missing section: " + section)
    if cfg["data"]["small_image_policy"] not in ("skip", "upscale"):
        raise ValueError("data.small_image_policy must be skip or upscale")
    return cfg


def discover_root(search_root, marker, depth=3):
    """Folder under ``search_root`` (<= ``depth`` levels deep) that directly contains ``marker``."""
    base = Path(search_root)
    hits = []
    if base.is_dir():
        for d in range(depth + 1):
            hits += [p.parent for p in base.glob("/".join(["*"] * d + [marker]))]
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError("no folder containing %s under %s (searched %d levels); pass the root explicitly"
                                % (marker, search_root, depth))
    if len(hits) > 1:
        raise ValueError("several folders contain %s: %s; pass the root explicitly" % (marker, [str(h) for h in hits]))
    return hits[0]


# ----------------------------------------------------------------------------- splits
def read_labels(labels_root, name="labels_raw.jsonl"):
    path = Path(labels_root) / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if not r.get("uid") or not r.get("file"):
            raise ValueError("label row without uid/file: %r" % (r.get("image_id"),))
    return rows


def load_splits(path):
    """uid -> split from splits_v1.json. Accepts {uid: split}, {"splits": {uid: split}} or {split: [uid, ...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = data.get("splits", data)
    if set(data) <= set(SPLITS):
        data = {uid: s for s, uids in data.items() for uid in uids}
    bad = {v for v in data.values()} - set(SPLITS)
    if bad:
        raise ValueError("unknown split names in splits file: %s" % sorted(bad))
    return data


def provisional_splits(rows, data_cfg=None):
    """Event-disjoint split from config/edge/data.yaml (not the frozen dedup split). Unlisted events -> excluded."""
    from Evaluation.edge.edge_config import load_config as load_edge
    from parameters.edge.splits import assign_event_splits
    cfg = data_cfg or load_edge("data")["splits"]
    listed = set(cfg["train_events"]) | set(cfg["val_events"]) | set(cfg["test_events"])
    keep = [dict(r, image_path=r["uid"]) for r in rows if not r.get("event") or r["event"] in listed]
    by_path, _ = assign_event_splits(keep, cfg)
    return {r["uid"]: by_path[r["uid"]] for r in keep}


def resolve_splits(rows, splits_path, allow_provisional):
    """Frozen splits if the file exists; else refuse unless provisional is allowed (loud warning)."""
    if splits_path and Path(splits_path).is_file():
        return load_splits(splits_path), "frozen"
    if not allow_provisional:
        raise FileNotFoundError("splits file not found (%s). Refusing to train without frozen splits; "
                                "pass --allow-provisional-splits to derive train events from config/edge/data.yaml" % splits_path)
    print("!" * 78 + "\n!! WARNING: PROVISIONAL SPLITS (config/edge/data.yaml events), NOT the frozen dedup splits.\n"
          "!! Near-duplicates of val/test images may be in train. Do not publish results from this bundle.\n" + "!" * 78,
          flush=True)
    return provisional_splits(rows), "provisional"


def select_train_rows(rows, splits, sources, max_images=None):
    """Only rows whose uid is assigned 'train' AND whose source is enabled. Rows without a split are dropped."""
    out = [r for r in rows if splits.get(r["uid"]) == "train" and r["source"] in sources]
    out.sort(key=lambda r: r["uid"])
    return out[:max_images] if max_images else out


# ----------------------------------------------------------------------------- manifest (Semantic_Lossless dataset.py format)
def image_size(path):
    """(width, height) after EXIF orientation, matching ImageOps.exif_transpose in the trainer's loader."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112) in (5, 6, 7, 8):
            w, h = h, w
    return w, h


def _upscale(src, dst, min_w, min_h):
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        k = max(min_w / im.width, min_h / im.height)
        im = im.resize((max(min_w, round(im.width * k)), max(min_h, round(im.height * k))), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "PNG")


def build_manifest(train_rows, images_root, data_cfg, work_dir):
    """Manifest dict readable by Semantic_Lossless ParentImages (dataset.py:106-131). Reads ONLY ``train_rows`` files.

    Row ``file`` is relative to the images dataset root, e.g. images/places365/x.jpg. The 'val' entry is one
    TRAIN image re-labelled val because the trainer's pilot stage evaluates one 'val' image (evaluate.py:110).
    """
    images_root, work_dir = Path(images_root), Path(work_dir)
    min_w, min_h = int(data_cfg["min_width"]), int(data_cfg["min_height"])
    records, skipped = [], []
    for r in train_rows:
        src, rel = r["source"], r["file"]
        prefix = "images/%s/" % src
        if not rel.startswith(prefix):
            raise ValueError("file %r does not start with %r" % (rel, prefix))
        path = images_root / rel
        w, h = image_size(path)
        source, relative = src, rel[len(prefix):]
        if w < min_w or h < min_h:
            if data_cfg["small_image_policy"] == "skip":
                skipped.append(r["uid"])
                continue
            dst = work_dir / "upscaled" / (hashlib.sha256(r["uid"].encode()).hexdigest()[:16] + ".png")
            _upscale(path, dst, min_w, min_h)
            path, source, relative = dst, "upscaled", dst.name
            w, h = image_size(path)
        records.append({"source": source, "split": "train", "relative_path": relative, "sha256": sha256(path),
                        "width": w, "height": h, "uid": r["uid"]})
    if not records:
        raise ValueError("no train images >= %dx%d left (skipped %d); see data.small_image_policy" % (min_w, min_h, len(skipped)))
    records.append(dict(records[0], split="val", pilot_smoke_only=True))
    identity = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    roots = {s: str(images_root / "images" / s) for s in {r["source"] for r in records} if s != "upscaled"}
    if any(r["source"] == "upscaled" for r in records):
        roots["upscaled"] = str(work_dir / "upscaled")
    return {"protocol": "edge-sqgan-v1-train-only", "identity": identity, "roots": roots,
            "counts": {"train": len(records) - 1, "val": 1, "test": 0}, "images": records,
            "skipped_too_small": skipped, "test_used_for_training": False}


def assert_train_only(manifest, splits):
    """Fail unless every manifest image (including the pilot smoke row) is a train uid in ``splits``."""
    for r in manifest["images"]:
        if splits.get(r["uid"]) != "train":
            raise AssertionError("non-train image in training manifest: %s (%s)" % (r["uid"], splits.get(r["uid"])))
    if any(r["split"] == "test" for r in manifest["images"]):
        raise AssertionError("test split present in manifest")


# ----------------------------------------------------------------------------- trainer plumbing
def trainer_options(cfg, manifest_path, checkpoint_dir, n_gpus):
    """Options dict for sqgan_train.execute (keys cited in config/edge/sqgan_finetune.yaml)."""
    t, tc = cfg["train"], cfg["teacher"]
    dp = (n_gpus >= 2) if t["data_parallel"] == "auto" else bool(t["data_parallel"])
    if dp and int(t["batch_size"]) < 2:
        raise ValueError("data_parallel requires batch_size >= 2 (sqgan_train.py:130)")
    return {"algorithm": "sqgan", "manifest": str(manifest_path), "checkpoint": str(checkpoint_dir),
            "teacher_model": tc["model"], "teacher_revision": tc["revision"], "seed": int(t["seed"]),
            "device": "cuda:0" if n_gpus else "cpu", "batch_size": int(t["batch_size"]), "data_parallel": dp,
            "learning_rate": float(t["learning_rate"]), "epochs": int(t["epochs"]), "max_seconds": float(t["max_seconds"]),
            "image_fraction": float(t["image_fraction"]), "semantic_fraction": float(t["semantic_fraction"]),
            "crop": 256, "evaluation_tile": [256, 512]}


def _git(args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def obtain_source(cfg, git=_git):
    """Clone Semantic_Lossless at the pinned commit; fall back to a Kaggle copy. Returns (path, banner)."""
    s, dest = cfg["source"], Path(cfg["source"]["local_dir"])
    pin = s["pinned_commit"]
    try:
        if not (dest / ".git").is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            git(["init", "-q"], dest)
            git(["remote", "add", "origin", s["repo_url"]], dest)
        git(["fetch", "-q", "--depth", "1", "origin", pin], dest)
        git(["checkout", "-q", pin], dest)
        head = git(["rev-parse", "HEAD"], dest)
        if head != pin:
            raise RuntimeError("checked out %s, wanted %s" % (head, pin))
        return dest, "SOURCE: GitHub clone of Semantic_Lossless at pinned commit " + head
    except Exception as exc:     # private repo, no network, bad pin -> dataset copy
        hits = sorted(Path(s["fallback_search_root"]).glob("*/**/src/sqgan_train.py")) if Path(s["fallback_search_root"]).is_dir() else []
        if len(hits) != 1:
            raise RuntimeError("GitHub clone failed (%s) and %d Kaggle copies of src/sqgan_train.py found under %s; "
                               "attach a dataset holding the Semantic_Lossless source" % (exc, len(hits), s["fallback_search_root"]))
        root = hits[0].parents[1]
        banner = ("*" * 70 + "\n* SOURCE: GitHub clone FAILED (%s).\n* USING KAGGLE DATASET COPY %s - commit NOT verified against pin %s\n" % (exc, root, pin) + "*" * 70)
        return root, banner


def find_checkpoint_dir(search_root):
    """Folder (<=3 levels) that holds all ten author component *.ckpt files."""
    base = Path(search_root)
    for d in range(4):
        for p in sorted(base.glob("/".join(["*"] * d + ["encoder.ckpt"]))):
            if all((p.parent / (c + ".ckpt")).is_file() for c in COMPONENTS):
                return p.parent
    raise FileNotFoundError("ten author *.ckpt components not found under %s" % search_root)


def ensure_upstream_sqgan(sl_root, cfg):
    """Make <sl_root>/.upstream/sqgan exist (sqgan_train.py:34 reads it) from the Kaggle source dataset zip."""
    target = Path(sl_root) / ".upstream" / "sqgan"
    if (target / "config" / "sqgan_cityscapes.yml").is_file():
        return target
    zips = sorted(Path(cfg["source"]["fallback_search_root"]).glob("*/**/" + cfg["source"]["upstream_sqgan_zip"]))
    if len(zips) != 1:
        raise FileNotFoundError("need exactly one %s under /kaggle/input, found %d" % (cfg["source"]["upstream_sqgan_zip"], len(zips)))
    with zipfile.ZipFile(zips[0]) as z:
        for name in z.namelist():
            if not (Path(sl_root) / name).resolve().is_relative_to(Path(sl_root).resolve()):
                raise ValueError("zip path escapes target: " + name)
        z.extractall(sl_root)
    return target


def subprocess_trainer(options, run_dir, ctx):
    """Default trainer: pilot then train via sqgan_driver.py in a subprocess. Returns result dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    opt_path = run_dir / "trainer_options.json"
    opt_path.write_text(json.dumps(options, indent=2), encoding="utf-8")
    driver = Path(__file__).with_name("sqgan_driver.py")
    for stage in ("pilot", "train"):
        subprocess.run([sys.executable, "-u", str(driver), stage, "--sl-root", str(ctx["sl_root"]), "--options", str(opt_path),
                        "--run", str(run_dir), "--seg-classes", str(ctx["seg_classes"])], check=True, cwd=str(ctx["sl_root"]))
    resolved = run_dir / "teacher_resolved.json"
    return {"last_pt": run_dir / "checkpoints" / "last.pt",
            "teacher_revision": json.loads(resolved.read_text())["revision"] if resolved.is_file() else None}


# ----------------------------------------------------------------------------- bundle
def write_bundle(run_dir, bundle_dir, checkpoint_dir, options, cfg, result, seg_classes_path, manifest_summary=None):
    """Write the bundle format verified by sqgan_payload.py:37-45 (files list with bytes+sha256), then verify it."""
    run_dir, bundle = Path(run_dir), Path(bundle_dir)
    if bundle.exists():
        raise FileExistsError("bundle dir exists: %s" % bundle)
    last_pt = Path(result["last_pt"])
    if not last_pt.is_file():
        raise FileNotFoundError("trainer produced no last.pt at %s" % last_pt)
    bundle.mkdir(parents=True)
    shutil.copy2(last_pt, bundle / "last.pt")
    (bundle / "assets" / "checkpoint").mkdir(parents=True)
    for c in COMPONENTS:
        shutil.copy2(Path(checkpoint_dir) / (c + ".ckpt"), bundle / "assets" / "checkpoint" / (c + ".ckpt"))
    for name in ("model.json", "training_summary.json", "training.jsonl"):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, bundle / name)
    shutil.copy2(seg_classes_path, bundle / "seg_classes.yaml")
    portable = dict(options, manifest=None, checkpoint="assets/checkpoint", class_map="seg_classes.yaml")
    if result.get("teacher_revision"):
        portable["teacher_revision"] = result["teacher_revision"]
    (bundle / "portable_config.json").write_text(json.dumps(portable, indent=2), encoding="utf-8")
    if manifest_summary is not None:
        (bundle / "data_summary.json").write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    (bundle / "README.md").write_text("# SQ-GAN v1 (edge)\n\nFine-tuned with Codec.edge.sqgan_finetune on train-split images only. "
                                      "Teacher output is merged to the project 20-class map (seg_classes.yaml).\n", encoding="utf-8")
    files = [{"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(bundle.rglob("*")) if p.is_file()]
    (bundle / "bundle.json").write_text(json.dumps({"algorithm": "sqgan", "model_name": "SQ-GAN", "run_id": cfg["output"]["run_id"],
        "missing_constructor_assets": [], "files": files, "pipeline_completed": True,
        "scientific_validity_certified": False}, indent=2), encoding="utf-8")
    res = bundle_check.verify_bundle(bundle)
    if not res["ok"]:
        raise ValueError("written bundle failed verification: %s" % res["errors"])
    return bundle


# ----------------------------------------------------------------------------- orchestration
def run_pipeline(cfg, args, trainer=None, n_gpus=None):
    """prepare (splits, manifest) -> trainer -> bundle. ``trainer`` defaults to the subprocess trainer."""
    out = Path(cfg["output"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    images_root = Path(args.images_root) if args.images_root else discover_root("/kaggle/input", "index.jsonl")
    labels_root = Path(args.labels_root) if args.labels_root else discover_root("/kaggle/input", "labels_raw.jsonl")
    rows = read_labels(labels_root, cfg["inputs"]["labels_file"])
    splits, kind = resolve_splits(rows, labels_root / cfg["inputs"]["splits_file"] if not args.splits else args.splits,
                                  args.allow_provisional_splits)
    train = select_train_rows(rows, splits, cfg["data"]["sources"], cfg["data"]["max_train_images"])
    manifest = build_manifest(train, images_root, cfg["data"], out)
    assert_train_only(manifest, splits)
    manifest_path = out / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary = {"splits": kind, "counts": manifest["counts"], "identity": manifest["identity"],
               "skipped_too_small": len(manifest["skipped_too_small"])}
    print("manifest:", summary, flush=True)
    seg = ROOT / cfg["teacher"]["seg_classes"]
    ctx = {"seg_classes": seg}
    if trainer is None:
        ctx["sl_root"], banner = obtain_source(cfg)
        print(banner, flush=True)
        ensure_upstream_sqgan(ctx["sl_root"], cfg)
        trainer = subprocess_trainer
    ckpt = find_checkpoint_dir(cfg["inputs"]["checkpoint_search_root"]) if args.checkpoint_dir is None else Path(args.checkpoint_dir)
    if n_gpus is None:
        try:
            import torch
            n_gpus = torch.cuda.device_count()
        except ImportError:
            n_gpus = 0
    options = trainer_options(cfg, manifest_path, ckpt, n_gpus)
    result = trainer(options, out / "run", ctx)
    return write_bundle(out / "run", cfg["output"]["bundle_dir"], ckpt, options, cfg, result, seg, summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config/edge/sqgan_finetune.yaml"))
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=2")
    ap.add_argument("--images-root", help="folder holding index.jsonl + images/ (default: auto-discover)")
    ap.add_argument("--labels-root", help="folder holding labels_raw.jsonl (default: auto-discover)")
    ap.add_argument("--splits", help="explicit splits_v1.json path")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--allow-provisional-splits", action="store_true")
    ap.add_argument("--stage", choices=("source", "all"), default="all", help="source: only fetch + print the source banner")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.stage == "source":
        path, banner = obtain_source(cfg)
        print(banner, "\n", path)
        return 0
    print("bundle written:", run_pipeline(cfg, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
