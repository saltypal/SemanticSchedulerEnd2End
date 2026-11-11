"""Everyday ("normal") image lists: Places365 scenes and COCO people (plan v3.1 S1).

Primary mode reads the Kaggle datasets mounted read-only under /kaggle/input:
  * Places365 (``benjaminkz/places365``): ``<split>/<class>/<name>.jpg`` (split from ``places365_split``, default val, fallback train; ``train.txt`` only for train); a small
    stratified sample of ``places365_per_class`` images from EVERY class, minus the blacklist.
  * COCO 2017 (``awsaf49/coco-2017-dataset``): ``train2017`` images + ``person_keypoints_train2017.json``;
    images with a sufficiently large person, in deterministic hash order.
The old Places365 ``val_256.tar`` path is only a fallback when the Kaggle dataset is not mounted.
Every item becomes a Low-tier row through ``parameters.edge.labels.normal_rows``. Sampling is
deterministic (sha256 of seed + relative path). Nothing here downloads or copies images.
"""
import hashlib
import io
import json
import logging
import os
import tarfile
from pathlib import Path

from Evaluation.edge.edge_config import load_config

log = logging.getLogger(__name__)
IMG_EXT = (".jpg", ".jpeg", ".png")
PLACES_KEYPOINTS = "person_keypoints_train2017.json"
VAL_KEYPOINTS = "person_keypoints_val2017.json"


def _digest(seed, text):
    return hashlib.sha256(("%s:%s" % (seed, text)).encode()).hexdigest()


def _norm(name):
    return str(name).strip("/").replace("/", "_")


# ---------------------------------------------------------------- Places365 (tar fallback) helpers
def parse_places_categories(text):
    """'/a/airfield 0' -> 'airfield'; nested '/a/arena/hockey 3' -> 'arena/hockey'."""
    names = []
    for line in text.splitlines():
        if line.strip():
            path = line.split()[0]
            names.append(path.split("/", 2)[2])
    return names


def places_val_from_tar(tar_bytes):
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("places365_val.txt"))
        return tar.extractfile(member).read().decode("utf-8")


def parse_places_val(text, categories):
    items = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, index = line.split()
        items.append({"id": name, "path": "val_256/" + name, "category": categories[int(index)]})
    return items


def check_blacklist(blacklist, categories):
    """Tar fallback only (full category list known): a typo would silently keep rubble-like scenes."""
    missing = [c for c in blacklist if c not in set(categories)]
    if missing:
        raise ValueError("blacklist categories not in Places365: %s" % missing)


def sample_places(items, blacklist, n, seed):
    """Stratified round-robin over categories in deterministic hash order."""
    banned = {_norm(b) for b in blacklist}
    by_cat = {}
    for item in items:
        if _norm(item["category"]) not in banned:
            by_cat.setdefault(item["category"], []).append(item)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda it: _digest(seed, it["id"]))
    chosen, round_idx = [], 0
    cats = sorted(by_cat)
    while len(chosen) < n and any(len(v) > round_idx for v in by_cat.values()):
        for cat in cats:
            if len(by_cat[cat]) > round_idx and len(chosen) < n:
                chosen.append(by_cat[cat][round_idx])
        round_idx += 1
    return chosen


# ---------------------------------------------------------------- Places365 (Kaggle mount)
def _subdirs(path):
    try:
        with os.scandir(path) as it:
            return sorted(e.name for e in it if e.is_dir())
    except OSError:
        return []


def discover_places_root(base="/kaggle/input", max_depth=4):
    """Directory (below ``base``, depth <= max_depth) holding ``val/`` or ``train/`` with class subfolders."""
    base = Path(base)

    def walk(d, depth):
        subs = _subdirs(d)
        if any(sp in subs and _subdirs(d / sp) for sp in ("val", "train")):
            return d
        if depth >= max_depth:
            return None
        for s in subs:
            found = walk(d / s, depth + 1)
            if found:
                return found
        return None
    return walk(base, 0) if base.is_dir() else None


def _is_img(name):
    return name.lower().endswith(IMG_EXT)


def resolve_split(root, split="val"):
    """The split folder actually used: ``split`` when it has class subfolders, else ``train`` (fallback)."""
    root = Path(root)
    if _subdirs(root / split):
        return split
    return "train"


def list_class_files(root, split="val"):
    """{class: [file rel. to the class folder]} for ``<root>/<split>/`` (fallback ``train``).

    train.txt is used only when the split used is ``train``; otherwise a plain per-class scandir.
    Class = directory path below the split folder (nested classes use '/')."""
    root = Path(root)
    split = resolve_split(root, split)
    by_cls = {}
    listing = root / "train.txt"
    if split == "train" and listing.is_file():
        with open(listing, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if not parts or not _is_img(parts[0]):
                    continue
                rel = parts[0].lstrip("/")
                if rel.startswith("train/"):
                    rel = rel[len("train/"):]
                cls, _, name = rel.rpartition("/")
                if cls:
                    by_cls.setdefault(cls, []).append(name)
        if by_cls:
            return by_cls
    train = root / split

    def scan(d, rel):
        with os.scandir(d) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if e.is_dir():
                scan(e.path, rel + "/" + e.name if rel else e.name)
            elif rel and _is_img(e.name):
                by_cls.setdefault(rel, []).append(e.name)
    scan(train, "")
    return by_cls


def sample_places_kaggle(root, per_class, blacklist, seed, class_files=None, split="val"):
    """Up to ``per_class`` files from EVERY non-blacklisted class, in deterministic hash order.

    Returns (items, info). Items are ordered round-robin by within-class rank (stays stratified when
    trimmed); each has id (class/name), category, name, src (absolute path), rel (below the root)."""
    split_used = resolve_split(root, split)
    class_files = class_files if class_files is not None else list_class_files(root, split)
    banned = {_norm(b) for b in blacklist}
    seen_black, kept = set(), {}
    for cls, names in class_files.items():
        keys = {_norm(cls), _norm(cls.rsplit("/", 1)[-1])}
        hit = keys & banned
        if hit:
            seen_black |= hit
            continue
        kept[cls] = names
    unmatched = sorted(b for b in blacklist if _norm(b) not in seen_black)
    for b in unmatched:
        log.warning("places365 blacklist entry %r matches no class folder; ignored", b)
    picked = {}
    for cls, names in kept.items():
        ordered = sorted(names, key=lambda n: _digest(seed, "%s/%s/%s" % (split_used, cls, n)))
        picked[cls] = ordered[:per_class]
    items = []
    for rank_in_class in range(per_class):
        for cls in sorted(picked):
            if rank_in_class < len(picked[cls]):
                n = picked[cls][rank_in_class]
                items.append({"id": "%s/%s" % (cls, n), "category": cls, "name": n,
                              "rel": "%s/%s/%s" % (split_used, cls, n), "src": str(Path(root) / split_used / cls / n)})
    info = {"places365_classes_total": len(class_files), "places365_classes_used": len(picked),
            "places365_blacklisted_classes": len(class_files) - len(kept),
            "places365_blacklist_unmatched": unmatched, "places365_split_used": split_used,
            "places365_used_train_txt": split_used == "train" and (Path(root) / "train.txt").is_file()}
    return items, info


# ---------------------------------------------------------------- COCO people
def find_coco(base="/kaggle/input", max_depth=5):
    """Search for directories named train2017/val2017 and the person_keypoints JSON files.

    Returns dict with train_dir, val_dir, keypoints_train, keypoints_val (each Path or None), or None
    when nothing COCO-like is mounted."""
    base = Path(base)
    found = {"train_dir": None, "val_dir": None, "keypoints_train": None, "keypoints_val": None}
    names = {"train2017": "train_dir", "val2017": "val_dir",
             PLACES_KEYPOINTS: "keypoints_train", VAL_KEYPOINTS: "keypoints_val"}

    def walk(d, depth):
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            key = names.get(e.name)
            if e.is_dir():
                if key in ("train_dir", "val_dir"):
                    found[key] = found[key] or Path(e.path)
                elif depth < max_depth:
                    walk(e.path, depth + 1)
            elif key and found[key] is None:
                found[key] = Path(e.path)
    if base.is_dir():
        walk(base, 1)
    return found if any(found.values()) else None


def people_from_keypoints(json_path, images_dir, n, min_frac, seed):
    """Images with a person box >= ``min_frac`` of the image area, first ``n`` in hash order."""
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    size = {im["id"]: (im["file_name"], im["width"] * im["height"]) for im in data["images"]}
    good = set()
    for a in data["annotations"]:
        if a.get("iscrowd") or a.get("category_id", 1) != 1 or a["image_id"] not in size:
            continue
        w, h = a["bbox"][2], a["bbox"][3]
        area = size[a["image_id"]][1]
        if area > 0 and w * h / area >= min_frac:
            good.add(a["image_id"])
    names = sorted((size[i][0] for i in good), key=lambda f: _digest(seed, f))
    del data, size
    return _first_existing(names, images_dir, n)


def _first_existing(names, images_dir, n):
    items = []
    for name in names:
        if len(items) >= n:
            break
        if (Path(images_dir) / name).is_file():
            items.append({"id": name, "category": "person", "name": name, "src": str(Path(images_dir) / name)})
    return items


def select_people(coco, n, min_frac, seed):
    """-> (items, report_part). Never raises for a missing mount: returns ([], {... 'unavailable'})."""
    if not coco:
        return [], {"people_source": "unavailable", "unavailable": ["coco_people: COCO 2017 not mounted"]}
    train, val = coco.get("train_dir"), coco.get("val_dir")
    if train and coco.get("keypoints_train"):
        return (people_from_keypoints(coco["keypoints_train"], train, n, min_frac, seed),
                {"people_source": "train2017+person_keypoints_train2017"})
    log.warning("COCO person_keypoints_train2017.json (or train2017) not found; falling back to val2017")
    if val and coco.get("keypoints_val"):
        return (people_from_keypoints(coco["keypoints_val"], val, n, min_frac, seed),
                {"people_source": "val2017+person_keypoints_val2017_fallback"})
    if val:
        names = sorted((p.name for p in Path(val).iterdir() if _is_img(p.name)), key=lambda f: _digest(seed, f))
        return (_first_existing(names, val, n),
                {"people_source": "val2017_fallback_no_person_filter",
                 "note": "no annotation file found: images are not verified to contain a person"})
    log.warning("no usable COCO image folder; people skipped")
    return [], {"people_source": "unavailable", "unavailable": ["coco_people: no train2017/val2017 folder found"]}


# ---------------------------------------------------------------- combined
def build_normals(fetch, cfg=None, input_root="/kaggle/input", max_images=None):
    """Returns (places_items, people_items, report). ``fetch(url) -> bytes`` is used only by the Places365
    tar fallback (when the Kaggle dataset is not mounted)."""
    cfg = cfg or load_config("data")["normals"]
    per_class, seed = int(cfg["places365_per_class"]), cfg["seed"]
    report = {}
    root = discover_places_root(input_root)
    if root:
        places, info = sample_places_kaggle(root, per_class, cfg["places365_blacklist"], seed,
                                           split=cfg.get("places365_split", "val"))
        report.update(info, places365_mode="kaggle:%s" % root)
    else:
        src = cfg["sources"]
        categories = parse_places_categories(fetch(src["places365_categories"]).decode("utf-8"))
        check_blacklist(cfg["places365_blacklist"], categories)
        val = parse_places_val(places_val_from_tar(fetch(src["places365_filelist"])), categories)
        places = sample_places(val, cfg["places365_blacklist"], per_class * len([c for c in categories if _norm(c) not in {_norm(b) for b in cfg["places365_blacklist"]}]), seed)
        report.update(places365_mode="fallback_val_256_tar", places365_pool=len(val))
        log.warning("Places365 Kaggle dataset not mounted under %s; using val_256 tar fallback", input_root)
    if max_images:
        places = places[:max_images]
    report.update(places365_selected=len(places), places365_categories_used=len({p["category"] for p in places}))
    people, prep = select_people(find_coco(input_root), int(cfg["people_count"]),
                                 float(cfg["coco_min_person_area_frac"]), seed)
    if max_images:
        people = people[:max_images]
    report.update(prep, people_selected=len(people))
    return places, people, report
