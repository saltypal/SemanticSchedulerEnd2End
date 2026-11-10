"""Everyday ("normal") image lists: Places365 scenes and Open Images selfies/people (plan v3.1 S1).

Metadata only: no images are downloaded here. Every item becomes a Low-tier row through
``parameters.edge.labels.normal_rows``. Places365 categories on the damage-lookalike blacklist are
dropped. Sampling is deterministic (hash order), stratified over Places365 categories.
"""
import csv
import hashlib
import io
import tarfile

from Evaluation.edge.edge_config import load_config


def _digest(seed, text):
    return hashlib.sha256(("%s:%s" % (seed, text)).encode()).hexdigest()


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
    """A typo in the blacklist would silently keep rubble-like scenes; fail loudly instead."""
    missing = [c for c in blacklist if c not in set(categories)]
    if missing:
        raise ValueError("blacklist categories not in Places365: %s" % missing)


def sample_places(items, blacklist, n, seed):
    """Stratified round-robin over categories in deterministic hash order."""
    by_cat = {}
    for item in items:
        if item["category"] not in blacklist:
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


def parse_openimages_people(labels_csv, images_csv, mids, n, seed):
    """Image ids confirmed (Confidence 1) for a MID. Selfie first, then Person, deterministic order."""
    wanted = {}
    for row in csv.DictReader(io.StringIO(labels_csv)):
        if float(row["Confidence"]) == 1.0:
            for kind, mid in mids.items():
                if row["LabelName"] == mid:
                    wanted.setdefault(kind, set()).add(row["ImageID"])
    info = {r["ImageID"]: r for r in csv.DictReader(io.StringIO(images_csv))}
    items, seen = [], set()
    for kind in mids:                                  # dict order: selfie, then person
        ids = sorted((i for i in wanted.get(kind, ()) if i in info and i not in seen),
                     key=lambda i: _digest(seed, i))
        for image_id in ids:
            if len(items) >= n:
                return items
            seen.add(image_id)
            r = info[image_id]
            items.append({"id": image_id, "path": "openimages/validation/%s.jpg" % image_id,
                          "category": kind, "url": r["OriginalURL"], "license": r["License"],
                          "author": r["Author"]})
    return items


def build_normals(fetch, cfg=None):
    """``fetch(url) -> bytes``. Returns (places_items, people_items, report)."""
    cfg = cfg or load_config("data")["normals"]
    src = cfg["sources"]
    categories = parse_places_categories(fetch(src["places365_categories"]).decode("utf-8"))
    check_blacklist(cfg["places365_blacklist"], categories)
    val = parse_places_val(places_val_from_tar(fetch(src["places365_filelist"])), categories)
    places = sample_places(val, set(cfg["places365_blacklist"]), cfg["places365_count"], cfg["seed"])
    people = parse_openimages_people(fetch(src["openimages_labels"]).decode("utf-8"),
                                     fetch(src["openimages_images"]).decode("utf-8"),
                                     cfg["people_mids"], cfg["people_count"], cfg["seed"])
    report = {"places365_pool": len(val), "places365_blacklisted_in_pool":
              sum(1 for it in val if it["category"] in set(cfg["places365_blacklist"])),
              "places365_selected": len(places), "places365_categories_used":
              len({it["category"] for it in places}), "people_selected": len(people),
              "people_by_kind": {k: sum(1 for it in people if it["category"] == k) for k in cfg["people_mids"]}}
    return places, people, report
