"""Build an event-separated CRASAR development cohort."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from Evaluation.scheduler_trial_crasar_audit import BASE, DATASET, REVISION, get
from Evaluation.scheduler_trial_crasar_crops import crop_rgb


EVENTS = {
    "Hurricane Ian": ("train", ("1001-", "1002-"), 16, 16),
    "Champlain Towers Collapse": ("calibration", ("20210703-",), 6, 12),
    "Hurricane Harvey": ("screen", ("0827-", "090302-", "090401-", "090402-", "090403-"), 6, 12),
    "Hurricane Ida": ("screen", ("20210901-", "20210902-"), 6, 12),
}
OBSTRUCTION = {"Partial Obstruction", "Total Obstruction"}
FRESH_EVENTS = {
    "Mayfield Tornado": ("screen", ("20211213-", "20211214-", "20211215-"), 6, 12),
    "Hurricane Michael": ("screen", ("10132018-", "10142018-"), 6, 12),
}


def points(shape):
    return [(point["x"], point["y"]) for point in shape["pixels"]]


def candidates(annotation, positive, prioritize_context=False):
    lines = [LineString(points(line)) for line in annotation["road_lines"]
             if line["label"] == "Road Line" and len(line["pixels"]) >= 2]
    polygons = [(p["label"], Polygon(points(p)).buffer(0)) for p in annotation["polygons"]
                if len(p["pixels"]) >= 3]
    adverse = unary_union([p for _, p in polygons])
    other = unary_union([p for label, p in polygons if label not in OBSTRUCTION])
    roads = unary_union(lines)
    if positive:
        for label, polygon in polygons:
            if label not in OBSTRUCTION:
                continue
            segment = polygon.intersection(roads)
            if segment.length < 8:
                continue
            point = segment.representative_point()
            if other.intersects(box(point.x-128, point.y-128, point.x+128, point.y+128)):
                continue
            yield point.x, point.y, label
    else:
        for line in lines:
            if line.length < 96:
                continue
            point = line.interpolate(.5, normalized=True)
            if not adverse.intersects(box(point.x-128, point.y-128, point.x+128, point.y+128)):
                nearby = prioritize_context and adverse.intersects(
                    box(point.x-256, point.y-256, point.x+256, point.y+256))
                label = ("Clear target with nearby adverse annotation" if nearby
                         else "Road Line without adverse annotation in target region")
                yield point.x, point.y, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--multievent-from", type=Path)
    parser.add_argument("--review-extra-negatives", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.review_extra_negatives <= 12 or (args.review_extra_negatives and not args.multievent_from):
        parser.error("extra review negatives require the multi-event profile and a quota from 0 to 12")
    hashes = json.loads(args.prior_audit.read_bytes())["source_sha256"]
    events = EVENTS
    if args.multievent_from:
        events = FRESH_EVENTS
        prefixes = tuple(p for _, group, _, _ in events.values() for p in group)
        listing = requests.get(f"{BASE}/api/datasets/{DATASET}/tree/{REVISION}/"
                               "test/annotations/UAS/road_damage_assessment", timeout=30)
        listing.raise_for_status()
        hashes = {item["path"]: None for item in listing.json()
                  if item["type"] == "file" and Path(item["path"]).name.startswith(prefixes)}
    args.output.mkdir(exist_ok=True, parents=True)
    annotations = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for path, raw in pool.map(get, sorted(hashes)):
            digest = hashlib.sha256(raw).hexdigest()
            if hashes[path] is not None and digest != hashes[path]:
                raise ValueError(f"annotation changed: {path}")
            hashes[path] = digest
            annotations[path] = json.loads(raw)
    rng = random.Random(2084)
    root = f"{BASE}/datasets/{DATASET}/resolve/{REVISION}"
    rows, thumbnails, attempted = [], [], Counter()
    journal = args.output / "acquired.jsonl"
    if args.multievent_from and not journal.exists():
        for old in json.loads(args.multievent_from.read_bytes())["rows"]:
            path = args.multievent_from.parent / old["image_path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != old["source_sha256"]:
                raise ValueError(f"previous crop changed: {path}")
            row = dict(old, role="train" if old["role"] == "screen" else old["role"])
            shutil.copyfile(path, args.output / row["image_path"])
            rows.append(row)
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if journal.exists():
        if (not args.resume and not rows) or (args.output / "cohort.json").exists():
            raise FileExistsError(journal)
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            image = args.output / row["image_path"]
            if hashlib.sha256(image.read_bytes()).hexdigest() != row["source_sha256"]:
                raise ValueError(f"journal image changed: {image}")
            with Image.open(image) as opened:
                thumbnails.append(np.asarray(opened.resize((32, 32)), dtype=np.int16))
    def fetch(path, x, y):
        source = Path(path).name.removesuffix(".json")
        try:
            return crop_rgb(f"{root}/{path.split('/')[0]}/imagery/UAS/{quote(source)}",
                            x, y, require_alpha=True)
        except (OSError, ValueError) as exc:
            print(f"Crop rejected: {source}: {exc}", flush=True)
            return None

    for event, (role, prefixes, positives, negatives) in events.items():
        paths = [p for p in annotations if Path(p).name.startswith(prefixes)]
        for positive, quota in ((True, positives), (False, negatives + args.review_extra_negatives)):
            pools = {path: list(candidates(annotations[path], positive, bool(args.multievent_from)))
                     for path in paths}
            for choices in pools.values():
                rng.shuffle(choices)
                if args.multievent_from and not positive:
                    choices.sort(key=lambda row: row[2].startswith("Clear target"))
            count = sum(r["event"] == event and r["obstruction"] == positive for r in rows)
            while count < quota and any(pools.values()):
                batch = []
                for path in paths:
                    while pools[path]:
                        x, y, label = pools[path].pop()
                        if not any(r["annotation"] == path and max(abs(x-r["x"]), abs(y-r["y"])) < 512
                                   for r in rows):
                            batch.append((path, x, y, label))
                            break
                with ThreadPoolExecutor(max_workers=5) as pool:
                    futures = [pool.submit(fetch, path, x, y) for path, x, y, _ in batch]
                    results = [future.result() for future in futures]
                attempted[event] += len(results)
                for (path, x, y, label), result in zip(batch, results):
                    if count == quota:
                        break
                    if result is None:
                        continue
                    image, window, coverage = result
                    left, top = window[:2]
                    if coverage != 1.0 or abs(x-left-256) > 1 or abs(y-top-256) > 1:
                        continue
                    thumb = np.asarray(image.resize((32, 32)), dtype=np.int16)
                    if any(np.abs(thumb - other).mean() <= 5 for other in thumbnails):
                        continue
                    thumbnails.append(thumb)
                    name = f"crop-{len(rows):03d}.png"
                    target = args.output / name
                    image.save(target)
                    rows.append(dict(event=event, role=role, obstruction=positive, annotation_label=label,
                                     annotation=path, annotation_sha256=hashes[path], image_path=name,
                                     source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                     window=window, x=x, y=y))
                    with journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(rows[-1]) + "\n")
                    count += 1
                    print(f"{event} {positive} {count}/{quota}", flush=True)
            if count != quota:
                raise ValueError(f"cohort quota not met: {event} {positive} {count}/{quota}")
    receipt = dict(dataset=DATASET, revision=REVISION, endpoint="Visible obstruction in centered road region",
                   negatives="Road Line without adverse annotation in the central 256 px square",
                   source_split=("publisher train plus selected reader-evaluation test events"
                                 if args.multievent_from else "publisher train only"),
                   profile="multievent" if args.multievent_from else "original",
                   attempted_this_invocation=dict(attempted), rows=rows)
    (args.output / "cohort.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
