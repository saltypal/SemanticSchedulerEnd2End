"""Build a small, event-separated CRASAR source screen using training data only."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import quote

import numpy as np
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


def points(shape):
    return [(point["x"], point["y"]) for point in shape["pixels"]]


def candidates(annotation, positive):
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
                yield point.x, point.y, "Road Line without adverse annotation in target region"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    hashes = json.loads(args.prior_audit.read_bytes())["source_sha256"]
    args.output.mkdir(exist_ok=True, parents=True)
    annotations = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for path, raw in pool.map(get, sorted(hashes)):
            if hashlib.sha256(raw).hexdigest() != hashes[path]:
                raise ValueError(f"annotation changed: {path}")
            annotations[path] = json.loads(raw)
    rng = random.Random(2084)
    root = f"{BASE}/datasets/{DATASET}/resolve/{REVISION}/train/imagery/UAS"
    rows, thumbnails, attempted = [], [], Counter()
    journal = args.output / "acquired.jsonl"
    if journal.exists():
        if not args.resume or (args.output / "cohort.json").exists():
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
            return crop_rgb(f"{root}/{quote(source)}", x, y, require_alpha=True)
        except (OSError, ValueError) as exc:
            print(f"Crop rejected: {source}: {exc}", flush=True)
            return None

    for event, (role, prefixes, positives, negatives) in EVENTS.items():
        paths = [p for p in annotations if Path(p).name.startswith(prefixes)]
        for positive, quota in ((True, positives), (False, negatives)):
            pools = {path: list(candidates(annotations[path], positive)) for path in paths}
            for choices in pools.values():
                rng.shuffle(choices)
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
                   source_split="publisher train only", attempted_this_invocation=dict(attempted), rows=rows)
    (args.output / "cohort.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
