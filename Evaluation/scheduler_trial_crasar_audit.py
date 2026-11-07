"""Audit public CRASAR training road annotations for candidate cross-view linkage."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests


REVISION = "4c2732a7ab9658517ca8caa69bfee9e1f87ca532"
DATASET = "CRASAR/CRASAR-U-DROIDs"
FOLDER = "train/annotations/UAS/road_damage_assessment"
BASE = "https://huggingface.co"


def get(path):
    response = requests.get(f"{BASE}/datasets/{DATASET}/resolve/{REVISION}/{path}", timeout=45)
    response.raise_for_status()
    return path, response.content


def geometry(line):
    points = line.get("EPSG:4326", [])
    if len(points) != 2:
        return None
    return tuple(sorted((round(p["lat"], 5), round(p["lon"], 5)) for p in points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    listing = requests.get(f"{BASE}/api/datasets/{DATASET}/tree/{REVISION}/{FOLDER}", timeout=30)
    listing.raise_for_status()
    paths = [item["path"] for item in listing.json() if item["type"] == "file"]
    ids = Counter()
    geometries = Counter()
    labels = Counter()
    sources = {}
    line_count = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        for path, raw in pool.map(get, paths):
            sources[path] = hashlib.sha256(raw).hexdigest()
            annotation = json.loads(raw)
            lines = annotation["road_lines"]
            line_count += len(lines)
            ids.update({line["id"] for line in lines if line.get("id")})
            geometries.update({geometry(line) for line in lines} - {None})
            labels.update(polygon["label"] for polygon in annotation["polygons"])
    result = dict(
        scope="Public training annotations only. Approximate geometry is candidate linkage, not ground truth.",
        dataset=DATASET, revision=REVISION, source_sha256=sources, files=len(paths),
        road_line_records=line_count, distinct_ids=len(ids),
        ids_in_multiple_files=sum(v > 1 for v in ids.values()),
        rounded_geometries_in_multiple_files=sum(v > 1 for v in geometries.values()),
        polygon_labels=dict(labels),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
