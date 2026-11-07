"""Audit public CRASAR training road annotations for candidate cross-view linkage."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from matplotlib.path import Path as PolygonPath
import numpy as np
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


def labels_at_midpoints(lines, polygons):
    """Return polygon labels at rounded line midpoints, without assigning negatives."""
    keys = {geometry(line) for line in lines} - {None}
    keys = sorted(keys)
    if not keys:
        return {}
    points = np.array([((a[1] + b[1]) / 2, (a[0] + b[0]) / 2) for a, b in keys])
    labels = {key: set() for key in keys}
    for polygon in polygons:
        vertices = polygon.get("EPSG:4326", [])
        if len(vertices) >= 3:
            boundary = PolygonPath([(v["lon"], v["lat"]) for v in vertices])
            for index in np.flatnonzero(boundary.contains_points(points)):
                labels[keys[index]].add(polygon["label"])
    return labels

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
    midpoint_labels = defaultdict(list)
    candidate_views = defaultdict(list)
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
            midpoints = {geometry(line): line["pixels"] for line in lines if len(line.get("pixels", [])) == 2}
            for key, values in labels_at_midpoints(lines, annotation["polygons"]).items():
                midpoint_labels[key].append(values)
                obstruction_labels = sorted(values & {"Partial Obstruction", "Total Obstruction"})
                if key in midpoints and obstruction_labels:
                    points = midpoints[key]
                    candidate_views[key].append(dict(source=path, labels=obstruction_labels,
                        x=sum(point["x"] for point in points) / 2,
                        y=sum(point["y"] for point in points) / 2))
    repeated = [views for views in midpoint_labels.values() if len(views) > 1]
    obstruction = {"Partial Obstruction", "Total Obstruction"}
    repeated_obstruction = sum(sum(bool(view & obstruction) for view in views) >= 2
                               for views in repeated)
    repeated_exact_obstruction = sum(
        any(sum(label in view for view in views) >= 2 for label in obstruction)
        for views in repeated
    )
    same_polygon_label = sum(
        any(sum(label in view for view in views) >= 2
            for label in set().union(*views) - {"Not Able To Determine"})
        for views in repeated
    )
    result = dict(
        scope="Public training annotations only. Approximate geometry is candidate linkage, not ground truth.",
        dataset=DATASET, revision=REVISION, source_sha256=sources, files=len(paths),
        road_line_records=line_count, distinct_ids=len(ids),
        ids_in_multiple_files=sum(v > 1 for v in ids.values()),
        rounded_geometries_in_multiple_files=sum(v > 1 for v in geometries.values()),
        polygon_labels=dict(labels),
        midpoint_candidates_with_same_known_label_in_multiple_views=same_polygon_label,
        midpoint_candidates_with_obstruction_in_multiple_views=repeated_obstruction,
        midpoint_candidates_with_same_obstruction_label_in_multiple_views=repeated_exact_obstruction,
        obstruction_candidate_views=[views for views in candidate_views.values() if len(views) > 1],
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
