"""Create a lane-masked ablation from already-scored obstruction development crops."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point

from Evaluation.scheduler_trial_crasar_audit import get


def lane_polygon(annotation, x, y):
    target = Point(x, y)
    candidates = []
    for row in annotation["road_lines"]:
        if len(row["pixels"]) != 2 or len(row.get("EPSG:4326", [])) != 2:
            continue
        a, b = row["pixels"]
        line = LineString([(a["x"], a["y"]), (b["x"], b["y"])])
        p, q = row["EPSG:4326"]
        meters = 111_195 * math.hypot(q["lat"]-p["lat"],
                                    (q["lon"]-p["lon"])*math.cos(math.radians((p["lat"]+q["lat"])/2)))
        if meters > 0 and line.length:
            candidates.append((line.distance(target), line, 3.6 * line.length / meters))
    distance, line, half_width = min(candidates, key=lambda item: item[0])
    if distance > 2 or not math.isfinite(half_width):
        raise ValueError("crop center is not on a usable annotated road line")
    return line.buffer(half_width), half_width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_bytes())
    rows = cohort["rows"]
    hashes = {r["annotation"]: r["annotation_sha256"] for r in rows}
    with ThreadPoolExecutor(max_workers=6) as pool:
        annotations = {}
        for name, data in pool.map(get, sorted(hashes)):
            if hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError("annotation hash changed")
            annotations[name] = json.loads(data)
    args.output.mkdir(exist_ok=False, parents=True)
    audit = []
    for row in rows:
        source = args.cohort.parent / row["image_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError("source crop changed")
        left, top, _, _ = row["window"]
        lane, width = lane_polygon(annotations[row["annotation"]], row["x"], row["y"])
        mask = Image.new("L", (512, 512))
        ImageDraw.Draw(mask).polygon([(x-left,y-top) for x,y in lane.exterior.coords], fill=255)
        with Image.open(source) as raw:
            result = Image.composite(raw.convert("RGB"), Image.new("RGB", (512,512), (127,)*3), mask)
        result.save(args.output / source.name)
        digest = hashlib.sha256((args.output / source.name).read_bytes()).hexdigest()
        audit.append(dict(image=source.name, original_sha256=row["source_sha256"], lane_sha256=digest,
                          half_width_px=width, lane_pixels=mask.crop((128,128,384,384)).histogram()[255],
                          event=row["event"], role=row["role"], obstruction=row["obstruction"]))
        row["source_sha256"] = digest
    cohort["rows"] = rows
    # The review describes original image bytes. Keep its selection semantics and bind
    # it to the transformed inputs by hash, while preserving the original hash in the audit.
    review = json.loads((args.cohort.parent / "visual_review.json").read_bytes())
    mapped = {r["original_sha256"]:r["lane_sha256"] for r in audit}
    for entry in review["rows"]:
        if entry["selected"]:
            entry["sha256"] = mapped[entry["sha256"]]
    review_path = args.output / "visual_review.json"
    review_path.write_bytes(json.dumps(review, indent=2).encode())
    cohort["visual_review_sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    (args.output / "cohort.json").write_bytes(json.dumps(cohort, indent=2).encode())
    (args.output / "lane_audit.json").write_bytes(json.dumps(audit, indent=2).encode())


if __name__ == "__main__":
    main()
