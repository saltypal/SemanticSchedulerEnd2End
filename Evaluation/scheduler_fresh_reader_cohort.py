"""Prepare a fresh LADI development cohort from the publisher's validation CSV."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps
import requests


def fingerprint(data):
    with Image.open(io.BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.int16)


def download(row):
    url = urlsplit(row["url"])
    if (url.scheme, url.netloc) != ("https", "fema-cap-imagery.s3.amazonaws.com") or not url.path.startswith("/Images/"):
        raise ValueError("unexpected dataset source")
    data = bytearray()
    with requests.get(url.geturl(), stream=True, timeout=35) as response:
        response.raise_for_status()
        for chunk in response.iter_content(1024 * 1024):
            data.extend(chunk)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("source image exceeds size cap")
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument("prior_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-prefixes-from", type=Path)
    parser.add_argument("--positive-count", type=int, choices=(9, 12), default=12)
    args = parser.parse_args()
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8-sig"))
    used = {row["row_idx"] for row in prior}
    prefixes = set()
    prior_paths = [args.prior_manifest.parent / row["image_path"] for row in prior]
    if args.exclude_prefixes_from:
        calibration = json.loads(args.exclude_prefixes_from.read_text(encoding="utf-8"))
        prefixes = {row["source_prefix"] for row in calibration}
        used.update(row["row_idx"] for row in calibration)
        prior_paths.extend(args.exclude_prefixes_from.parent / row["image_path"] for row in calibration)
    thumbs, hashes = [], set()
    for path in prior_paths:
        data = path.read_bytes()
        thumbs.append(fingerprint(data))
        hashes.add(hashlib.sha256(data).hexdigest())
    raw_csv = args.validation_csv.read_bytes()
    labels = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8-sig"))))
    by_label = {True: defaultdict(list), False: defaultdict(list)}
    for index, row in enumerate(labels):
        if index not in used and row["roads_any"] == "1":
            prefix = row["local_path"].split("/")[2]
            if prefix not in prefixes | {"CAP_-_Testing", "STG_Test_10062022"}:
                by_label[row["roads_damage"] == "1"][prefix].append((index, row))
    args.output.mkdir(exist_ok=False, parents=True)
    selected, rejected = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for positive, groups in by_label.items():
            quota = args.positive_count if positive else 12
            keys = sorted(groups, key=lambda key: (not key.startswith("CAP_-_"), key))
            candidates = [group[i] for i in range(max(map(len, groups.values())))
                          for key in keys if i < len(group := groups[key])]
            count = 0
            for batch in range(0, len(candidates), 4):
                block = candidates[batch:batch+4]
                futures = [pool.submit(download, row) for _, row in block]
                for (index, row), future in zip(block, futures):
                    try:
                        data = future.result()
                        thumb = fingerprint(data)
                    except (requests.RequestException, OSError, ValueError) as exc:
                        rejected.append(dict(row=index, reason=type(exc).__name__))
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in hashes or any(np.abs(thumb - old).mean() <= 5 for old in thumbs):
                        rejected.append(dict(row=index, reason="duplicate thumbnail screen"))
                        continue
                    hashes.add(digest)
                    thumbs.append(thumb)
                    name = f"ladi-val-{index:04d}.jpg"
                    (args.output / name).write_bytes(data)
                    prefix = row["local_path"].split("/")[2]
                    selected.append(dict(item_id=f"ladi-val-{index:04d}", row_idx=index, image_path=name,
                                         source_split="validation", source_sha256=digest, source_prefix=prefix,
                                         event_group=prefix if prefix.startswith("CAP_-_") else None,
                                         positive_labels=[k for k, v in row.items() if v == "1"]))
                    count += 1
                    if count == quota:
                        break
                if count == quota:
                    break
            if count != quota:
                raise ValueError("fresh balanced cohort quota was not met")
    (args.output / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    audit = dict(publisher_csv_sha256=hashlib.sha256(raw_csv).hexdigest(),
                 excluded_prior_rows=len(used), excluded_prefixes=sorted(prefixes),
                 selected=len(selected), rejected=rejected)
    (args.output / "selection.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
