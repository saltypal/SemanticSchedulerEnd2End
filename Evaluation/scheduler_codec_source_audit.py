"""Verify development-image provenance against the publisher's validation CSV."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps
import requests


def verify(row, labels, parent):
    reference = labels[row["row_idx"]]
    if {key for key, value in reference.items() if value == "1"} != set(row["positive_labels"]):
        raise ValueError("publisher label vector does not match cached image")
    url = urlsplit(reference["url"])
    if url.scheme != "https" or url.netloc != "fema-cap-imagery.s3.amazonaws.com" or not url.path.startswith("/Images/"):
        raise ValueError("unexpected publisher image host")
    path = parent / row["image_path"]
    cached = path.read_bytes()

    def thumbnail(raw):
        with Image.open(io.BytesIO(raw)) as picture:
            image = ImageOps.exif_transpose(picture).convert("RGB")
            return np.asarray(image.resize((64, 64), Image.Resampling.LANCZOS), dtype=float)

    with requests.get(url.geturl(), timeout=30, stream=True) as response:
        response.raise_for_status()
        chunks = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            chunks.extend(chunk)
            if len(chunks) > 32 * 1024 * 1024:
                raise ValueError("publisher image exceeds audit size cap")
    error = float(np.abs(thumbnail(cached) - thumbnail(chunks)).mean())
    prefix = url.path.split("/")[2]
    return dict(identity=row["item_id"], source_sha256=hashlib.sha256(cached).hexdigest(),
                publisher_sha256=hashlib.sha256(chunks).hexdigest(), source_prefix=prefix,
                candidate_event=prefix if prefix.startswith("CAP_-_") else None,
                thumbnail_mae=error, image_match=error <= 5.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if any(row.get("source_split") != "validation" for row in rows):
        raise ValueError("validation rows only")
    raw_csv = args.validation_csv.read_bytes()
    labels = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8-sig"))))
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(lambda row: verify(row, labels, args.manifest.parent), rows))
    output = dict(scope="Publisher validation provenance audit. Numeric prefixes have unresolved disaster identity.",
                  publisher_csv_sha256=hashlib.sha256(raw_csv).hexdigest(), rows=verified)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
