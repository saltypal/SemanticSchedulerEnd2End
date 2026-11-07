"""Download small development-only windows from candidate CRASAR road views."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote

import fsspec
import numpy as np
from PIL import Image
import tifffile


def crop_rgb(url, x, y, size=512):
    with fsspec.open(url, mode="rb", block_size=65536) as stream:
        with tifffile.TiffFile(stream) as raster:
            page = raster.pages[0]
            height, width, channels = page.shape
            if not page.is_tiled or channels not in (3, 4) or page.dtype != np.uint8:
                raise ValueError("expected a tiled RGB or RGBA uint8 source")
            if min(height, width) < size:
                raise ValueError("window exceeds the source size")
            left = min(max(0, round(x) - size // 2), width - size)
            top = min(max(0, round(y) - size // 2), height - size)
            patch = np.zeros((size, size, channels), dtype=np.uint8)
            tile_width, tile_height = page.tilewidth, page.tilelength
            tile_columns = math.ceil(width / tile_width)
            for row in range(top // tile_height, (top + size - 1) // tile_height + 1):
                for col in range(left // tile_width, (left + size - 1) // tile_width + 1):
                    index = row * tile_columns + col
                    stream.seek(page.dataoffsets[index])
                    compressed = stream.read(page.databytecounts[index])
                    tile, _, _ = page.decode(compressed, index, jpegtables=page.jpegtables)
                    if tile is None:
                        continue
                    tx, ty = col * tile_width, row * tile_height
                    ax, ay = max(tx, left), max(ty, top)
                    bx, by = min(tx + tile_width, left + size), min(ty + tile_height, top + size)
                    patch[ay - top:by - top, ax - left:bx - left] = tile[0, ay - ty:by - ty, ax - tx:bx - tx]
            coverage = float(np.count_nonzero(patch[:, :, 3]) / (size * size)) if channels == 4 else None
            return Image.fromarray(patch[:, :, :3]), [left, top, size, size], coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.candidates.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    base = f"https://huggingface.co/datasets/{audit['dataset']}/resolve/{audit['revision']}"
    records = []
    for i, views in enumerate(audit["obstruction_candidate_views"]):
        for j, view in enumerate(views):
            source = Path(view["source"]).name.removesuffix(".json")
            path = args.output / f"group-{i + 1}-view-{j + 1}.png"
            image, window, coverage = crop_rgb(
                f"{base}/train/imagery/UAS/{quote(source)}", view["x"], view["y"])
            image.save(path)
            records.append(dict(candidate_group=i + 1, source=source, annotation_labels=view["labels"],
                                window=window, alpha_coverage=coverage, crop_name=path.name,
                                crop_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(
        scope="Training-source geographic candidate windows. No adjudicated same-road condition labels.",
        source_audit_sha256=hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        crops=records,
    )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
