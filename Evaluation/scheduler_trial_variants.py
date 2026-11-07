"""Construct mildly perturbed development images with evaluator-only source linkage."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    variants = []
    for row in rows:
        source = args.source_manifest.parent / row["image_path"]
        identifier = row["item_id"] + "-variant"
        path = args.output / f"{identifier}.png"
        with Image.open(source) as image:
            # No crop or geometric remapping. Keep the same visible scene.
            changed = ImageEnhance.Brightness(image.convert("RGB")).enhance(0.9)
            changed.filter(ImageFilter.GaussianBlur(radius=1.2)).save(path)
        variants.append(dict(row, item_id=identifier, image_path=path.name, source_id=row["item_id"],
                             transformation="RGB brightness 0.9 followed by Gaussian blur radius 1.2"))
    (args.output / "sample_manifest.json").write_text(json.dumps(variants, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
