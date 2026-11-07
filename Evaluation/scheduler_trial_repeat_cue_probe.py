"""Screen a compact CLIP repeat cue on three CRASAR training-only view pairs."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import clip
import numpy as np
from PIL import Image
import torch

from Evaluation.scheduler_trial_probe import appearance_hash


def distance(a, b):
    return (a ^ b).bit_count()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = args.crop_dir / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))["crops"]
    torch.set_num_threads(2)
    model, preprocess = clip.load(str(args.checkpoint), device="cpu")
    model.eval()

    def embedding(image):
        with torch.inference_mode():
            x = model.encode_image(preprocess(image).unsqueeze(0)).float()[0]
            return torch.nn.functional.normalize(x, dim=0).numpy()

    dimensions = len(embedding(Image.new("RGB", (512, 512))))
    seed = 1729
    projections = np.random.default_rng(seed).choice([-1.0, 1.0], size=(128, dimensions)).astype(np.float32)
    hashes = []
    for row in rows:
        with Image.open(args.crop_dir / row["crop_name"]) as opened:
            image = opened.convert("RGB")
        start = perf_counter_ns()
        dhash = appearance_hash(image)
        dhash_ms = (perf_counter_ns() - start) / 1_000_000
        start = perf_counter_ns()
        vector = embedding(image)
        sketch = int.from_bytes(np.packbits(projections @ vector > 0).tobytes(), "big")
        clip_ms = (perf_counter_ns() - start) / 1_000_000
        hashes.append(dict(group=row["candidate_group"], location=row["source"].split(".")[0],
                           crop=row["crop_name"], dhash=dhash, sketch=sketch,
                           dhash_ms=dhash_ms, clip_ms=clip_ms))

    matched = []
    cross_location = []
    for a, b in combinations(hashes, 2):
        scores = dict(crops=[a["crop"], b["crop"]], dhash_distance=distance(a["dhash"], b["dhash"]),
                      clip_sketch_distance=distance(a["sketch"], b["sketch"]))
        if a["group"] == b["group"]:
            matched.append(scores)
        elif a["location"] != b["location"]:
            cross_location.append(scores)
    with args.checkpoint.open("rb") as stream:
        checkpoint = hashlib.file_digest(stream, "sha256").hexdigest()
    output = dict(scope="Six selected training-source crops from one event. No holdout or fitted threshold.",
                  checkpoint_sha256=checkpoint, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  projection_seed=seed, sketch_bytes=16, image_hash_bytes=8,
                  median_dhash_ms=median(row["dhash_ms"] for row in hashes),
                  median_clip_sketch_ms=median(row["clip_ms"] for row in hashes),
                  matched_candidates=matched, cross_location_pairs=cross_location)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
