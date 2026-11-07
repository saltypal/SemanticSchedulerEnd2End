"""Audit legacy vs no-COCO wire sizes on the tracked received-feature cache."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from Evaluation.compact_scene_transport import CompactSceneTransport
from Evaluation.no_coco_scene_transport import NoCocoSceneTransport


CACHE = "experiments/stream_scheduler/runs/mini-wsl-ingress-20261030-v2"


def cached_file(revision, name):
    return subprocess.check_output(["git", "show", f"{revision}:{CACHE}/{name}"])


def mask(values):
    return sum(int(v) << i for i, v in enumerate(values)).to_bytes(12, "little")


def reconstruct(row, legacy):
    counts = np.rint(np.expm1(row[:93].astype(np.float64)))
    areas = np.rint(row[93:186].astype(np.float64) * 255)
    if (not np.allclose(np.log1p(counts), row[:93], atol=1e-6, rtol=0)
            or not np.allclose(areas / 255, row[93:186], atol=1e-7, rtol=0)
            or not np.isin(row[186:374], (0, 1)).all()):
        raise ValueError("received features do not match the quantized scene contract")
    flags = int(row[372]) | (int(row[373]) << 1)
    return (bytes([2, flags]) + legacy.registry + legacy.producer
            + struct.pack("<93H", *map(int, counts)) + bytes(map(int, areas))
            + mask(row[186:279]) + mask(row[279:372]))


def projected(wire):
    keep = ((1 << 93) - 1) ^ ((1 << 80) - 1)
    return (wire[:66] + bytes(160) + wire[226:252] + bytes(80) + wire[332:345]
            + b"".join((int.from_bytes(wire[i:i + 12], "little") & keep).to_bytes(12, "little")
                       for i in (345, 357)))


def stats(sizes):
    return dict(min=int(min(sizes)), mean=float(np.mean(sizes)), median=float(np.median(sizes)),
                p95=float(np.quantile(sizes, 0.95)), max=int(max(sizes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = cached_file(args.revision, "received_features.npz")
    summary = json.loads(cached_file(args.revision, "summary.json"))
    # Cache export contains quantized measurements. The original registry hash
    # is immaterial to their size and is replaced by an audit-only binding.
    legacy = CompactSceneTransport(1, "00" * 32, summary["producer_ids"][0], [True] * 85 + [False] * 8)
    reduced = NoCocoSceneTransport(legacy)
    old, new = [], []
    with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
        for row in cache["features"]:
            wire = reconstruct(row, legacy)
            baseline, trial = legacy.encode(wire), reduced.encode(wire)
            if legacy.decode(baseline) != wire or reduced.decode(trial) != projected(wire):
                raise ValueError("scene wire round-trip or projection mismatch")
            old.append(len(baseline))
            new.append(len(trial))
    if len(old) != summary["images_verified"] or not np.isclose(
        np.mean(old), summary["proposed_lossless_scene_bytes"]["mean"]
    ):
        raise ValueError("cached legacy compact-size evidence disagrees")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(dict(scope="Scene block only, from tracked quantized feature cache. No label access.",
                       cache_revision=args.revision, cache_sha256=hashlib.sha256(raw).hexdigest(),
                       rows=len(old), projection_checks=len(old), legacy_compact=stats(old),
                       no_coco_compact=stats(new), mean_reduction_percent=100 * (1 - np.mean(new) / np.mean(old)),
                       legacy_dense_bytes=307, no_coco_dense_bytes=30), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
