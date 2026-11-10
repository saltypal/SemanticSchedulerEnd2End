"""Semantic Token Summary (13 floats in [0,1]) from a segmentation map and the kept-token set.

Layout: 7 kept-token class-group fractions (person, vehicle, building, road, vegetation, sky, other);
4 retention ratios (kept/all tokens of person, vehicle, building, road; 0 if absent); then 2 values
for kept person+vehicle tokens: distance of their centroid from the image centre (normalised by the
max possible, sqrt(0.5) in unit coordinates) and spatial spread (Shannon entropy over a 4x8 coarse cell
histogram, normalised by log(32)). Both are 0 when no person/vehicle token is kept.
Class ids: Cityscapes 19 train ids 0..18 plus 19 = void.
"""
import numpy as np

GROUPS = ("person", "vehicle", "building", "road", "vegetation", "sky", "other")
_GROUP_OF = {"person": (11, 12), "vehicle": (13, 14, 15, 16, 17, 18), "building": (2, 3, 4),
             "road": (0, 1), "vegetation": (8, 9), "sky": (10,), "other": (5, 6, 7, 19)}
# class id (0..19: road, sidewalk, building, wall, fence, pole, light, sign, vegetation, terrain, sky,
# person, rider, car, truck, bus, train, motorcycle, bicycle, void) -> group index
SUPERCLASS = np.zeros(20, dtype=np.int64)
for _g, _name in enumerate(GROUPS):
    SUPERCLASS[list(_GROUP_OF[_name])] = _g


def majority_vote_grid(seg_map_20cls, tile=16):
    """Per-tile majority class (ties -> lowest id). (H, W) -> (H/tile, W/tile)."""
    seg = np.asarray(seg_map_20cls)
    h, w = seg.shape
    if h % tile or w % tile:
        raise ValueError("map shape %r not divisible by tile %d" % (seg.shape, tile))
    if seg.min() < 0 or seg.max() > 19:
        raise ValueError("class ids must be in 0..19")
    t = seg.reshape(h // tile, tile, w // tile, tile).transpose(0, 2, 1, 3).reshape(h // tile, w // tile, -1)
    counts = np.stack([(t == c).sum(axis=2) for c in range(20)], axis=2)
    return counts.argmax(axis=2)


def compute_sts(seg_map, kept_idx, grid=(16, 32)):
    """13-float STS. ``seg_map`` is a 20-class pixel map or an already-voted ``grid`` class map."""
    seg = np.asarray(seg_map)
    cls = seg if seg.shape == tuple(grid) else majority_vote_grid(seg, seg.shape[0] // grid[0])
    if cls.shape != tuple(grid):
        raise ValueError("segmentation does not reduce to grid %r" % (grid,))
    n = grid[0] * grid[1]
    kept = np.unique(np.asarray(kept_idx, dtype=np.int64))
    if kept.size and (kept.min() < 0 or kept.max() >= n):
        raise ValueError("kept_idx outside 0..%d" % (n - 1))
    group = SUPERCLASS[cls.ravel()]
    kept_g = group[kept]
    kept_count = np.bincount(kept_g, minlength=7)
    hist = kept_count / max(kept.size, 1)
    total = np.bincount(group, minlength=7)[:4]
    retention = np.where(total > 0, kept_count[:4] / np.maximum(total, 1), 0.0)
    pos = kept[kept_g <= 1]                                   # kept person + vehicle tokens
    dist = spread = 0.0
    if pos.size:
        ry, rx = pos // grid[1] / (grid[0] - 1), pos % grid[1] / (grid[1] - 1)
        dist = float(np.hypot(rx.mean() - 0.5, ry.mean() - 0.5) / np.sqrt(0.5))
        cell = (pos // grid[1]) * 4 // grid[0] * 8 + (pos % grid[1]) * 8 // grid[1]
        p = np.bincount(cell, minlength=32) / pos.size
        p = p[p > 0]
        spread = float(-(p * np.log(p)).sum() / np.log(32))
    return np.clip(np.concatenate([hist, retention, [dist, spread]]), 0.0, 1.0).tolist()
