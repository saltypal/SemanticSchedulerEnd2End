"""Event-disjoint split assignment and cross-split near-duplicate removal (plan v3.1 sec. 3, S2c).

Rows are dicts with at least ``image_path`` and optionally ``event``. Event splits come from
``config/edge/data.yaml``; rows from no listed event (everyday images) split 60/10/30 by a stable
SHA-256 hash of ``image_path``. No randomness.
"""
import hashlib
from collections import Counter

import numpy as np

from Evaluation.edge.edge_config import load_config

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hash_unit(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2.0 ** 64


def assign_event_splits(rows, cfg=None):
    """Return ``(splits, report)``: ``splits[image_path] -> train|val|test``."""
    cfg = cfg or load_config("data")["splits"]
    by_event = {e: s for s, key in (("train", "train_events"), ("val", "val_events"),
                                    ("test", "test_events")) for e in cfg[key]}
    fractions = cfg.get("normals_split", [0.6, 0.1, 0.3])
    cut_train, cut_val = fractions[0], fractions[0] + fractions[1]
    splits = {}
    for r in rows:
        event = r.get("event")
        if not event:           # everyday images have no event: stable hash split
            u = _hash_unit(r["image_path"])
            s = "train" if u < cut_train else "val" if u < cut_val else "test"
        elif event in by_event:
            s = by_event[event]
        else:                   # an unlisted event would silently break event-disjointness
            raise ValueError("event %r is not assigned to a split in config/edge/data.yaml" % event)
        splits[r["image_path"]] = s
    return splits, {"counts": dict(Counter(splits.values()))}


def _hamming_min(keys_a, keys_b):
    """For each key in a, min Hamming distance to any key in b (uint64 arrays)."""
    out = np.full(len(keys_a), 64, dtype=np.int64)
    if len(keys_b) == 0:
        return out
    for i, k in enumerate(keys_a):
        x = np.bitwise_xor(keys_b, k).view(np.uint8).reshape(len(keys_b), 8)
        out[i] = _POP[x].sum(axis=1).min()
    return out


def drop_cross_split_near_duplicates(rows, sem_keys, perc_keys, sem_thr, perc_thr):
    """Drop val/test rows within ``sem_thr`` (semantic) OR ``perc_thr`` (perceptual) bits of a train row.

    ``rows[i]['split']`` in {train,val,test}; keys are parallel to rows. Returns
    ``(kept_rows, report)`` with ``report['dropped']`` listing dropped indices per split.
    """
    if not (len(rows) == len(sem_keys) == len(perc_keys)):
        raise ValueError("rows and keys must have equal length")
    sem = np.array([int(k) for k in sem_keys], dtype=np.uint64)
    perc = np.array([int(k) for k in perc_keys], dtype=np.uint64)
    split = np.array([r["split"] for r in rows])
    tr = split == "train"
    other = np.where(~tr)[0]
    d_s = _hamming_min(sem[other], sem[tr])
    d_p = _hamming_min(perc[other], perc[tr])
    drop = set(other[(d_s <= sem_thr) | (d_p <= perc_thr)].tolist())
    kept = [r for i, r in enumerate(rows) if i not in drop]
    return kept, {"before": dict(Counter(split.tolist())), "after": dict(Counter(r["split"] for r in kept)),
                  "dropped": {s: sorted(i for i in drop if split[i] == s) for s in ("val", "test")}}
