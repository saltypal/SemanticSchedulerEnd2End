"""Candidate image pairs for human duplicate labelling (plan S2b), stratified by key distance.

Distance bands (``BANDS``) over the semantic Hamming distance cover near, borderline and far pairs so
the labelled set brackets the decision boundary. Rows need ``id`` and optional ``tweet_id``.
"""
import csv

import numpy as np

from parameters.edge.features import hamming

BANDS = (("near", 0, 8), ("borderline", 9, 20), ("far", 21, 64))   # inclusive semantic-distance ranges
COLUMNS = ("id_a", "id_b", "d_sem", "d_perc", "same_tweet", "label")


def candidate_pairs(rows, sem_keys, perc_keys, n_pairs, rng, max_tries=200):
    """~``n_pairs`` pairs split evenly across bands. Tuples (id_a, id_b, d_sem, d_perc, same_tweet)."""
    n = len(rows)
    if n < 2:
        raise ValueError("need at least two rows")
    quota = {b[0]: n_pairs // len(BANDS) + (i < n_pairs % len(BANDS)) for i, b in enumerate(BANDS)}
    got, seen = {b[0]: [] for b in BANDS}, set()
    for _ in range(max_tries * max(n_pairs, 1)):
        if all(len(got[k]) >= quota[k] for k in quota):
            break
        i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
        if (i, j) in seen:
            continue
        seen.add((i, j))
        ds = hamming(sem_keys[i], sem_keys[j])
        band = next(name for name, lo, hi in BANDS if lo <= ds <= hi)
        if len(got[band]) < quota[band]:
            same = bool(rows[i].get("tweet_id")) and rows[i].get("tweet_id") == rows[j].get("tweet_id")
            got[band].append((rows[i]["id"], rows[j]["id"], ds, hamming(perc_keys[i], perc_keys[j]), same))
    return [p for b in BANDS for p in got[b[0]]]


def _nearest(keys):
    """Index and Hamming distance of each key's nearest other key (uint64 array, chunked)."""
    keys = np.asarray(keys, dtype=np.uint64)
    idx, dist = np.zeros(len(keys), dtype=np.int64), np.full(len(keys), 65, dtype=np.int64)
    for i0 in range(0, len(keys), 512):
        d = np.bitwise_count(keys[i0:i0 + 512, None] ^ keys[None, :]).astype(np.int64)
        d[np.arange(d.shape[0]), np.arange(i0, i0 + d.shape[0])] = 65   # ignore self
        idx[i0:i0 + 512], dist[i0:i0 + 512] = d.argmin(axis=1), d.min(axis=1)
    return idx, dist


def mined_candidates(rows, sem_keys, perc_keys, quotas, rng):
    """S2c candidates that actually contain duplicates (plan: "retweets + near matches").

    quotas: {"phash_nn": n, "simhash_nn": n, "same_tweet": n, "random": n}. Nearest-neighbour pairs are
    taken evenly across distances (so the labels bracket the boundary); random pairs use the BANDS sampler.
    """
    sem, perc = np.asarray(sem_keys, dtype=np.uint64), np.asarray(perc_keys, dtype=np.uint64)
    chosen, seen = [], set()

    def add(i, j):
        i, j = sorted((int(i), int(j)))
        if i == j or (i, j) in seen:
            return False
        seen.add((i, j))
        same = bool(rows[i].get("tweet_id")) and rows[i].get("tweet_id") == rows[j].get("tweet_id")
        chosen.append((rows[i]["id"], rows[j]["id"], hamming(sem[i], sem[j]), hamming(perc[i], perc[j]), same))
        return True

    for name, keys in (("phash_nn", perc), ("simhash_nn", sem)):
        nn, d = _nearest(keys)
        order = np.argsort(d, kind="stable")
        q = quotas.get(name, 0)
        # half from the very closest pairs (true copies), half spread up to distance 20 (the boundary)
        spread = np.linspace(0, min(len(order), max(1, int((d <= 20).sum()))) - 1, q * 2).astype(int)
        picks = np.concatenate([np.arange(min(q, len(order))), spread])
        n0 = len(chosen)
        for p in picks:
            if len(chosen) - n0 >= quotas.get(name, 0):
                break
            add(order[p], nn[order[p]])
    by_tweet = {}
    for i, r in enumerate(rows):
        if r.get("tweet_id"):
            by_tweet.setdefault(r["tweet_id"], []).append(i)
    groups = [g for g in by_tweet.values() if len(g) > 1]
    rng.shuffle(groups)
    n0 = len(chosen)
    for g in groups:
        if len(chosen) - n0 >= quotas.get("same_tweet", 0):
            break
        add(g[0], g[1])
    pos = {r["id"]: k for k, r in enumerate(rows)}
    for p in candidate_pairs(rows, sem, perc, quotas.get("random", 0), rng):
        add(pos[p[0]], pos[p[1]])
    return chosen


def write_csv(pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for p in pairs:
            w.writerow(list(p) + [""])


def load_labelled(path):
    """Read a labelled CSV; every label must be ``duplicate`` or ``distinct``."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["label"] not in ("duplicate", "distinct"):
            raise ValueError("bad label %r for pair %s/%s" % (r["label"], r["id_a"], r["id_b"]))
    return rows
