"""Candidate image pairs for human duplicate labelling (plan S2b), stratified by key distance.

Distance bands (``BANDS``) over the semantic Hamming distance cover near, borderline and far pairs so
the labelled set brackets the decision boundary. Rows need ``id`` and optional ``tweet_id``.
"""
import csv

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
