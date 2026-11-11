"""S2b + S2c driver: freeze event-disjoint splits with cross-split near-duplicate removal (plan v3 sec. 10).

Inputs: dataset B (``labels_raw.jsonl``) and dataset C1 (``shards/*/{uids.json,simhash64.npy,phash64.npy}``).
Outputs in ``--out``:
  splits_v1.json            {"version", "rule", "splits": {uid: train|val|test}}   (dropped uids are absent)
  split_report.json         counts per split x tier before/after, every dropped uid with its distances
  dup_pairs_candidates.csv  S2c pairs (train + val only) for the team to label duplicate / distinct

    python -m parameters.edge.freeze_splits --labels-root <B> --cache-root <C1> --out <dir>
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from Evaluation.edge.edge_config import load_config
from parameters.edge import dup_pairs
from parameters.edge.splits import assign_event_splits, drop_cross_split_near_duplicates


def load_labels(labels_root):
    with open(Path(labels_root) / "labels_raw.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_keys(cache_root):
    """uid -> (simhash, phash) from the cache shards; rows whose hash failed are skipped."""
    keys = {}
    for shard in sorted((Path(cache_root) / "shards").iterdir()):
        uids = json.loads((shard / "uids.json").read_text(encoding="utf-8"))
        sem, per = np.load(shard / "simhash64.npy"), np.load(shard / "phash64.npy")
        ok = np.load(shard / "simhash_ok.npy") & np.load(shard / "phash_ok.npy")
        for u, s, p, k in zip(uids, sem, per, ok):
            if k:
                keys[u] = (int(s), int(p))
    return keys


def _tier_counts(rows):
    out = {}
    for r in rows:
        out.setdefault(r["split"], Counter())[r["tier"]] += 1
    return {s: dict(c) for s, c in sorted(out.items())}


def freeze(labels, keys, cfg):
    """Return (splits, report, rows_kept). Rows without keys are kept (cannot be checked) and reported."""
    rows = [dict(r, id=r["uid"]) for r in labels]
    assigned, _ = assign_event_splits([{"image_path": r["uid"], "event": r.get("event")} for r in rows], cfg)
    for r in rows:
        r["split"] = assigned[r["uid"]]
    have = [r for r in rows if r["uid"] in keys]
    missing = [r["uid"] for r in rows if r["uid"] not in keys]
    d = cfg["dedup"]
    kept, rep = drop_cross_split_near_duplicates(
        have, [keys[r["uid"]][0] for r in have], [keys[r["uid"]][1] for r in have],
        sem_thr=d["simhash_max"], perc_thr=d["phash_max"])
    kept_rows = kept + [r for r in rows if r["uid"] in set(missing)]
    dropped = [have[i]["uid"] for s in ("val", "test") for i in rep["dropped"][s]]
    splits = {r["uid"]: r["split"] for r in kept_rows}
    report = {"rule": {"phash_max": d["phash_max"], "simhash_max": d["simhash_max"],
                       "drop": "val/test image within threshold of any train image"},
              "before": _tier_counts(rows), "after": _tier_counts(kept_rows),
              "n_dropped": len(dropped), "dropped": dropped, "no_keys": missing,
              "gates": gate_check(kept_rows, cfg)}
    return splits, report, kept_rows


def gate_check(rows, cfg):
    g = cfg["gates"]
    crit = Counter(r["split"] for r in rows if r["tier"] == "Critical")
    return {"test_critical": crit["test"], "test_ok": crit["test"] >= g["min_test_critical"],
            "val_critical": crit["val"], "val_ok": crit["val"] >= g["min_val_critical"]}


def candidate_pairs(rows, keys, cfg):
    """S2c: candidate duplicate pairs from train + val only (test is never looked at)."""
    pool = [r for r in rows if r["split"] in ("train", "val") and r["uid"] in keys]
    rng = np.random.default_rng(cfg.get("dup_pair_seed", 22))
    n = cfg.get("dup_pair_candidates", 400)
    quotas = {"phash_nn": n // 4, "simhash_nn": n // 4, "same_tweet": n // 8}
    quotas["random"] = n - sum(quotas.values())
    return dup_pairs.mined_candidates(pool, [keys[r["uid"]][0] for r in pool], [keys[r["uid"]][1] for r in pool],
                                      quotas, rng)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = load_config("data")["splits"]
    labels, keys = load_labels(a.labels_root), load_keys(a.cache_root)
    splits, report, kept = freeze(labels, keys, cfg)
    if not (report["gates"]["test_ok"] and report["gates"]["val_ok"]):
        raise SystemExit("split gates failed after dedup: %s" % report["gates"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits_v1.json").write_text(json.dumps({"version": 1, "rule": report["rule"], "splits": splits}),
                                        encoding="utf-8")
    (out / "split_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    dup_pairs.write_csv(candidate_pairs(kept, keys, cfg), out / "dup_pairs_candidates.csv")
    print(json.dumps({"after": report["after"], "n_dropped": report["n_dropped"], "gates": report["gates"],
                      "no_keys": len(report["no_keys"])}, indent=1))


if __name__ == "__main__":
    main()
