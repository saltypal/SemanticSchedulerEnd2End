"""Simulator table builder (plan v3 sec. 9 inputs; S3 finalize; D14, D15). One row per image, everything precomputed.

Inputs: dataset B (``labels_raw.jsonl``, ``splits_v1.json``) and dataset C1 (``shards/*``: CLIP-B/32 embeddings,
SimHash, pHash). Output folder ``--out``:
  simtable_v1.jsonl   rows in the ``Evaluation.edge.tables`` schema (+ extra columns, see ROW_DOC)
  spi_head.npz        final SPI head (CLIP-B/32 logistic C=1, class-balanced) + temperature
  spi_report.json     val-only SPI metrics, per source, CrisisMMD-only, no-normals ablation (test is never scored)
  manifest.json       inputs, counts, placeholders still to fill from the SQ-GAN v1 pass

Codec-dependent fields (STS, payload bytes, latency) are placeholders until the SQ-GAN v1 full pass (C2) exists:
``sts == []`` and ``payload_source == "placeholder"``; ``fill_codec`` merges C2 later.

    python -m parameters.edge.simtable --labels-root <B> --cache-root <C1> --out <dir>

``fill-codec`` merges the SQ-GAN v1 pass (C2 = the feature-pass cache folder(s) of ``--codec v1``) into a built table:
    python -m parameters.edge.simtable fill-codec --table <simtable_v1.jsonl> --c2 <C2 dir> [<C2 dir> ...] --out <dir>
and writes ``simtable_v1.jsonl``, ``latency_v1.json`` (``{"samples": {"1": [s, ...], ...}}``, read by
``Evaluation.edge.tables.latency_from_profile_json``) and ``codec_fill_report.json`` into ``--out``.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from Evaluation.edge.contracts import TIERS
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.tables import validate_row
from parameters.edge import labels as labels_mod
from parameters.edge import spi as spi_mod

TWITTER_EPOCH_MS = 1288834974657
PLACEHOLDER_PAYLOAD = 464          # bytes at m = 0.25 (checklist), until the SQ-GAN v1 pass
SPI_C = 1.0                        # chosen head (journal 2026-11-11): CLIP-B/32 logistic C=1
ROW_DOC = {"tier": "int 0..3 (Critical..Low), plan v3 main mapping", "tier_alt": "v3 sensitivity (affected -> High)",
           "tier_probs": "calibrated SPI; out-of-fold for train rows", "relevance": "for the row's default mission",
           "relevance_by_mission": "one value per mission id (D15)", "dup_group": "D14 rule components, null if alone",
           "capture_time_s": "tweet snowflake time for CrisisMMD, null for everyday images",
           "fidelity": "always 1 (D5: fidelity dropped)"}


# ----------------------------------------------------------------------------- small pure helpers
def tweet_time_s(tweet_id):
    """Unix seconds encoded in a Twitter snowflake id (post time, not capture time: v3 sec. 9 limitation)."""
    try:
        return ((int(tweet_id) >> 22) + TWITTER_EPOCH_MS) / 1000.0
    except (TypeError, ValueError):
        return None


def relevance_matrix(emb, mission_emb):
    """(n, m) relevance in [0, 1] = (cos + 1) / 2 for L2-normalised rows (same map as features.relevance)."""
    e = np.asarray(emb, np.float64)
    m = np.asarray(mission_emb, np.float64)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return np.clip((e @ m.T + 1.0) / 2.0, 0.0, 1.0)


def dup_groups(sem, perc, sem_max, perc_max, chunk=512):
    """Connected components of 'pHash <= perc_max or SimHash <= sem_max' (D14). Returns int array, -1 = alone."""
    sem, perc = np.asarray(sem, np.uint64), np.asarray(perc, np.uint64)
    n = len(sem)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i0 in range(0, n, chunk):
        ds = np.bitwise_count(sem[i0:i0 + chunk, None] ^ sem[None, :])
        dp = np.bitwise_count(perc[i0:i0 + chunk, None] ^ perc[None, :])
        a, b = np.nonzero((ds <= sem_max) | (dp <= perc_max))
        for i, j in zip(a + i0, b):
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    sizes = Counter(roots.tolist())
    ids, out = {}, np.full(n, -1)
    for i, r in enumerate(roots):
        if sizes[r] > 1:
            out[i] = ids.setdefault(r, len(ids))
    return out


# ----------------------------------------------------------------------------- inputs
def load_cache(cache_root, encoder="clip_b32"):
    """uid -> row index; arrays of embeddings, SimHash, pHash (rows whose features failed are skipped)."""
    uids, emb, sem, per = [], [], [], []
    for shard in sorted((Path(cache_root) / "shards").iterdir()):
        u = json.loads((shard / "uids.json").read_text(encoding="utf-8"))
        ok = (np.load(shard / ("emb_%s_ok.npy" % encoder)) & np.load(shard / "simhash_ok.npy")
              & np.load(shard / "phash_ok.npy"))
        e, s, p = np.load(shard / ("emb_%s.npy" % encoder)), np.load(shard / "simhash64.npy"), np.load(shard / "phash64.npy")
        for i in np.nonzero(ok)[0]:
            uids.append(u[i]); emb.append(e[i]); sem.append(s[i]); per.append(p[i])
    return {u: i for i, u in enumerate(uids)}, np.asarray(emb, np.float32), np.asarray(sem, np.uint64), np.asarray(per, np.uint64)


def mission_embeddings(missions_cfg, device="cpu"):
    """(n_missions, 512) mean prompt embedding per mission, same CLIP model as the cached image embeddings."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(missions_cfg["model"])
    model = CLIPModel.from_pretrained(missions_cfg["model"]).to(device).eval()
    out = []
    with torch.no_grad():
        for m in sorted(missions_cfg["missions"], key=lambda x: x["id"]):
            t = tok(m["prompts"], padding=True, return_tensors="pt").to(device)
            f = model.text_projection(model.text_model(**t).pooler_output)
            f = torch.nn.functional.normalize(f, dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.stack(out)


# ----------------------------------------------------------------------------- SPI (S3 finalize)
def fit_final_spi(X, y, split, source, seed=0, folds=5):
    """Head on train, temperature on val, out-of-fold calibrated probs for train. Returns (probs, head, report)."""
    from sklearn.model_selection import StratifiedKFold
    tr, va = split == "train", split == "val"
    head = spi_mod._fit_lr(X[tr], y[tr], SPI_C, seed)
    head.temperature = spi_mod.fit_temperature(head.logits(X[va]), y[va])
    probs = head.predict_proba(X)
    oof = np.zeros((tr.sum(), 4))
    idx = np.nonzero(tr)[0]
    for k, (a, b) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(idx, y[tr])):
        h = spi_mod._fit_lr(X[idx[a]], y[idx[a]], SPI_C, seed)
        h.temperature = head.temperature
        oof[b] = h.predict_proba(X[idx[b]])
    probs[tr] = oof
    rep = {"val": spi_mod.evaluate_spi(head, X[va], y[va], source[va]),
           "val_crisismmd_only": spi_mod.evaluate_spi(head, X[va & (source == "crisismmd")],
                                                       y[va & (source == "crisismmd")]),
           "train_out_of_fold": spi_mod._report(oof, y[tr]),
           "temperature": head.temperature, "C": SPI_C, "test_scored": False}
    everyday_val = source[va] != "crisismmd"
    rep["no_normals_ablation_val"] = (spi_mod.no_normals_ablation(X[tr], y[tr], source[tr] != "crisismmd", X[va],
                                                                  everyday_val, SPI_C, seed)
                                      if everyday_val.any() else None)
    return probs, head, rep


# ----------------------------------------------------------------------------- build
def build(label_rows, splits, cache, mission_emb, missions_cfg, dedup_cfg):
    """Return (rows, report). Rows are validated against the simulator schema."""
    pos, emb, sem, per = cache
    alt_rows, _ = labels_mod.retier_rows(label_rows, alternative="affected_high")
    main_rows, changed = labels_mod.retier_rows(label_rows)
    alt_tier = {r["uid"]: r["tier"] for r in alt_rows}
    rows = [r for r in main_rows if r["uid"] in splits and r["uid"] in pos]
    ix = np.array([pos[r["uid"]] for r in rows])
    X, y = emb[ix].astype(np.float64), np.array([TIERS.index(r["tier"]) for r in rows])
    split = np.array([splits[r["uid"]] for r in rows])
    source = np.array([r["source"] for r in rows])
    probs, head, spi_rep = fit_final_spi(X, y, split, source)
    rel = relevance_matrix(X, mission_emb)
    groups = dup_groups(sem[ix], per[ix], dedup_cfg["simhash_max"], dedup_cfg["phash_max"])
    by_name = {m["name"]: m["id"] for m in missions_cfg["missions"]}
    out = []
    for i, r in enumerate(rows):
        mission = by_name[missions_cfg["event_mission"].get(r.get("event") or "", missions_cfg["everyday_mission"])]
        p = probs[i] / probs[i].sum()
        row = {"image_id": r["uid"], "event": r.get("event"), "source": r["source"], "split": str(split[i]),
               "tier": int(y[i]), "tier_alt": TIERS.index(alt_tier[r["uid"]]), "tier_probs": [float(v) for v in p],
               "mission": int(mission), "relevance": float(rel[i, mission]),
               "relevance_by_mission": [float(v) for v in rel[i]],
               "sem_key": int(sem[ix[i]]), "perc_key": int(per[ix[i]]), "sts": [], "payload_bytes": PLACEHOLDER_PAYLOAD,
               "payload_source": "placeholder", "codec": "v1", "fidelity": 1,
               "dup_group": None if groups[i] < 0 else int(groups[i]),
               "tweet_id": r.get("tweet_id"), "capture_time_s": tweet_time_s(r.get("tweet_id"))}
        out.append(validate_row(row))
    counts = {s: dict(Counter(TIERS[r["tier"]] for r in out if r["split"] == s)) for s in ("train", "val", "test")}
    report = {"rows": len(out), "tier_changes_vs_dataset_b": {"%s->%s" % k: v for k, v in changed.items()},
              "counts": counts, "dup_groups": int(groups.max() + 1), "rows_in_dup_groups": int((groups >= 0).sum()),
              "spi": spi_rep}
    return out, head, report


# ----------------------------------------------------------------------------- C2: SQ-GAN v1 pass merge
def _fkey(m):
    return "%g" % float(m)


def load_c2(c2_dirs):
    """Read feature-pass cache folder(s) -> ({image_id: fields}, latency samples {batch: [s]}).

    Per image (only when the first, smallest m encoded): ``sts`` (13 floats of that m), ``payload_bytes_by_m`` (every
    m that encoded), ``lpips`` and ``recon_miou`` (first m, when finite)."""
    out, samples = {}, {}
    for root in ([c2_dirs] if isinstance(c2_dirs, (str, Path)) else c2_dirs):
        for shard in sorted((Path(root) / "shards").glob("shard_*")):
            for lat in shard.glob("latency_samples.json"):
                for b, vals in json.loads(lat.read_text(encoding="utf-8"))["samples"].items():
                    samples.setdefault(str(b), []).extend(float(v) for v in vals)
            if not (shard / "uids.json").exists() or not (shard / "sqgan_ok.npy").exists():
                continue
            uids = json.loads((shard / "uids.json").read_text(encoding="utf-8"))
            tags = sorted((f.stem[len("offsets_"):] for f in shard.glob("offsets_m*.npy")), key=lambda t: float(t[1:]))
            if not tags:
                continue
            ok = np.load(shard / "sqgan_ok.npy").reshape(len(uids), -1)
            offs = {t: np.load(shard / ("offsets_%s.npy" % t)) for t in tags}
            t0 = tags[0]
            sts0 = np.load(shard / ("sts_%s.npy" % t0))
            q = {k: np.load(shard / ("%s_%s.npy" % (k, t0))) if (shard / ("%s_%s.npy" % (k, t0))).exists() else None
                 for k in ("lpips", "miou")}
            for i, uid in enumerate(uids):
                if not ok[i, 0]:
                    continue
                by_m = {_fkey(t[1:]): int(offs[t][i + 1] - offs[t][i]) for j, t in enumerate(tags)
                        if j < ok.shape[1] and ok[i, j] and offs[t][i + 1] > offs[t][i]}
                if _fkey(t0[1:]) not in by_m:
                    continue
                rec = {"sts": [float(v) for v in sts0[i]], "payload_bytes_by_m": by_m}
                for src, name in (("lpips", "lpips"), ("miou", "recon_miou")):
                    if q[src] is not None and np.isfinite(q[src][i]):
                        rec[name] = float(q[src][i])
                out[uid] = rec
    return out, samples


def fill_codec(rows, c2, samples=None):
    """Merge C2 into table rows by image_id. Returns (rows, report, latency dict or None).

    Filled rows get ``sts``, ``payload_bytes`` (first/smallest m, 0.25), ``payload_bytes_by_m``, ``payload_source
    "sqgan_v1"`` and, when present, ``lpips`` / ``recon_miou``. Rows absent from C2 keep their placeholders."""
    out, missing, filled = [], [], 0
    for r in rows:
        rec = c2.get(r["image_id"])
        if rec is None:
            missing.append(r["image_id"])
            out.append(validate_row(dict(r)))
            continue
        row = dict(r)
        by_m = rec["payload_bytes_by_m"]
        row.update(sts=list(rec["sts"]), payload_bytes=by_m[min(by_m, key=float)], payload_bytes_by_m=dict(by_m),
                   payload_source="sqgan_v1")
        for k in ("lpips", "recon_miou"):
            if k in rec:
                row[k] = rec[k]
        out.append(validate_row(row))
        filled += 1
    ids = {r["image_id"] for r in rows}
    report = {"rows": len(rows), "filled": filled, "missing_from_c2": len(missing), "missing_sample": missing[:20],
              "c2_not_in_table": len([u for u in c2 if u not in ids]),
              "latency_batches": {b: len(v) for b, v in (samples or {}).items()}}
    latency = {"samples": {b: list(v) for b, v in sorted(samples.items(), key=lambda kv: int(kv[0]))},
               "emulated_batching": True, "source": "sqgan_v1 feature pass"} if samples else None
    return out, report, latency


def fill_main(argv):
    ap = argparse.ArgumentParser(prog="python -m parameters.edge.simtable fill-codec", description=fill_codec.__doc__)
    ap.add_argument("--table", required=True, help="simtable_v1.jsonl (or the folder holding it)")
    ap.add_argument("--c2", required=True, nargs="+", help="feature-pass cache folder(s) of the SQ-GAN v1 pass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    table = Path(a.table)
    table = table / "simtable_v1.jsonl" if table.is_dir() else table
    rows = [json.loads(x) for x in table.read_text(encoding="utf-8").splitlines() if x.strip()]
    c2, samples = load_c2(a.c2)
    out_rows, report, latency = fill_codec(rows, c2, samples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "simtable_v1.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    if latency:
        (out / "latency_v1.json").write_text(json.dumps(latency, indent=1), encoding="utf-8")
    (out / "codec_fill_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fill-codec":
        return fill_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lroot, out = Path(a.labels_root), Path(a.out)
    label_rows = [json.loads(x) for x in (lroot / "labels_raw.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    splits = json.loads((lroot / "splits_v1.json").read_text(encoding="utf-8"))["splits"]
    missions_cfg = load_config("missions")
    rows, head, report = build(label_rows, splits, load_cache(a.cache_root), mission_embeddings(missions_cfg),
                               missions_cfg, load_config("data")["splits"]["dedup"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "simtable_v1.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(out / "spi_head.npz", coef=head.coef, intercept=head.intercept, classes=np.array(head.classes),
             temperature=head.temperature)
    (out / "spi_report.json").write_text(json.dumps(report["spi"], indent=1), encoding="utf-8")
    manifest = {k: v for k, v in report.items() if k != "spi"}
    manifest.update({"git_sha": git_sha(), "columns": ROW_DOC, "missions": missions_cfg["missions"],
                     "placeholders": {"sts": "empty until SQ-GAN v1 full pass (C2)",
                                      "payload_bytes": "%d until C2" % PLACEHOLDER_PAYLOAD,
                                      "latency": "not in this table; LatencyProfile JSON from the SQ-GAN pass"}})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    sp = report["spi"]["val"]
    print(json.dumps({"rows": report["rows"], "tier_changes": report["tier_changes_vs_dataset_b"],
                      "counts": report["counts"], "dup_groups": report["dup_groups"],
                      "rows_in_dup_groups": report["rows_in_dup_groups"],
                      "val_macro_f1": round(sp["macro_f1"], 4), "val_critical_recall": round(sp["critical_recall"], 4),
                      "val_ece": round(sp["ece"], 4)}, indent=1))


if __name__ == "__main__":
    main()
