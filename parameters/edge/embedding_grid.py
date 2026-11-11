"""Resumable embedding extraction + SPI classifier grid search (plan v3 S3, CHECKLIST step 5).

Extraction writes ``emb_<encoder>.npz`` (image_path, emb float16) in shards with a done-marker, guards
disk space before starting, and frees GPU memory/temp files after each encoder. The grid trains every
classifier/hyperparameter candidate on train rows, evaluates on val only (macro-F1, Critical recall, ECE,
per-source) and saves one JSON per candidate immediately. Selection applies the Critical-recall gate and
tie-breaks; the winner is temperature-scaled via ``parameters.edge.spi``. Test-split rows are never used
except in ``final_eval(..., confirm_test=True)``. Heavy libraries are imported lazily.
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from Evaluation.edge.edge_config import load_config
from parameters.edge import spi
from parameters.edge.spi import SPIHead, N_CLASSES, TIERS, _softmax


# ---------------------------------------------------------------- extraction

def disk_guard(min_free_gb, path="."):
    """Raise before any write if free space at ``path`` is below ``min_free_gb``."""
    free = shutil.disk_usage(str(path)).free / 1e9
    if free < float(min_free_gb):
        raise RuntimeError("not enough free disk: {:.2f} GB free < {:.2f} GB required at {}".format(
            free, float(min_free_gb), path))
    return free


def _atomic_savez(path, **arrays):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, str(path))


def _free_gpu():
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _encoder_cfg(name, grid_cfg=None):
    cfg = grid_cfg or load_config("grid")
    for e in cfg["encoders"]:
        if e["name"] == name:
            return e
    raise ValueError("unknown encoder " + name)


def _first_attr(obj, names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _image_embedding(model, family, pixel_values):
    """Raw (n, d) image embedding tensor, robust to transformers 4 (tensor) and 5 (ModelOutput) APIs."""
    if family == "clip":
        # vision tower + projection: the 512-d projected embedding, independent of get_image_features' return type
        pooled = model.vision_model(pixel_values=pixel_values).pooler_output
        return model.visual_projection(pooled)
    if family == "siglip":
        out = model.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "reshape"):                       # a plain tensor
            return out
        f = _first_attr(out, ("pooler_output", "image_embeds"))
        if f is None:
            raise ValueError("siglip get_image_features returned %s without pooler_output/image_embeds"
                             % type(out).__name__)
        return f
    out = model(pixel_values=pixel_values)
    f = getattr(out, "pooler_output", None)
    return f if f is not None else out.last_hidden_state[:, 0]


def _check_dim(f, enc):
    got = int(f.shape[-1])
    if got != int(enc["dim"]):
        raise ValueError("encoder %s: embedding dim got %d, expected %d (grid.yaml); output type %s"
                         % (enc["name"], got, int(enc["dim"]), type(f).__name__))


def load_embedder(name, device="cpu", grid_cfg=None):
    """Return ``fn(list[PIL.Image]) -> (n, dim) float32`` L2-normalised. Lazy transformers/torch import."""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    enc = _encoder_cfg(name, grid_cfg)
    proc = AutoImageProcessor.from_pretrained(enc["hf_id"])
    model = AutoModel.from_pretrained(enc["hf_id"]).to(device).eval()

    def embed(images):
        inputs = proc(images=[im.convert("RGB") for im in images], return_tensors="pt").to(device)
        with torch.no_grad():
            f = _image_embedding(model, enc["family"], inputs["pixel_values"])
            f = f.reshape(f.shape[0], -1)
            _check_dim(f, enc)
            f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy().astype(np.float32)
    embed.model = model
    return embed


def extract_encoder(name, items, loader, batch_size, device, out_dir, shard_size=500,
                    min_free_gb=3.0, embedder=None, max_images=None):
    """Extract embeddings for ``items`` (image paths) into ``<out_dir>/emb_<name>.npz`` in shards.

    Resumable: finished shards and a finished encoder (``emb_<name>.done``) are skipped. ``loader(item)``
    returns a PIL image; ``embedder`` (injected in tests) maps a list of images to embeddings.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final, marker = out_dir / ("emb_%s.npz" % name), out_dir / ("emb_%s.done" % name)
    if marker.exists() and final.exists():
        return final
    items = list(items)[:max_images] if max_images else list(items)
    disk_guard(min_free_gb, out_dir)                      # before loading anything
    shard_dir = out_dir / ("shards_%s" % name)
    shard_dir.mkdir(exist_ok=True)
    n_shards = (len(items) + shard_size - 1) // shard_size
    emb_fn = embedder
    try:
        for s in range(n_shards):
            shard = shard_dir / ("shard_%05d.npz" % s)
            if shard.exists():
                continue
            disk_guard(min_free_gb, out_dir)
            if emb_fn is None:
                emb_fn = load_embedder(name, device)
            chunk = items[s * shard_size:(s + 1) * shard_size]
            parts = []
            for i in range(0, len(chunk), batch_size):
                parts.append(np.asarray(emb_fn([loader(it) for it in chunk[i:i + batch_size]])))
            _atomic_savez(shard, image_path=np.array([str(c) for c in chunk]),
                          emb=np.concatenate(parts).astype(np.float16))
        paths, embs = [], []
        for s in range(n_shards):
            with np.load(shard_dir / ("shard_%05d.npz" % s), allow_pickle=False) as z:
                paths.append(z["image_path"])
                embs.append(z["emb"])
        _atomic_savez(final, image_path=np.concatenate(paths), emb=np.concatenate(embs))
        marker.write_text(json.dumps({"n": len(items), "time": time.time()}), encoding="utf-8")
        shutil.rmtree(shard_dir, ignore_errors=True)
    finally:
        emb_fn = None
        _free_gpu()
    return final


def ue_cost_probe(encoder_name, loader, n=16, device="cpu", items=None, grid_cfg=None):
    """Parameter count and ms/image of an encoder on ``n`` items (heavy; not used in tests)."""
    embed = load_embedder(encoder_name, device, grid_cfg)
    params = sum(p.numel() for p in embed.model.parameters())
    imgs = [loader(it) for it in list(items)[:n]]
    embed(imgs[:1])
    t0 = time.perf_counter()
    embed(imgs)
    ms = (time.perf_counter() - t0) * 1000.0 / max(len(imgs), 1)
    return {"encoder": encoder_name, "params": int(params), "ms_per_image": float(ms)}


# ---------------------------------------------------------------- classifiers

def _expand(grid):
    keys = sorted(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def candidate_id(encoder, family, params):
    return "{}__{}__{}".format(encoder, family, "_".join("{}={}".format(
        k, "x".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in sorted(params.items())))


class _Model:
    """Fitted classifier exposing ``logits(X) -> (n, 4)`` so one temperature fit works for all families."""

    def __init__(self, family, clf, head=None):
        self.family, self.clf, self.head = family, clf, head

    def logits(self, X):
        if self.head is not None:
            return self.head.logits(X)
        p = np.clip(self.clf.predict_proba(X), 1e-9, 1.0)
        full = np.full((len(X), N_CLASSES), np.log(1e-9))
        full[:, [int(c) for c in self.clf.classes_]] = np.log(p)
        return full


def fit_classifier(family, params, X, y, seed):
    """Fit one candidate (class-balanced where the family supports it)."""
    if family == "logistic":
        return _Model(family, None, spi._fit_lr(X, y, params["C"], seed))
    if family == "linear_svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=params["C"], class_weight="balanced", random_state=seed, max_iter=5000).fit(X, y)
        return _Model(family, clf, SPIHead(clf.coef_, clf.intercept_, clf.classes_))
    if family == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return _Model(family, KNeighborsClassifier(n_neighbors=min(params["k"], len(y)),
                                                   weights="distance").fit(X, y))
    if family == "mlp":
        from sklearn.neural_network import MLPClassifier
        return _Model(family, MLPClassifier(hidden_layer_sizes=tuple(params["hidden"]), alpha=params["alpha"],
                                            max_iter=300, random_state=seed).fit(X, y))
    if family == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return _Model(family, HistGradientBoostingClassifier(
            max_leaf_nodes=params["max_leaf_nodes"], learning_rate=params["learning_rate"],
            class_weight="balanced", random_state=seed, max_iter=100).fit(X, y))
    raise ValueError("unknown classifier family " + family)


# ---------------------------------------------------------------- grid

_USED = ("train", "val")


def _prepare(emb_pack, labels, splits):
    """Rows of one encoder restricted to train/val; test rows are dropped here and nowhere re-added."""
    paths, emb = emb_pack
    out = {s: ([], [], [], []) for s in _USED}
    for i, p in enumerate(map(str, paths)):
        sp = splits.get(p)
        if sp not in _USED or p not in labels:
            continue
        out[sp][0].append(i)
        out[sp][1].append(int(labels[p]["tier"]))
        out[sp][2].append(str(labels[p].get("source") or ""))
        out[sp][3].append(p)
    emb = np.asarray(emb, dtype=np.float32)
    data = {s: (emb[v[0]], np.array(v[1], int), np.array(v[2])) for s, v in out.items()}
    used_paths = set(out["train"][3]) | set(out["val"][3])
    assert not any(splits.get(p) == "test" for p in used_paths), "test rows leaked into fit/selection"
    return data


class GridDataError(ValueError):
    """The labelled rows cannot support a classifier grid (train has < 2 classes, or val is empty)."""


def _class_counts(y):
    vals, counts = np.unique(np.asarray(y, int), return_counts=True)
    return {TIERS[int(v)] if 0 <= int(v) < len(TIERS) else str(int(v)): int(c) for v, c in zip(vals, counts)}


def check_split_data(ytr, yva, enc="?"):
    """Raise GridDataError (with counts per split and class) when train has < 2 classes or val is empty."""
    if len(set(np.asarray(ytr).tolist())) < 2 or len(yva) == 0:
        raise GridDataError(
            "cannot fit a grid for encoder %s: need >= 2 classes in train and a non-empty val; "
            "train n=%d classes=%s, val n=%d classes=%s"
            % (enc, len(ytr), _class_counts(ytr), len(yva), _class_counts(yva)))


DEGENERATE_FILE = "grid_degenerate.json"


def run_grid(embeddings_by_encoder, labels, splits, grid_cfg, out_dir, seed=0):
    """Train/evaluate every candidate on train/val, saving one JSON each; resumable. Returns candidates."""
    out_dir = Path(out_dir)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cost = {e["name"]: e.get("ue_cost", 0) for e in grid_cfg["encoders"]}
    cands = []
    for enc, pack in embeddings_by_encoder.items():
        data = _prepare(pack, labels, splits)
        Xtr, ytr, _ = data["train"]
        Xva, yva, sva = data["val"]
        check_split_data(ytr, yva, enc)
        for family, hp in grid_cfg["classifiers"].items():
            for params in _expand(hp):
                cid = candidate_id(enc, family, params)
                f = cand_dir / (cid + ".json")
                if f.exists():
                    cands.append(json.loads(f.read_text(encoding="utf-8")))
                    continue
                model = fit_classifier(family, params, Xtr, ytr, seed)
                lg = model.logits(Xva)
                temp = spi.fit_temperature(lg, yva)
                probs = _softmax(lg / temp)
                rep = spi._report(probs, yva)
                rep["by_source"] = {s: spi._report(probs[sva == s], yva[sva == s])
                                    for s in sorted(set(sva.tolist()))}
                rec = {"id": cid, "encoder": enc, "family": family, "params": params, "seed": seed,
                       "ue_cost": cost.get(enc, 0), "temperature": temp, "val": rep,
                       "val_macro_f1": rep["macro_f1"], "critical_recall": rep["critical_recall"],
                       "ece": rep["ece"], "n_train": int(len(ytr)), "n_val": int(len(yva))}
                f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                cands.append(rec)
    return cands


def load_candidates(grid_dir):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((Path(grid_dir) / "candidates").glob("*.json"))]


def select_best(cands, selection_cfg):
    """Gate on Critical recall, take max val macro-F1, break near-ties by lower ue_cost then lower ECE."""
    gate = float(selection_cfg["min_critical_recall"])
    tol = float(selection_cfg.get("f1_tie_tol", 0.0))
    ok = [c for c in cands if c["critical_recall"] >= gate]
    if not ok:
        raise ValueError("no candidate meets the Critical-recall gate {}".format(gate))
    top = max(c["val_macro_f1"] for c in ok)
    tied = [c for c in ok if c["val_macro_f1"] >= top - tol]
    return min(tied, key=lambda c: (c["ue_cost"], c["ece"], -c["val_macro_f1"], c["id"]))


_CSV_COLS = ("id", "encoder", "family", "params", "ue_cost", "val_macro_f1", "critical_recall", "ece",
             "temperature", "passes_gate")


def write_outputs(cands, selection_cfg, out_dir, chosen=None):
    out_dir = Path(out_dir)
    chosen = chosen or select_best(cands, selection_cfg)
    gate = float(selection_cfg["min_critical_recall"])
    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_COLS)
        for c in sorted(cands, key=lambda c: -c["val_macro_f1"]):
            w.writerow([c["id"], c["encoder"], c["family"], json.dumps(c["params"]), c["ue_cost"],
                        c["val_macro_f1"], c["critical_recall"], c["ece"], c["temperature"],
                        c["critical_recall"] >= gate])

    def slim(c):
        return {k: v for k, v in c.items() if k != "val"}
    (out_dir / "chosen_spi.json").write_text(json.dumps(slim(chosen), indent=2), encoding="utf-8")
    (out_dir / "grid_report.json").write_text(json.dumps(
        {"selection": selection_cfg, "chosen": chosen["id"], "n_candidates": len(cands),
         "candidates": [slim(c) for c in cands], "chosen_val": chosen["val"]}, indent=2), encoding="utf-8")
    return chosen


def fit_chosen(embeddings_by_encoder, labels, splits, chosen, out_dir, seed=0):
    """Refit the winner on train, temperature-scale on val, save the head. Returns the fitted model."""
    out_dir = Path(out_dir)
    data = _prepare(embeddings_by_encoder[chosen["encoder"]], labels, splits)
    model = fit_classifier(chosen["family"], chosen["params"], data["train"][0], data["train"][1], seed)
    temp = spi.fit_temperature(model.logits(data["val"][0]), data["val"][1])
    if model.head is not None:
        model.head.temperature = temp
        model.head.save(out_dir / "chosen_head")
    else:
        import pickle
        (out_dir / "chosen_head.pkl").write_bytes(pickle.dumps({"clf": model.clf, "temperature": temp}))
    model.temperature = temp
    return model


def grid_and_select(embeddings_by_encoder, labels, splits, grid_cfg, out_dir, seed=0):
    """run_grid + select_best + save winner head and reports. Returns (chosen, model)."""
    out_dir = Path(out_dir)
    cands = run_grid(embeddings_by_encoder, labels, splits, grid_cfg, out_dir, seed)
    chosen = write_outputs(cands, grid_cfg["selection"], out_dir)
    return chosen, fit_chosen(embeddings_by_encoder, labels, splits, chosen, out_dir, seed)


def final_eval(model, emb_pack, labels, splits, confirm_test=False):
    """Evaluate a fitted winner on the TEST split. Refuses unless ``confirm_test=True``."""
    if not confirm_test:
        raise PermissionError("final_eval touches the test split; pass confirm_test=True explicitly")
    paths, emb = emb_pack
    idx = [i for i, p in enumerate(map(str, paths)) if splits.get(str(p)) == "test" and str(p) in labels]
    ps = [str(paths[i]) for i in idx]
    y = np.array([int(labels[p]["tier"]) for p in ps], int)
    src = np.array([str(labels[p].get("source") or "") for p in ps])
    probs = _softmax(model.logits(np.asarray(emb, np.float32)[idx]) / getattr(model, "temperature", 1.0))
    rep = spi._report(probs, y)
    rep["by_source"] = {s: spi._report(probs[src == s], y[src == s]) for s in sorted(set(src.tolist()))}
    return rep


# ---------------------------------------------------------------- CLI

def _read_labels(path):
    tier_idx = {t: i for i, t in enumerate(TIERS)}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            t = r["tier"]
            out[r["image_path"]] = {"tier": tier_idx[t] if isinstance(t, str) else int(t),
                                    "source": r.get("source")}
    return out


def _read_splits(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--encoder", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--splits", required=True)
    e.add_argument("--images-root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--shard-size", type=int, default=None)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--device", default="cpu")
    g = sub.add_parser("grid")
    g.add_argument("--embeddings-dir", required=True)
    g.add_argument("--labels", required=True)
    g.add_argument("--splits", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--allow-degenerate", action="store_true",
                   help="pilot: a GridDataError is recorded in grid_degenerate.json and exits 0 (plumbing only)")
    s = sub.add_parser("select")
    s.add_argument("--grid-dir", required=True)
    s.add_argument("--allow-degenerate", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config("grid")
    if a.cmd == "extract":
        from PIL import Image
        labels, splits = _read_labels(a.labels), _read_splits(a.splits)
        items = sorted(p for p in labels if splits.get(p) in _USED)   # test rows are not extracted here
        lim = cfg["limits"]
        root = Path(a.images_root)
        extract_encoder(a.encoder, items, lambda p: Image.open(root / p), a.batch_size, a.device, a.out,
                        shard_size=a.shard_size or lim["shard_size"], min_free_gb=lim["min_free_gb"],
                        max_images=lim["max_images_per_encoder"])
    elif a.cmd == "grid":
        labels, splits = _read_labels(a.labels), _read_splits(a.splits)
        embs = {}
        for enc in cfg["encoders"]:
            f = Path(a.embeddings_dir) / ("emb_%s.npz" % enc["name"])
            if f.exists() and f.with_suffix(".done").exists():
                z = np.load(f, allow_pickle=False)
                embs[enc["name"]] = (z["image_path"], z["emb"])
        if not embs:
            raise FileNotFoundError("no finished embeddings in " + a.embeddings_dir)
        (Path(a.out) / DEGENERATE_FILE).unlink(missing_ok=True)
        try:
            chosen, _ = grid_and_select(embs, labels, splits, cfg, a.out, a.seed)
        except GridDataError as error:
            if not a.allow_degenerate:
                raise
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / DEGENERATE_FILE).write_text(json.dumps({"message": str(error)}), encoding="utf-8")
            print("GridDataError (allowed, plumbing only): %s" % error)
            return 0
        print(json.dumps({k: v for k, v in chosen.items() if k != "val"}, indent=2))
    else:
        if a.allow_degenerate and (Path(a.grid_dir) / DEGENERATE_FILE).exists():
            print("select skipped: degenerate grid data (see %s)" % DEGENERATE_FILE)
            return 0
        chosen = write_outputs(load_candidates(a.grid_dir), cfg["selection"], a.grid_dir)
        print(chosen["id"])


if __name__ == "__main__":
    main()
