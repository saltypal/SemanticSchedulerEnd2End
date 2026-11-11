"""Kaggle entry point: Qwen2.5-VL AI feedback on TRAIN images (plus a val sample for the S3 kappa audit only).

``python -m Scheduler.training.edge.qwen_feedback --mode pilot|full`` discovers dataset A (index.jsonl) and
B (labels_raw.jsonl, splits_v1.json) under /kaggle/input, writes ``plan.json`` (the frozen sample), launches one
worker per GPU (each loads Qwen once, 4-bit) and merges the shards. Everything is resumable: re-running keeps
``plan.json`` and skips finished keys (``rlaif.done_ids``). Heavy libraries are imported only inside workers.
Output dir: <out>/{plan.json, point_val.jsonl, point_train.jsonl, pair_train.jsonl, feedback_manifest.json}
plus per-worker shard files ``*.<gpu>.jsonl``.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from Codec.edge.sqgan_finetune import discover_root
from Evaluation.edge.contracts import TIERS
from Scheduler.training.edge import rlaif

MODELS = ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct")
PILOT = {"val": 0, "point": 20, "pair": 10}
PAIR_DIFF_WEIGHTS = {0: 0.1, 1: 0.3, 2: 0.3, 3: 0.3}
MAX_SIDE = 448   # images are shrunk before the VLM: bounds visual tokens, speed


# ------------------------------------------------------------------ data + sampling (pure, tested)
def discover_inputs(search_root="/kaggle/input"):
    return discover_root(search_root, "index.jsonl"), discover_root(search_root, "labels_raw.jsonl")


def load_rows(labels_root, splits_name="splits_v1.json"):
    text = (Path(labels_root) / "labels_raw.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    splits = json.loads((Path(labels_root) / splits_name).read_text(encoding="utf-8"))["splits"]
    for r in rows:
        r["split"] = splits.get(r["uid"])
    return rows


def stratified_sample(rows, n, seed, tiers=TIERS):
    """n rows balanced over tiers (equal share, spill-over to larger tiers), proportional over sources in a tier."""
    rng = random.Random(seed)
    by_tier = {t: defaultdict(list) for t in tiers}
    for r in sorted(rows, key=lambda r: r["uid"]):
        by_tier[r["tier"]][r["source"]].append(r)
    for d in by_tier.values():
        for lst in d.values():
            rng.shuffle(lst)
    sizes = {t: sum(map(len, by_tier[t].values())) for t in tiers}
    quota, left, open_t = {t: 0 for t in tiers}, min(n, sum(sizes.values())), [t for t in tiers if sizes[t]]
    while left and open_t:
        share = max(left // len(open_t), 1)
        for t in list(open_t):
            take = min(share, sizes[t] - quota[t], left)
            quota[t] += take
            left -= take
            if quota[t] >= sizes[t]:
                open_t.remove(t)
            if not left:
                break
    out = []
    for t in tiers:
        srcs = by_tier[t]
        total = sum(map(len, srcs.values())) or 1
        picked = []
        for s in sorted(srcs):
            k = min(len(srcs[s]), max(1 if quota[t] else 0, round(quota[t] * len(srcs[s]) / total)))
            picked += srcs[s][:k]
        chosen = {r["uid"] for r in picked}
        rest = [r for s in sorted(srcs) for r in srcs[s] if r["uid"] not in chosen]
        picked = picked[:quota[t]] + rest[:max(0, quota[t] - len(picked))]
        out += picked
    return out


def sample_pairs(rows, n, seed, tiers=TIERS, weights=PAIR_DIFF_WEIGHTS):
    """n distinct unordered pairs of rows, spread over tier differences by ``weights``; A/B order randomised."""
    rng = random.Random(seed)
    idx = {t: sorted((r for r in rows if r["tier"] == t), key=lambda r: r["uid"]) for t in tiers}
    rank = {t: i for i, t in enumerate(tiers)}
    by_diff = {d: [(a, b) for a in tiers for b in tiers if rank[b] - rank[a] == d and idx[a] and idx[b]] for d in weights}
    diffs = [d for d in sorted(weights) if by_diff[d]]
    seen, out, tries = set(), [], 0
    while diffs and len(out) < n and tries < 50 * n + 100:
        tries += 1
        d = rng.choices(diffs, [weights[x] for x in diffs])[0]
        ta, tb = rng.choice(by_diff[d])
        a, b = rng.choice(idx[ta]), rng.choice(idx[tb])
        key = frozenset((a["uid"], b["uid"]))
        if a["uid"] == b["uid"] or key in seen:
            continue
        seen.add(key)
        out.append((a, b) if rng.random() < 0.5 else (b, a))
    return out


def size_plan(budget_h, sec_point, sec_pair, n_gpus=2, n_val=500, point_share=0.5):
    """Item counts that fit ``budget_h`` hours: val first, remaining time split point_share / rest."""
    total = budget_h * 3600.0 * n_gpus
    rest = max(total - n_val * sec_point, 0.0)
    return {"val": n_val, "point": int(rest * point_share / sec_point), "pair": int(rest * (1 - point_share) / sec_pair)}


def build_plan(rows, sizes, seed=0, tiers=TIERS):
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    pt = stratified_sample(train, sizes["point"], seed, tiers)
    pairs = sample_pairs(train, sizes["pair"], seed + 1, tiers)
    vs = stratified_sample(val, sizes["val"], seed + 2, tiers) if sizes["val"] else []
    slim = lambda r: {k: r.get(k) for k in ("uid", "file", "tier", "source", "event")}
    return {"seed": seed, "sizes": sizes, "available": {"train": len(train), "val": len(val)},
            "val": [slim(r) for r in vs], "point": [slim(r) for r in pt],
            "pair": [[slim(a), slim(b)] for a, b in pairs]}


def shard_of(seq, i, n):
    return [x for k, x in enumerate(seq) if k % n == i]


def summarize(groups):
    """Parse rate / order-consistency per group name -> records."""
    out = {}
    for name, recs in groups.items():
        out[name] = {"n": len(recs)}
        if name.startswith("pair") and recs:
            out[name]["consistent_rate"] = sum(1 for r in recs if r.get("consistent")) / len(recs)
        if name.startswith("point") and recs:
            ok = [r for r in recs if "label" in r and len(r.get("logp", [])) == 4]
            out[name]["parse_rate"] = len(ok) / len(recs)
    return out


def read_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def write_manifest(out_dir, info):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "feedback_manifest.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------------ worker (GPU only)
def _load_image(root, rel):
    from PIL import Image
    im = Image.open(Path(root) / rel).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def _make_scorer(model_id):
    for mid in [model_id] + [m for m in MODELS if m != model_id]:
        try:
            return rlaif.QwenScorer(mid, load_4bit=True), mid
        except Exception as e:                       # most likely out of memory: fall back to the smaller model
            print("model %s failed to load: %r" % (mid, e), flush=True)
    raise RuntimeError("no Qwen model could be loaded")


def _chunks(it, k):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == k:
            yield buf
            buf = []
    if buf:
        yield buf


def worker(a):
    plan = json.loads((Path(a.out) / "plan.json").read_text(encoding="utf-8"))
    i, n = (int(x) for x in a.shard.split("/"))
    scorer, mid = _make_scorer(a.model_id)
    rev = getattr(getattr(scorer.model, "config", None), "_commit_hash", None)
    print("[w%d] model %s rev %s" % (i, mid, rev), flush=True)
    stats = {"model_id": mid, "revision": rev, "shard": a.shard}
    for name, mode, items in (("point_val", "pointwise", plan["val"]), ("point_train", "pointwise", plan["point"]),
                              ("pair_train", "pairwise", plan["pair"])):
        path = str(Path(a.out) / ("%s.%d.jsonl" % (name, i)))
        done = rlaif.done_ids(path)
        key = (lambda it: it["uid"]) if mode == "pointwise" else (lambda it: "%s|%s" % (it[0]["uid"], it[1]["uid"]))
        todo = [it for it in shard_of(items, i, n) if key(it) not in done]
        if mode == "pointwise":
            gen = ((r["uid"], _load_image(a.images_root, r["file"])) for r in todo)
        else:
            gen = ((x["uid"], _load_image(a.images_root, x["file"]), y["uid"], _load_image(a.images_root, y["file"]))
                   for x, y in todo)
        t0, last, cnt = time.time(), time.time(), 0
        for chunk in _chunks(gen, 10):
            cnt += rlaif.collect_to_jsonl(chunk, scorer, path, mode)
            if time.time() - last > 60:
                print("[w%d] %s %d/%d  %.2fs/item" % (i, name, cnt, len(todo), (time.time() - t0) / max(cnt, 1)), flush=True)
                last = time.time()
        stats[name] = {"done_now": cnt, "todo": len(todo), "sec_per_item": (time.time() - t0) / cnt if cnt else None}
        print("[w%d] %s finished: %s" % (i, name, stats[name]), flush=True)
    (Path(a.out) / ("worker_%d.json" % i)).write_text(json.dumps(stats, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ orchestrator
def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def orchestrate(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.images_root and a.labels_root:
        images_root, labels_root = Path(a.images_root), Path(a.labels_root)
    else:
        images_root, labels_root = discover_inputs(a.search_root)
    print("images root", images_root, "| labels root", labels_root, flush=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        print("reusing existing plan.json (resume)", flush=True)
    else:
        sizes = dict(PILOT) if a.mode == "pilot" else size_plan(a.budget_hours, a.sec_point, a.sec_pair, a.gpus, a.val_n)
        plan_path.write_text(json.dumps(build_plan(load_rows(labels_root), sizes, a.seed)), encoding="utf-8")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    print("plan sizes", plan["sizes"], "available", plan["available"], flush=True)
    t0, procs = time.time(), []
    for g in range(a.gpus):
        cmd = [sys.executable, "-u", "-m", "Scheduler.training.edge.qwen_feedback", "worker", "--out", str(out),
               "--images-root", str(images_root), "--shard", "%d/%d" % (g, a.gpus), "--model-id", a.model_id]
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
    rcs = [p.wait() for p in procs]
    shards = {}
    for name in ("point_val", "point_train", "pair_train"):
        recs = [r for g in range(a.gpus) for r in read_jsonl(out / ("%s.%d.jsonl" % (name, g)))]
        shards[name] = recs
        with open(out / (name + ".jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    workers = [json.loads((out / ("worker_%d.json" % g)).read_text()) for g in range(a.gpus)
               if (out / ("worker_%d.json" % g)).exists()]
    summary = summarize(shards)
    true_tier = {r["uid"]: r["tier"] for r in plan["point"] + plan["val"]}
    examples = {}
    for name in ("point_val", "point_train"):
        ok = [(TIERS[r["label"]], true_tier.get(r["key"])) for r in shards[name] if "label" in r]
        if ok:
            summary[name]["exact_match_true_tier"] = sum(p == t for p, t in ok) / len(ok)
            examples[name] = [{"qwen": p, "true": t} for p, t in ok[:8]]
    info = {"mode": a.mode, "git_sha": _git_sha(), "return_codes": rcs, "wall_time_s": time.time() - t0,
            "sizes": plan["sizes"], "available": plan["available"], "workers": workers, "summary": summary,
            "examples": examples, "model_id": workers[0]["model_id"] if workers else None,
            "revision": workers[0]["revision"] if workers else None, "args": vars(a)}
    print(write_manifest(out, info), json.dumps(summary), flush=True)
    if any(rcs):
        raise SystemExit("worker failure: %s" % rcs)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qwen_feedback")
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--out", default="/kaggle/working/runs/edge-qwen-feedback")
    ap.add_argument("--search-root", default="/kaggle/input")
    ap.add_argument("--images-root", default="")
    ap.add_argument("--labels-root", default="")
    ap.add_argument("--model-id", default=MODELS[0])
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--sec-point", type=float, default=1.5, help="measured seconds per pointwise item (from the pilot)")
    ap.add_argument("--sec-pair", type=float, default=4.0, help="measured seconds per pair, both orders (pilot)")
    ap.add_argument("--val-n", type=int, default=500)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "worker":
        ap.add_argument("worker")
        ap.add_argument("--shard", required=True)
    a = ap.parse_args(argv)
    worker(a) if getattr(a, "worker", None) else orchestrate(a)


if __name__ == "__main__":
    main()
