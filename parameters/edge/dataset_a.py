"""Stage A: download ALL images into the Kaggle dataset ``scheduler-dataset-v1-images`` (no labels, no tiers).

Sources: every CrisisMMD v2.0 image named in the 9 pinned metadata JSONs (copied from the attached Kaggle
dataset, HF fallback), a small stratified Places365 sample (Kaggle ``benjaminkz/places365``, val_256 tar
only as fallback) and COCO people (Kaggle ``awsaf49/coco-2017-dataset``). Kaggle inputs are mounted read-only
and only the chosen files are copied. Resumable, free-disk guard, ``--max-images`` dry run.

    python -m parameters.edge.dataset_a --meta-dir <dir> --out /kaggle/working/dataset-a --workers 16
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from Evaluation.edge.edge_config import load_config, repo_root
from parameters.edge import fetch_images, kaggle_publish, labels, normals

SLUG = "scheduler-dataset-v1-images"
TITLE = "Scheduler dataset v1: images (A)"
OWNER = "satyapaladuguplus"
README = ("# Scheduler dataset v1: images (A)\n\nRaw images only, no labels. `index.jsonl` maps `uid` to `file` "
          "(`images/crisismmd|places365|coco_people/...`) with bytes, size and sha256; normals carry `category` and `rank` "
          "(deterministic order). Labels are in scheduler-dataset-v1-labels (B). "
          "Licences: CrisisMMD (research use, QCRI), Places365 (CC0, Kaggle benjaminkz/places365), COCO 2017 (CC BY 4.0, Kaggle awsaf49/coco-2017-dataset).\n")


MAX_COPY_FAIL_FRACTION = 0.02   # above this, missing/corrupt CrisisMMD copies are fetched from HF


class DiskFull(RuntimeError):
    pass


def make_fetcher(cache_dir):
    """``fetch(url) -> bytes`` with an on-disk cache (metadata lists only, never images)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(url):
        target = cache / hashlib.sha256(url.encode()).hexdigest()
        if not target.exists():
            with urllib.request.urlopen(url, timeout=300) as response:
                target.write_bytes(response.read())
        return target.read_bytes()
    return fetch


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_manifest(out, known=None):
    """manifest.json: bytes+sha256 per file (``known`` = {rel: {bytes, sha256}} avoids re-hashing)."""
    known = known or {}
    files = {}
    for p in sorted(Path(out).rglob("*")):
        if not p.is_file() or p.name in ("manifest.json",) or p.name.endswith(".part"):
            continue
        rel = p.relative_to(out).as_posix()
        k = known.get(rel)
        files[rel] = k if k and k["bytes"] == p.stat().st_size else {"bytes": p.stat().st_size,
                                                                     "sha256": sha256_file(p)}
    manifest = {"repo_commit": git_sha(), "files": files}
    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_metadata(out, slug, title, owner=OWNER):
    (Path(out) / "dataset-metadata.json").write_text(json.dumps({
        "id": "%s/%s" % (owner, slug), "title": title, "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8")


def check_disk(path, min_free_gb, usage=shutil.disk_usage):
    free = usage(str(path)).free / 1e9
    if free < min_free_gb:
        raise DiskFull("only %.2f GB free (< %.2f GB guard); stopping so what exists can still be published"
                       % (free, min_free_gb))


def candidates(fetch, cfg, max_images=None, input_root="/kaggle/input"):
    """Ranked normals: (places items, people items, report). See parameters.edge.normals.build_normals."""
    return normals.build_normals(fetch, cfg["normals"], input_root, max_images)


def _info(root, rel, prior):
    path = Path(root) / rel
    if not path.is_file() or path.stat().st_size == 0:
        return None
    old = prior.get(rel)
    if old and old.get("bytes") == path.stat().st_size:
        return {k: old[k] for k in ("bytes", "width", "height", "sha256")}
    data = path.read_bytes()
    try:
        _, (w, h) = fetch_images.check_image(data)
    except ValueError:
        return None
    return {"bytes": len(data), "width": w, "height": h, "sha256": hashlib.sha256(data).hexdigest()}


def _load_prior(out):
    path = Path(out) / "index.jsonl"
    if not path.is_file():
        return {}
    return {r["file"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x)}


def _index_rows(out, items, prior):
    rows, missing = [], []
    for base in items:
        info = _info(out, base["file"], prior)
        if info is None:
            missing.append(base)
        else:
            rows.append(dict(base, **info))
    return rows, missing


def build_a(meta_dir, out_dir, fetch, cfg=None, opener=None, tar_source=None, workers=8, max_images=None,
            min_free_gb=2.0, progress_dir=None, disk_usage=shutil.disk_usage, chunk=1000,
            crisismmd_root=None, input_root="/kaggle/input"):
    """Download everything, write index.jsonl/failures.json/manifest.json/README/metadata. Returns manifest."""
    cfg = cfg or load_config("data")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prog = Path(progress_dir) if progress_dir else out.resolve().parent
    opener = opener or fetch_images.default_opener
    tar_source = tar_source or (lambda dest: fetch_images.download_with_resume(dest))
    records = labels.load_crisismmd(meta_dir, cfg["crisismmd"])
    crisis = [{"image_path": k, "event": records[k]["event"], "tweet_id": records[k]["tweet_id"]}
              for k in sorted(records)]
    if max_images:
        crisis = crisis[:max_images]
    places, people, nrep = candidates(fetch, cfg, max_images, input_root)
    kaggle_places = str(nrep["places365_mode"]).startswith("kaggle:")
    failures, stopped = [], None
    crisismmd_source = "huggingface"
    try:
        if crisismmd_root:
            check_disk(out, min_free_gb, disk_usage)
            copy_fails, _ = fetch_images.copy_crisismmd_from_root(crisis, crisismmd_root, out, workers)
            crisismmd_source = "kaggle_input:%s" % crisismmd_root
            if crisis and len(copy_fails) / len(crisis) > MAX_COPY_FAIL_FRACTION:
                bad = {f["key"] for f in copy_fails}
                for key in bad:    # drop partial/corrupt leftovers so the downloader refetches
                    (out / "images" / "crisismmd" / key).unlink(missing_ok=True)
                copy_fails = fetch_images.fetch_crisismmd([c for c in crisis if c["image_path"] in bad], out,
                                                          opener, workers,
                                                          progress_path=prog / "_progress_a_crisismmd.json")
                crisismmd_source += "+huggingface_fallback"
            failures += copy_fails
        else:
            for i in range(0, len(crisis), chunk):
                check_disk(out, min_free_gb, disk_usage)
                failures += fetch_images.fetch_crisismmd(crisis[i:i + chunk], out, opener, workers,
                                                         progress_path=prog / "_progress_a_crisismmd.json")
        check_disk(out, min_free_gb, disk_usage)
        if kaggle_places:
            failures += fetch_images.copy_verified(
                [(p["src"], out / "images" / "places365" / p["id"], p["id"]) for p in places], workers, "places365")
        else:
            failures += fetch_images.fetch_places365([{"id": p["id"]} for p in places], out, tar_source)
        check_disk(out, min_free_gb, disk_usage)
        failures += fetch_images.copy_verified(
            [(p["src"], out / "images" / "coco_people" / p["id"], p["id"]) for p in people], workers, "coco_people")
    except DiskFull as exc:
        stopped = str(exc)
    prior = _load_prior(out)
    cm = [{"uid": "crisismmd:" + c["image_path"], "file": "images/crisismmd/" + c["image_path"],
           "source": "crisismmd", "image_path": c["image_path"], "event": c["event"], "tweet_id": c["tweet_id"]}
          for c in crisis]
    pl = [{"uid": "places365:" + p["id"], "file": "images/places365/" + p["id"], "source": "places365",
           "image_id": p["id"], "category": p["category"], "rank": r} for r, p in enumerate(places)]
    co = [{"uid": "coco_people:" + p["id"], "file": "images/coco_people/" + p["id"], "source": "coco_people",
           "image_id": p["id"], "category": p["category"], "rank": r} for r, p in enumerate(people)]
    rows, missing = [], []
    for group in (cm, pl, co):
        r, m = _index_rows(out, group, prior)
        rows += r
        missing += m
    seen = {f.get("key") for f in failures}
    for m in missing:
        key = m.get("image_path") or m.get("image_id")
        if key not in seen:
            failures.append({"source": m["source"], "key": key, "reason": stopped or "not downloaded"})
    write_jsonl(out / "index.jsonl", rows)
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    nrep.update(crisismmd_source=crisismmd_source, index_rows=len(rows),failures=len(failures), stopped=stopped,
                by_source={s: sum(1 for r in rows if r["source"] == s) for s in ("crisismmd", "places365", "coco_people")})
    (out / "README.md").write_text(README, encoding="utf-8")
    write_metadata(out, SLUG, TITLE)
    manifest = write_manifest(out, {r["file"]: {"bytes": r["bytes"], "sha256": r["sha256"]} for r in rows})
    manifest["report"] = nrep
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def resolve_crisismmd_root(arg):
    """None/'' -> auto-discover; 'none' -> force HF download (returns None); else the given path."""
    if arg is None or arg == "":
        return fetch_images.find_crisismmd_root()
    if str(arg).lower() == "none":
        return None
    return str(arg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=None, help="dry run: cap images per source")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--cache", default=None, help="metadata download cache (default: next to --out)")
    ap.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crisismmd-root", default=None,
                    help="folder holding data_image/ (default: auto-discover under /kaggle/input; "
                         "'none' forces the Hugging Face download)")
    a = ap.parse_args(argv)
    cache = a.cache or str(Path(a.out).resolve().parent / "_edge_a_cache")
    root = resolve_crisismmd_root(a.crisismmd_root)
    manifest = build_a(a.meta_dir, a.out, make_fetcher(cache), workers=a.workers, max_images=a.max_images,
                       min_free_gb=a.min_free_gb, crisismmd_root=root)
    print(json.dumps(manifest["report"]))
    if a.publish:
        print(json.dumps(kaggle_publish.publish_dataset(a.out, SLUG, TITLE, "stage A images")))


if __name__ == "__main__":
    main()
