"""Assemble the Kaggle dataset ``edge-labels-v1`` (plan v3.1 S1): all labels, no images.

Contents: labels_raw.jsonl (CrisisMMD human-derived tiers + everyday images as Low), counts.json,
normals_places365.jsonl, normals_people.jsonl, the pinned CrisisMMD metadata JSON files,
manifest.json (sha256 per file, repo commit) and dataset-metadata.json. Run locally or on Kaggle.

    python -m parameters.edge.build_label_dataset --meta-dir <9 CrisisMMD JSON files> --out <dir> --owner <kaggle user>
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from Evaluation.edge.edge_config import load_config, repo_root
from parameters.edge import labels, normals


def make_fetcher(cache_dir):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(url):
        target = cache / hashlib.sha256(url.encode()).hexdigest()
        if not target.exists():
            with urllib.request.urlopen(url, timeout=300) as response:
                target.write_bytes(response.read())
        return target.read_bytes()
    return fetch


def _jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


README = (
    "# Edge scheduler labels v1\n\nLabels only, no images. Tiers (Critical/High/Medium/Low) derive from "
    "CrisisMMD human annotations (damage severity, humanitarian category, informativeness) with a "
    "life-safety override; everyday images (Places365 validation scenes, Open Images selfies/people) are "
    "Low. Rules: config/edge/data.yaml in the source repo. Sources keep their own licences: CrisisMMD "
    "(research use, QCRI), Places365 (MIT CSAIL), Open Images (per-image licence in `labels_raw.jsonl`).\n")


def build(meta_dir, out_dir, owner, fetch, slug="edge-labels-v1", cfg=None):
    cfg = cfg or load_config("data")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crisis = labels.label_table(labels.load_crisismmd(meta_dir, cfg["crisismmd"]), cfg=cfg)
    places, people, nrep = normals.build_normals(fetch, cfg["normals"])
    p_rows, _ = labels.normal_rows(places, "places365", cfg["normals"])
    o_rows, _ = labels.normal_rows(people, "openimages_people", cfg["normals"])
    for rows, items in ((p_rows, places), (o_rows, people)):
        for row, item in zip(rows, items):
            row["image_id"] = item["id"]
            if "url" in item:
                row.update(url=item["url"], license=item["license"], author=item["author"])
    everything = crisis + p_rows + o_rows
    _jsonl(out / "labels_raw.jsonl", everything)
    _jsonl(out / "normals_places365.jsonl", p_rows)
    _jsonl(out / "normals_people.jsonl", o_rows)
    counts = labels.report_counts(everything, cfg)
    by_source = {}
    for r in everything:
        tiers = by_source.setdefault(r["source"], {})
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    counts["totals_by_source_tier"] = by_source
    counts["normals"] = nrep
    (out / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    meta_out = out / "crisismmd_metadata"
    meta_out.mkdir(exist_ok=True)
    for f in sorted(Path(meta_dir).glob("*.json")):
        shutil.copy2(f, meta_out / f.name)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "id": "%s/%s" % (owner, slug), "title": "Edge scheduler labels v1",
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"repo_commit": _git_sha(),
                "files": {p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                          for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--cache", default=None, help="download cache (default: next to --out)")
    ap.add_argument("--slug", default="edge-labels-v1")
    a = ap.parse_args(argv)
    cache = a.cache or str(Path(a.out).resolve().parent / "_edge_labels_cache")
    counts = build(a.meta_dir, a.out, a.owner, make_fetcher(cache), a.slug)
    print(json.dumps(counts["gates"]), json.dumps(counts["normals"]))


if __name__ == "__main__":
    main()
