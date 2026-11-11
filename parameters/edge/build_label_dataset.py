"""Stage B: labels for the images of dataset A (``scheduler-dataset-v1-labels``); no images are copied.

Reads A's ``index.jsonl``; labels CrisisMMD rows from the 9 pinned metadata JSONs (only images present
in A); takes every Places365 / COCO-people row of A (A already holds the configured counts) that exists on disk.
Outputs labels_raw.jsonl (rows carry ``uid``/``file``), counts.json (gate verdicts), normals_*.jsonl,
crisismmd_metadata/, failures.json, manifest.json, README.md, dataset-metadata.json; then publishes.

    python -m parameters.edge.build_label_dataset --meta-dir <dir> --out /kaggle/working/dataset-b [--images-root <A>]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from Evaluation.edge.edge_config import load_config
from parameters.edge import kaggle_publish, labels
from parameters.edge.dataset_a import write_jsonl, write_manifest, write_metadata

SLUG = "scheduler-dataset-v1-labels"
TITLE = "Scheduler dataset v1: labels (B)"
OWNER = "satyapaladuguplus"
README = ("# Scheduler dataset v1: labels (B)\n\nLabels for the images in scheduler-dataset-v1-images (A); no images here. "
          "`file` is relative to A's root. Tiers (Critical/High/Medium/Low) derive from CrisisMMD human "
          "annotations (damage severity, humanitarian category, informativeness) with a life-safety override; "
          "everyday images (Places365 scenes, COCO people) are Low. Rules: config/edge/data.yaml. "
          "Sources keep their own licences (CrisisMMD research use QCRI, Places365 CC0, COCO 2017 CC BY 4.0).\n")
NORMAL_SOURCES = ("places365", "coco_people")


def discover_images_root(base="/kaggle/input", max_depth=4):
    """Directory under ``base`` holding A's index.jsonl."""
    base = Path(base)
    for dirpath, dirs, files in os.walk(base):
        if "index.jsonl" in files:
            return Path(dirpath)
        if len(Path(dirpath).relative_to(base).parts) >= max_depth:
            dirs[:] = []
    raise FileNotFoundError("no dataset A (index.jsonl) found under %s; pass --images-root" % base)


def read_index(root):
    return [json.loads(x) for x in (Path(root) / "index.jsonl").read_text(encoding="utf-8").splitlines() if x]


def _normal_rows(index, source, cfg, root, failures):
    """Every A row of ``source`` whose file exists -> Low label rows (blacklist applies to places365)."""
    rows = []
    for a in sorted((r for r in index if r["source"] == source), key=lambda r: r["rank"]):
        if not (Path(root) / a["file"]).is_file():
            failures.append({"source": source, "key": a["file"], "uid": a["uid"],
                             "reason": "listed in A index but file missing"})
            continue
        item = {"id": a.get("image_id"), "path": a["file"], "category": a.get("category")}
        got, _ = labels.normal_rows([item], source, cfg["normals"])
        for row in got:
            row["uid"], row["file"] = a["uid"], a["file"]
            rows.append(row)
    return rows


def build_b(meta_dir, out_dir, images_root, cfg=None):
    cfg = cfg or load_config("data")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(images_root)
    by_uid = {r["uid"]: r for r in index if (Path(images_root) / r["file"]).is_file()}
    failures, crisis = [], []
    for row in labels.label_table(labels.load_crisismmd(meta_dir, cfg["crisismmd"]), cfg=cfg):
        uid = "crisismmd:" + row["image_path"]
        if uid in by_uid:
            row["uid"], row["file"] = uid, by_uid[uid]["file"]
            crisis.append(row)
        else:
            failures.append({"source": "crisismmd", "key": row["image_path"], "uid": uid, "reason": "missing in A"})
    places = _normal_rows(index, "places365", cfg, images_root, failures)
    people = _normal_rows(index, "coco_people", cfg, images_root, failures)
    everything = crisis + places + people
    uids = [r["uid"] for r in everything]
    if len(set(uids)) != len(uids):
        raise ValueError("uid collision in labels_raw")
    write_jsonl(out / "labels_raw.jsonl", everything)
    write_jsonl(out / "normals_places365.jsonl", places)
    write_jsonl(out / "normals_coco_people.jsonl", people)
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    counts = labels.report_counts(everything, cfg)
    by_source = {}
    for r in everything:
        tiers = by_source.setdefault(r["source"], {})
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    counts["totals_by_source_tier"] = by_source
    counts["normals"] = {"places365_final": len(places), "coco_people_final": len(people),
                         "missing_in_a": len(failures), "a_index_rows": len(index)}
    (out / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    meta_out = out / "crisismmd_metadata"
    meta_out.mkdir(exist_ok=True)
    for task in cfg["crisismmd"]["tasks"]:
        for split in cfg["crisismmd"]["native_splits"]:
            src = Path(meta_dir) / ("%s_%s.json" % (task, split))
            if src.is_file():
                shutil.copy2(src, meta_out / src.name)
    (out / "README.md").write_text(README, encoding="utf-8")
    write_metadata(out, SLUG, TITLE)
    write_manifest(out)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-root", default=None, help="dataset A root (default: discover under /kaggle/input)")
    ap.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args(argv)
    root = a.images_root or discover_images_root()
    counts = build_b(a.meta_dir, a.out, root)
    print(json.dumps(counts["gates"]), json.dumps(counts["normals"]))
    if a.publish:
        print(json.dumps(kaggle_publish.publish_dataset(a.out, SLUG, TITLE, "stage B labels")))


if __name__ == "__main__":
    main()
