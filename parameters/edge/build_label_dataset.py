"""Stage B: labels for the images of dataset A (``scheduler-dataset-v1-labels``); no images are copied.

Reads A's ``index.jsonl``; labels CrisisMMD rows from the 9 pinned metadata JSONs (only images present
in A); takes the first ``places365_count`` / ``people_count`` normals by ``rank`` that exist in A.
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
          "everyday images (Places365 scenes, Open Images selfies/people) are Low. Rules: config/edge/data.yaml. "
          "Sources keep their own licences (CrisisMMD research use QCRI, Places365 MIT CSAIL, Open Images per-image).\n")
NORMAL_PATH = {"places365": lambda r: "val_256/" + r["image_id"],
               "openimages_people": lambda r: "openimages/validation/%s.jpg" % r["image_id"]}


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


def _normal_rows(index, source, count, cfg, root):
    """First ``count`` ranked A rows of ``source`` whose file exists -> label rows."""
    ranked = sorted((r for r in index if r["source"] == source and (Path(root) / r["file"]).is_file()),
                    key=lambda r: r["rank"])[:count]
    items = [{"id": r["image_id"], "path": NORMAL_PATH[source](r), "category": r.get("category")} for r in ranked]
    rows, _ = labels.normal_rows(items, source, cfg["normals"])
    for row, a in zip(rows, ranked):
        row["image_id"] = a["image_id"]
        if "url" in a:
            row.update(url=a["url"], license=a["license"], author=a["author"])
        row["uid"], row["file"] = a["uid"], a["file"]
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
    ncfg = cfg["normals"]
    places = _normal_rows(index, "places365", ncfg["places365_count"], cfg, images_root)
    people = _normal_rows(index, "openimages_people", ncfg["people_count"], cfg, images_root)
    for source, got, want in (("places365", places, ncfg["places365_count"]),
                              ("openimages_people", people, ncfg["people_count"])):
        if len(got) < want:
            failures.append({"source": source, "key": "*", "reason": "shortfall: %d of %d available in A" % (len(got), want)})
    everything = crisis + places + people
    uids = [r["uid"] for r in everything]
    if len(set(uids)) != len(uids):
        raise ValueError("uid collision in labels_raw")
    write_jsonl(out / "labels_raw.jsonl", everything)
    write_jsonl(out / "normals_places365.jsonl", places)
    write_jsonl(out / "normals_people.jsonl", people)
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    counts = labels.report_counts(everything, cfg)
    by_source = {}
    for r in everything:
        tiers = by_source.setdefault(r["source"], {})
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    counts["totals_by_source_tier"] = by_source
    counts["normals"] = {"places365_final": len(places), "people_final": len(people),
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
