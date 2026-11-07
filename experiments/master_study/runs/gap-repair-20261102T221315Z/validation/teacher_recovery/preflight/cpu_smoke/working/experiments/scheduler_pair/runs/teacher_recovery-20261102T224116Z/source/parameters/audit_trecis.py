"""Audit complete official annotation snapshots without conflating merged rows and votes.

Usage: python parameters/audit_trecis.py --run <download-run-directory>
No network requests, model inference, or modification of downloaded source files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def topic_records(path: Path) -> list[dict]:
    """The official topic files are concatenated TREC SGML blocks, not XML."""
    text = path.read_text(encoding="utf8")
    records = []
    for block in re.findall(r"<top>(.*?)</top>", text, re.S):
        record = {}
        for field in ("num", "dataset", "title", "type", "url", "narr"):
            match = re.search(fr"<{field}>(.*?)</{field}>", block, re.S)
            record[field] = " ".join(match[1].split()) if match else None
        number = re.search(r"(\d+)$", record.get("num") or "")
        record["number"] = int(number[1]) if number else None
        records.append(record)
    if len(records) != text.count("<top>"):
        raise ValueError(f"Unclosed topic block in {path}")
    return records


def audit(run: Path) -> dict:
    raw = run / "raw"
    annotation_paths = sorted(raw.glob("annotations_*.json.gz"))
    if not annotation_paths:
        raise FileNotFoundError(f"No consolidated annotation files under {raw}")
    result = {"scope": "All downloaded consolidated annotation rows; not all archive pixels",
              "files": {}, "events": [], "source_hashes": {}}
    priorities = defaultdict(set)
    events_by_post = defaultdict(set)
    texts_by_post = defaultdict(set)
    leaves = {}
    post_categories = defaultdict(set)
    post_files = defaultdict(set)
    per_event_posts = defaultdict(set)
    for path in annotation_paths:
        result["source_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt", encoding="utf8") as handle:
            document = json.load(handle)
        rows = [row for event in document["events"] for row in event["tweets"]]
        nested = [vote for row in rows for vote in (row.get("multipleJudgements") or [])]
        for row in rows:
            identity = str(row["postID"])
            priorities[identity].add(row.get("postPriority"))
            events_by_post[identity].add(row.get("eventID"))
            texts_by_post[identity].add(row.get("postText"))
            post_categories[identity].update(row.get("postCategories") or [])
            post_files[identity].add(path.name)
            per_event_posts[row.get("eventID")].add(identity)
            # Merged parent rows are not additional independent annotator votes.
            for vote in row.get("multipleJudgements") or [row]:
                key = json.dumps({k: v for k, v in vote.items() if k != "multipleJudgements"}, sort_keys=True)
                leaves[key] = vote
        result["files"][path.name] = {
            "published_metadata": document["metadata"],
            "event_records": len(document["events"]), "top_level_rows": len(rows),
            "unique_post_ids": len({str(r["postID"]) for r in rows}),
            "unique_event_post_pairs": len({(r["eventID"], str(r["postID"])) for r in rows}),
            "priority_rows": dict(Counter(r.get("postPriority") for r in rows)),
            "category_rows": dict(Counter(c for r in rows for c in r.get("postCategories", []))),
            "rows_with_nested_judgements": sum(bool(r.get("multipleJudgements")) for r in rows),
            "nested_judgements": len(nested),
            "blank_text_rows": sum(not (r.get("postText") or "").strip() for r in rows),
            "unknown_priority_rows": sum(r.get("postPriority") not in {"Low", "Medium", "High", "Critical"} for r in rows),
            "row_keys": sorted({k for r in rows for k in r}),
        }
        for event in document["events"]:
            rs = event["tweets"]
            result["events"].append({"file": path.name, "event_id": event["eventid"],
                                     "trecis_id": event.get("trecisid"), "type": event.get("type"),
                                     "rows": len(rs), "unique_posts": len({str(r["postID"]) for r in rs}),
                                     "priorities": dict(Counter(r.get("postPriority") for r in rs))})
    vote_priorities = defaultdict(set)
    assessor_keys = defaultdict(list)
    annotated_events = {e["event_id"] for e in result["events"]}
    nested_depth = 0
    for vote in leaves.values():
        vote_priorities[str(vote["postID"])].add(vote.get("postPriority"))
        assessor_keys[(vote.get("eventID"), str(vote["postID"]), vote.get("annotator"))].append(vote)
        nested_depth += bool(vote.get("multipleJudgements"))
    if nested_depth:
        raise ValueError("Unexpected deeper judgement nesting; audit needs explicit recursion")
    result["combined"] = {
        "unique_post_ids": len(priorities),
        "unique_leaf_records": len(leaves),
        "leaf_priority_distribution": dict(Counter(v.get("postPriority") for v in leaves.values())),
        "posts_with_conflicting_top_level_priority": sum(len(v) > 1 for v in priorities.values()),
        "posts_with_conflicting_leaf_priority": sum(len(v) > 1 for v in vote_priorities.values()),
        "posts_in_multiple_event_ids": sum(len(v) > 1 for v in events_by_post.values()),
        "posts_with_multiple_text_versions": sum(len(v) > 1 for v in texts_by_post.values()),
        "posts_in_both_annotation_files": sum(len(v) > 1 for v in post_files.values()),
        "distinct_event_ids": len(annotated_events),
        "unique_event_post_pairs": sum(len(v) for v in per_event_posts.values()),
        "leaf_assessor_keys": len(assessor_keys),
        "assessor_keys_with_multiple_record_versions": sum(len(v) > 1 for v in assessor_keys.values()),
        "assessor_keys_with_priority_conflict": sum(len({r.get("postPriority") for r in v}) > 1 for v in assessor_keys.values()),
        "top_level_priority_set_counts": dict(Counter("|".join(sorted(str(x) for x in ps)) for ps in priorities.values())),
    }
    image_labels = {"Satellite", "WeatherForecast", "Infographic", "WeatherPhoto", "Victims", "Damage",
                    "Crowds", "RecoveryEfforts", "Screenshot", "Informative"}
    result["image_annotation_proxy"] = {
        "definition": "At least one observed image-category label, not proof of all image-bearing posts or complete annotation",
        "categories": sorted(image_labels),
        "unique_posts_with_any": sum(bool(cs & image_labels) for cs in post_categories.values()),
        "category_unique_posts": {c: sum(c in cs for cs in post_categories.values()) for c in sorted(image_labels)},
        "multimedia_share_unique_posts": sum("MultimediaShare" in cs for cs in post_categories.values()),
    }
    result["assessor_conflict_breakdown"] = {
        str(a): {"keys": len(keys), "conflicting_priority_keys": sum(len({r.get("postPriority") for r in vs}) > 1 for vs in keys)}
        for a in sorted({k[2] for k in assessor_keys}, key=str)
        for keys in [[v for k, v in assessor_keys.items() if k[2] == a]]
    }
    # Conservative candidate event components: same-post links, excluding the artificial
    # training panel. Panel appearances do not mean all represented crises are one event.
    parents = {e: e for e in annotated_events if e != "2021atraining"}
    def find(event):
        while parents[event] != event:
            parents[event] = parents[parents[event]]
            event = parents[event]
        return event
    for assignments in events_by_post.values():
        es = sorted(e for e in assignments if e in parents)
        for e in es[1:]:
            parents[find(e)] = find(es[0])
    components = defaultdict(list)
    for event in sorted(parents):
        components[find(event)].append(event)
    result["candidate_event_components"] = {
        "definition": "Same-post linkage only, excluding 2021atraining; titles/sites/near-duplicate media still require reconciliation",
        "count": len(components),
        "multi_event_components": [v for v in components.values() if len(v) > 1],
        "calibration_panel_posts": len(per_event_posts.get("2021atraining", set())),
        "panel_only_posts": sum(es == {"2021atraining"} for es in events_by_post.values()),
    }
    # Drive's public HTML exposes one page. Never call its first-page listing complete.
    drive = raw / "drive_folder.html"
    if drive.exists():
        match = re.search(r"window\['_DRIVE_ivd'\] = '(.*?)';", drive.read_text(encoding="utf8"))
        if match:
            decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), match[1])
            listing = json.loads(decoded.replace("\\/", "/"))
            inventory = [{"file_id": r[0], "name": r[2], "mime": r[3], "listed_bytes": r[13]} for r in listing[0]]
            result["media_listing"] = {"scope": "First public Drive listing page; pagination not exhausted",
                                       "files": inventory, "listed_files": len(inventory),
                                       "listed_bytes": sum(r["listed_bytes"] or 0 for r in inventory),
                                       "pixels_audited": 0}
    for path in raw.glob("topics_*.xml"):
        records = topic_records(path)
        result.setdefault("topics", {})[path.name] = {"format": "TREC SGML", "records": records, "count": len(records)}
    ontology = json.loads((raw / "information_types.json").read_text(encoding="utf8"))
    result["information_type_ontology"] = ontology
    result["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = run / "AUDIT.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"output": str(output.resolve()), "combined": result["combined"],
                      "files": {k: {x: v[x] for x in ["top_level_rows", "unique_post_ids", "unique_event_post_pairs", "priority_rows", "nested_judgements"]} for k, v in result["files"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    audit(parser.parse_args().run)
