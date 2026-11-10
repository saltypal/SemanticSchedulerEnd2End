"""Four-tier urgency labels from CrisisMMD human annotations (plan v3.1 step S1).

Reuses the pinned QCRI/CrisisMMD per-task JSON layout already used by
``experiments/full_medic_labeling/select_crisismmd.py``. No VLM is involved: tiers are a fixed,
documented function of the human ``damage``, ``humanitarian`` and ``informative`` labels.
Rules live in ``config/edge/data.yaml``.

CLI:
    python -m parameters.edge.labels --meta-dir <dir with {task}_{split}.json> --out labels_raw.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from Evaluation.edge.edge_config import load_config

TIERS = ("Critical", "High", "Medium", "Low")
TIER_INDEX = {name: i for i, name in enumerate(TIERS)}


def _task_files(meta_dir, tasks, splits):
    for task in tasks:
        for split in splits:
            path = Path(meta_dir) / "{}_{}.json".format(task, split)
            if path.is_file():
                yield task, split, path


def load_crisismmd(meta_dir, cfg=None):
    """Merge the per-task JSON files into one record per image.

    A record keeps *sets* of labels per task, so annotation conflicts stay visible instead of
    being silently resolved.
    """
    cfg = cfg or load_config("data")["crisismmd"]
    records = {}
    seen = 0
    for task, split, path in _task_files(meta_dir, cfg["tasks"], cfg["native_splits"]):
        seen += 1
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = row["image_path"]
            rec = records.setdefault(key, {
                "image_path": key, "image_id": row.get("image_id"), "tweet_id": str(row["tweet_id"]),
                "event": row["event_name"], "source": "crisismmd",
                "labels": defaultdict(set), "native_splits": defaultdict(set)})
            rec["labels"][task].add(row["label"])
            rec["native_splits"][task].add(split)
    if not seen:
        raise FileNotFoundError("no CrisisMMD task files found in " + str(meta_dir))
    return records


def _matches(rule, labels):
    """True when any label set intersects the rule's lists."""
    return (bool(labels.get("damage", set()) & set(rule.get("damage", ())))
            or bool(labels.get("humanitarian", set()) & set(rule.get("humanitarian", ()))))


def _rules(tier_rules, alternative=None, alt_cfg=None):
    """Copy the tier rules, applying an optional named alternative mapping."""
    rules = {k: {kk: list(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
             for k, v in tier_rules.items()}
    if alternative:
        for tier, edits in alt_cfg[alternative].items():
            rules[tier].setdefault("humanitarian", []).extend(edits.get("humanitarian_add", []))
            for other in rules:      # a label may belong to one tier only
                if other != tier:
                    rules[other]["humanitarian"] = [x for x in rules[other].get("humanitarian", [])
                                                    if x not in edits.get("humanitarian_add", [])]
    return rules


def assign_tier(labels, rules):
    """Return (tier, flags) for one image's label sets. First matching tier wins, top to bottom."""
    flags = []
    informative = labels.get("informative", set())
    if len(informative) > 1 or any(len(labels.get(t, ())) > 1 for t in ("damage", "humanitarian")):
        flags.append("label_conflict")
    for tier in TIERS[:-1]:
        rule = rules[tier]
        if not _matches(rule, labels):
            continue
        if rule.get("requires_informative"):
            via_damage = bool(labels.get("damage", set()) & set(rule.get("damage", ())))
            if not via_damage and informative and "informative" not in informative:
                flags.append("humanitarian_vs_informative_conflict")
                continue
        if "not_informative" in informative and tier != "Critical":
            flags.append("informative_flag_overridden")
        return tier, flags
    return "Low", flags


def label_table(records, alternative=None, cfg=None):
    """One output row per image with tier, matching flags and the raw label sets."""
    cfg = cfg or load_config("data")
    rules = _rules(cfg["tier_rules"], alternative, cfg.get("alternative_mappings"))
    rows = []
    for key in sorted(records):
        rec = records[key]
        tier, flags = assign_tier(rec["labels"], rules)
        rows.append({
            "image_path": rec["image_path"], "image_id": rec["image_id"], "tweet_id": rec["tweet_id"],
            "event": rec["event"], "source": rec["source"], "tier": tier, "flags": flags,
            "damage": sorted(rec["labels"].get("damage", ())),
            "humanitarian": sorted(rec["labels"].get("humanitarian", ())),
            "informative": sorted(rec["labels"].get("informative", ()))})
    return rows


def normal_rows(items, source, cfg=None):
    """Everyday images are Low by rule. ``items``: iterable of dicts with ``path`` and optional
    ``category``. Places365 categories on the blacklist are dropped (damage look-alikes)."""
    cfg = cfg or load_config("data")["normals"]
    banned = set(cfg["places365_blacklist"]) if source == "places365" else set()
    rows, dropped = [], 0
    for item in items:
        if item.get("category") in banned:
            dropped += 1
            continue
        rows.append({"image_path": item["path"], "image_id": item.get("id"), "tweet_id": None,
                     "event": None, "source": source, "tier": cfg["source_tier"], "flags": [],
                     "damage": [], "humanitarian": [], "informative": [],
                     "category": item.get("category")})
    return rows, dropped


def report_counts(rows, cfg=None):
    """Tier counts per event and source, plus gate verdicts for the planned splits."""
    cfg = cfg or load_config("data")
    per_event = defaultdict(Counter)
    for row in rows:
        per_event[row["event"] or row["source"]][row["tier"]] += 1
    gates = cfg["splits"]["gates"]
    test_c = sum(per_event[e]["Critical"] for e in cfg["splits"]["test_events"])
    val_c = sum(per_event[e]["Critical"] for e in cfg["splits"]["val_events"])
    return {
        "per_event_tier": {e: {t: c.get(t, 0) for t in TIERS} for e, c in sorted(per_event.items())},
        "flags": dict(Counter(f for r in rows for f in r["flags"])),
        "gates": {
            "test_critical": test_c, "test_critical_ok": test_c >= gates["min_test_critical"],
            "val_critical": val_c, "val_critical_ok": val_c >= gates["min_val_critical"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--out", required=True, help="labels JSONL")
    parser.add_argument("--report", help="counts JSON (default: next to --out)")
    parser.add_argument("--alternative", help="named alternative mapping from data.yaml")
    args = parser.parse_args(argv)
    rows = label_table(load_crisismmd(args.meta_dir), args.alternative)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report = report_counts(rows)
    report_path = Path(args.report) if args.report else out.with_name(out.stem + "_counts.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
