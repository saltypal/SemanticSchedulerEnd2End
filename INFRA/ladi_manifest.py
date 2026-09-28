"""Curate and validate a reviewed 60-image LADI v2 scheduling manifest.

LADI labels are multi-label observations, not semantic-priority ground truth.
The draft pairing step uses image similarity only to nominate related views;
human review is mandatory before the manifest is accepted for evaluation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


DATASET_ID = "MITLL/LADI-v2-dataset"
NATIVE_LABELS = (
    "bridges_any", "buildings_any", "buildings_affected_or_greater",
    "buildings_minor_or_greater", "debris_any", "flooding_any",
    "flooding_structures", "roads_any", "roads_damage", "trees_any",
    "trees_damage", "water_any",
)
STRATA = (
    "flooding_structures", "buildings_affected_or_greater",
    "roads_damage", "debris_any", "trees_damage", "infrastructure_control",
)
RUBRIC = {
    "flooding_structures": (3, 3, 0),
    "buildings_affected_or_greater": (3, 2, 1),
    "roads_damage": (2, 3, 1),
    "debris_any": (2, 2, 2),
    "trees_damage": (1, 1, 2),
    "infrastructure_control": (0, 0, 3),
}
MANIFEST_FIELDS = (
    "image_id", "source_uri", "dataset_revision", "source_split",
    "source_row_index", "image_path", "source_sha256", "native_labels_json",
    "selection_stratum", "evidence_group_id", "split", "base_priority",
    "task_impact", "urgency", "evidence_sufficiency",
    "annotation_source", "reviewer", "rationale", "review_status",
)


def selection_stratum(labels: Mapping[str, object]) -> str | None:
    """Assign one selection stratum while retaining every native label."""
    for label in STRATA[:-1]:
        if bool(labels.get(label, False)):
            return label
    if (bool(labels.get("buildings_any", False))
            or bool(labels.get("roads_any", False))
            or bool(labels.get("bridges_any", False))):
        return "infrastructure_control"
    return None


def _image_bytes(image) -> bytes:
    if isinstance(image, dict):
        raw = image.get("bytes")
        if raw is None:
            raise ValueError("Lazy dataset row lacks embedded image bytes")
        from PIL import Image
        with Image.open(BytesIO(raw)) as opened:
            image = opened.convert("RGB")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def draft_ladi60_manifest(
    dataset_rows: Iterable[Mapping[str, object]],
    output_dir: str | Path,
    *, dataset_revision: str,
    source_split: str,
    embedding_backend,
    pool_per_stratum: int = 20,
) -> Path:
    """Create 60 distinct candidate images and a review-required manifest.

    ``dataset_rows`` is normally a pinned Hugging Face streaming split. The
    function stops as soon as each stratum has enough candidates, so it does
    not intentionally download the whole LADI dataset.
    """
    if pool_per_stratum < 10:
        raise ValueError("Need at least ten candidates per stratum")
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for index, row in enumerate(dataset_rows):
        labels = {name: bool(row.get(name, False)) for name in NATIVE_LABELS}
        stratum = selection_stratum(labels)
        if stratum is None or len(pools[stratum]) >= pool_per_stratum:
            continue
        image = row.get("image")
        if image is None:
            raise ValueError("LADI row lacks image data")
        payload = _image_bytes(image)
        digest = sha256(payload).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        image_id = f"ladi-{digest[:16]}"
        path = image_dir / f"{image_id}.png"
        path.write_bytes(payload)
        embedding = np.asarray(embedding_backend.encode_image(path), dtype=np.float32).reshape(-1)
        if not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
            raise ValueError(f"Invalid embedding for {image_id}")
        pools[stratum].append({
            "image_id": image_id, "source_sha256": digest,
            "image_path": str(Path("images") / path.name), "source_row_index": index,
            "source_uri": f"hf://datasets/{DATASET_ID}@{dataset_revision}/{source_split}/{index}",
            "native_labels_json": json.dumps(labels, sort_keys=True),
            "embedding": embedding / np.linalg.norm(embedding),
        })
        if all(len(pools[name]) >= pool_per_stratum for name in STRATA):
            break
    missing = {name: pool_per_stratum - len(pools[name]) for name in STRATA
               if len(pools[name]) < pool_per_stratum}
    if missing:
        raise ValueError(f"Dataset stream ended before candidate pools filled: {missing}")

    selected: list[dict[str, object]] = []
    for stratum in STRATA:
        available = list(pools[stratum])
        impact, urgency, priority = RUBRIC[stratum]
        for group_index in range(5):
            best = max(((left, right) for left in range(len(available))
                        for right in range(left + 1, len(available))),
                       key=lambda pair: (_cosine(available[pair[0]]["embedding"],
                                                  available[pair[1]]["embedding"]),
                                         -available[pair[0]]["source_row_index"],
                                         -available[pair[1]]["source_row_index"]))
            pair = [available[best[0]], available[best[1]]]
            for item in sorted(pair, key=lambda value: value["image_id"]):
                selected.append({
                    **{key: value for key, value in item.items() if key != "embedding"},
                    "dataset_revision": dataset_revision, "source_split": source_split,
                    "selection_stratum": stratum,
                    "evidence_group_id": f"{stratum}-group{group_index + 1}",
                    "split": "train" if group_index < 3 else "validation" if group_index == 3 else "test",
                    "base_priority": priority, "task_impact": impact, "urgency": urgency,
                    "evidence_sufficiency": "", "annotation_source": "rubric_draft",
                    "reviewer": "", "rationale": "", "review_status": "pending",
                })
            for index in sorted(best, reverse=True):
                del available[index]
    manifest = output / "ladi60_semantic_scheduler_v1.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    return manifest


def validate_ladi60_manifest(path: str | Path, *, require_reviewed: bool = True) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("LADI manifest columns or order do not match schema v1")
        rows = list(reader)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 LADI images, found {len(rows)}")
    if len({row["image_id"] for row in rows}) != 60:
        raise ValueError("LADI manifest image IDs are not unique")
    if len({row["source_sha256"] for row in rows}) != 60:
        raise ValueError("LADI manifest image hashes are not unique")
    if Counter(row["selection_stratum"] for row in rows) != Counter({name: 10 for name in STRATA}):
        raise ValueError("Expected exactly ten images in each selection stratum")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        raw_image_path = Path(row["image_path"])
        resolved_image_path = (raw_image_path if raw_image_path.is_absolute()
                               else manifest_path.parent / raw_image_path)
        row["image_path"] = str(resolved_image_path)
        groups[row["evidence_group_id"]].append(row)
        labels = json.loads(row["native_labels_json"])
        if set(labels) != set(NATIVE_LABELS):
            raise ValueError(f"Native label schema is incomplete for {row['image_id']}")
        if row["selection_stratum"] != selection_stratum(labels):
            raise ValueError(f"Selection precedence disagrees for {row['image_id']}")
        if not resolved_image_path.is_file():
            raise FileNotFoundError(f"Missing curated image {resolved_image_path}")
        if sha256(resolved_image_path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Image hash differs for {row['image_id']}")
        if require_reviewed:
            if row["review_status"] != "reviewed" or not row["reviewer"]:
                raise ValueError(f"Human review is incomplete for {row['image_id']}")
            if row["evidence_sufficiency"] not in ("0", "1"):
                raise ValueError(f"Evidence sufficiency is unreviewed for {row['image_id']}")
        default = RUBRIC[row["selection_stratum"]]
        chosen = tuple(int(row[name]) for name in ("task_impact", "urgency", "base_priority"))
        if chosen != default and not (row["reviewer"] and row["rationale"]):
            raise ValueError(f"Rubric override lacks reviewer/rationale for {row['image_id']}")
    if len(groups) != 30 or any(len(items) != 2 for items in groups.values()):
        raise ValueError("Expected thirty two-image evidence groups")
    for group, items in groups.items():
        if len({row["split"] for row in items}) != 1:
            raise ValueError(f"Evidence group {group} crosses splits")
    if Counter(row["split"] for row in rows) != Counter({"train": 36, "validation": 12, "test": 12}):
        raise ValueError("Expected a group-safe 36/12/12 split")
    return rows


def render_contact_sheet(manifest_path: str | Path, output_path: str | Path) -> Path:
    """Render the proposed pairs side by side for required visual review."""
    from PIL import Image, ImageDraw, ImageOps

    rows = validate_ladi60_manifest(manifest_path, require_reviewed=False)
    ordered = sorted(rows, key=lambda row: (
        STRATA.index(row["selection_stratum"]), row["evidence_group_id"], row["image_id"]))
    tile_width, tile_height = 300, 210
    sheet = Image.new("RGB", (6 * tile_width, 10 * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(ordered):
        column = index // 10
        line = index % 10
        x, y = column * tile_width, line * tile_height
        with Image.open(row["image_path"]) as opened:
            thumbnail = ImageOps.contain(opened.convert("RGB"), (tile_width - 12, tile_height - 45))
        sheet.paste(thumbnail, (x + (tile_width - thumbnail.width) // 2, y + 5))
        label = f"{row['selection_stratum']} | {row['evidence_group_id'].split('-')[-1]}"
        draw.text((x + 6, y + tile_height - 38), label, fill="black")
        draw.text((x + 6, y + tile_height - 20), row["image_id"], fill="black")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination
