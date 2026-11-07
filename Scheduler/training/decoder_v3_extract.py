"""Pinned LADI-v2 image extraction for the decoder-v3 scheduler study.

The extractor keeps model features and native evaluator labels in separate
JSONL files. It never manufactures labels, event IDs, or capture timestamps.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "MITLL/LADI-v2-dataset"
DATASET_REVISION = "5f2dbfe8c466d32edafd1bab847ec5252309acdb"
TASK_PROFILES = ("building_damage", "road_damage", "structural_flooding")
SPLIT_ROLES = {"validation": ("A", 0.50), "test": ("test", 1.0)}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_image(image):
    """Apply EXIF orientation before pixels are hashed or analyzed."""
    from PIL import ImageOps
    return ImageOps.exif_transpose(image).convert("RGB")


def normalized_phash(image) -> int:
    """Small deterministic DCT-free pHash; duplicate threshold is <=4 bits."""
    import numpy as np
    gray = image.convert("L").resize((32, 32))
    pixels = np.asarray(gray, dtype=np.float32)
    # Low-frequency cosine projection (8x8); no image/model dependency beyond Pillow.
    x = np.arange(32, dtype=np.float32)
    basis = np.cos((2 * x[None, :] + 1) * np.arange(8, dtype=np.float32)[:, None] * (np.pi / 64))
    coeff = basis @ pixels @ basis.T
    low = coeff[:8, :8].copy()
    threshold = float(np.median(low.reshape(-1)[1:]))
    bits = low >= threshold
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming64(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


def duplicate_groups(rows: list[dict[str, Any]], max_distance: int = 4) -> None:
    """Assign conservative connected near-duplicate groups in-place."""
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, row in enumerate(rows):
        for j in range(i):
            if row["image_sha256"] == rows[j]["image_sha256"] or hamming64(row["phash64"], rows[j]["phash64"]) <= max_distance:
                a, b = root(i), root(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    for members in groups.values():
        group_hash = hashlib.sha256("\n".join(sorted(rows[i]["image_sha256"] for i in members)).encode()).hexdigest()[:16]
        for i in members:
            rows[i]["duplicate_group"] = group_hash


def assign_roles(rows: list[dict[str, Any]], seed: int = 20261022) -> None:
    """Assign A/B/C by duplicate group, 50/20/30 within official validation."""
    import random
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["duplicate_group"], []).append(row)
    for group, members in list(groups.items()):
        if {row['official_split'] for row in members} == {'validation', 'test'}:
            for row in members:
                if row['official_split'] == 'validation':
                    row['role'] = 'excluded_test_overlap'
            groups[group] = [row for row in members if row['official_split'] == 'test']
    by_split: dict[str, list[str]] = {}
    for group, members in groups.items():
        split = members[0]["official_split"]
        by_split.setdefault(split, []).append(group)
    rng = random.Random(seed)
    for split, members in by_split.items():
        members.sort()
        rng.shuffle(members)
        if split == "test":
            role_by_group = {group: "test" for group in members}
        elif split == "validation":
            n = len(members)
            a_end, b_end = int(n * .50), int(n * .70)
            role_by_group = {group: ("A" if i < a_end else "B" if i < b_end else "C") for i, group in enumerate(members)}
        else:
            raise ValueError(f"Unexpected split for decoder-v3: {split}")
        for group in members:
            for row in groups[group]:
                row["role"] = role_by_group[group]


def download_arrow(root: str | Path, allow_network: bool = False) -> Path:
    """Fetch only pinned validation/test Arrow shards; never the 8k train rows."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not allow_network:
        raise RuntimeError("Dataset is not present; set allow_network=True explicitly to download pinned validation/test shards")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
        local_dir=str(root), allow_patterns=[
            "data/validation/data-00000-of-00040.arrow", "data/validation/data-00001-of-00040.arrow",
            "data/validation/data-00002-of-00040.arrow", "data/validation/data-00003-of-00040.arrow",
            "data/validation/data-00004-of-00040.arrow", "data/validation/data-00005-of-00040.arrow",
            "data/validation/data-00006-of-00040.arrow", "data/validation/data-00007-of-00040.arrow",
            "data/validation/data-00008-of-00040.arrow", "data/validation/data-00009-of-00040.arrow",
            "data/validation/data-00010-of-00040.arrow", "data/validation/data-00011-of-00040.arrow",
            "data/validation/data-00012-of-00040.arrow", "data/validation/data-00013-of-00040.arrow",
            "data/validation/data-00014-of-00040.arrow", "data/validation/data-00015-of-00040.arrow",
            "data/validation/data-00016-of-00040.arrow", "data/validation/data-00017-of-00040.arrow",
            "data/validation/data-00018-of-00040.arrow", "data/validation/data-00019-of-00040.arrow",
            "data/validation/data-00020-of-00040.arrow", "data/validation/data-00021-of-00040.arrow",
            "data/validation/data-00022-of-00040.arrow", "data/validation/data-00023-of-00040.arrow",
            "data/validation/data-00024-of-00040.arrow", "data/validation/data-00025-of-00040.arrow",
            "data/validation/data-00026-of-00040.arrow", "data/validation/data-00027-of-00040.arrow",
            "data/validation/data-00028-of-00040.arrow", "data/validation/data-00029-of-00040.arrow",
            "data/validation/data-00030-of-00040.arrow", "data/validation/data-00031-of-00040.arrow",
            "data/validation/data-00032-of-00040.arrow", "data/validation/data-00033-of-00040.arrow",
            "data/validation/data-00034-of-00040.arrow", "data/validation/data-00035-of-00040.arrow",
            "data/validation/data-00036-of-00040.arrow", "data/validation/data-00037-of-00040.arrow",
            "data/validation/data-00038-of-00040.arrow", "data/validation/data-00039-of-00040.arrow",
        ] + [f'data/test/data-{index:05d}-of-00040.arrow' for index in range(40)]
          + [f'data/{split}/{name}' for split in ('validation', 'test')
             for name in ('dataset_info.json', 'state.json')],
        max_workers=4, etag_timeout=30,
    )
    return root


def load_arrow_rows(root: str | Path) -> Iterable[dict[str, Any]]:
    """Read Arrow shards directly; no dataset viewer/export transformation."""
    from datasets import Dataset, Image as DatasetImage
    root = Path(root)
    for split, count in (("validation", 40), ("test", 40)):
        for index in range(count):
            path = root / "data" / split / f"data-{index:05d}-of-{count:05d}.arrow"
            if not path.is_file():
                raise FileNotFoundError(path)
            dataset = Dataset.from_file(str(path)).cast_column('image', DatasetImage(decode=False))
            for row in dataset:
                row["_official_split"] = split
                yield row


def exclude_conflicting_labels(manifest):
    """Keep original rows, but do not train/evaluate on contradictory copies.

    Exact byte identity is sufficient here; no semantic matching or label voting.
    Grouped split assignment stays unchanged for every unaffected observation.
    """
    by_id = {}
    for row in manifest:
        by_id.setdefault(row['image_id'], []).append(row)
    conflicts = []
    for image_id, copies in by_id.items():
        fields = sorted(name for name in copies[0]['labels']
                        if len({row['labels'][name] for row in copies}) > 1)
        if not fields:
            continue
        conflicts.append(dict(image_id=image_id, conflicting_fields=fields, copies=len(copies)))
        for row in copies:
            row.setdefault('original_role', row['role'])
            row['role'] = 'excluded_conflicting_labels'
            row['conflicting_fields'] = fields
    return conflicts


def prepare_manifest(rows: Iterable[dict[str, Any]], image_root=None) -> tuple[list[dict[str, Any]], dict]:
    """Normalize upstream image rows, split duplicate groups, and isolate labels."""
    from io import BytesIO
    from PIL import Image
    from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import LADI_LABELS
    manifest, images = [], {}
    if image_root is not None:
        image_root = Path(image_root)
        image_root.mkdir(parents=True, exist_ok=True)
    for source in rows:
        image_value = source.get("image")
        raw = image_value.get("bytes") if isinstance(image_value, dict) else None
        if raw is None and isinstance(image_value, dict) and image_value.get("path"):
            raw = Path(image_value["path"]).read_bytes()
        if not raw:
            raise ValueError("LADI row has no image bytes; refusing path-derived or synthetic substitute")
        with Image.open(BytesIO(raw)) as opened:
            image = canonical_image(opened)
        image_sha = sha256_bytes(raw)
        image_id = image_sha
        if any(name not in source or not isinstance(source[name], (bool, int)) or source[name] not in (0, 1) for name in LADI_LABELS):
            raise ValueError('Expected all twelve native boolean label columns in LADI Arrow row')
        labels = {name: bool(source[name]) for name in LADI_LABELS}
        manifest.append({"image_id": image_id, "official_split": source["_official_split"],
                         "image_sha256": image_sha, "phash64": normalized_phash(image),
                         "width": image.width, "height": image.height, "labels": labels,
                         "event_id": None, "capture_time_ms": None})
        if image_root is None:
            images[image_id] = raw
        else:
            # Keep original compressed bytes instead of much larger PNG copies.
            path = image_root / f'{image_id}.image'
            if not path.exists():
                path.write_bytes(raw)
            images[image_id] = str(path)
    duplicate_groups(manifest)
    assign_roles(manifest)
    exclude_conflicting_labels(manifest)
    return manifest, images


def extract_rows(manifest: list[dict[str, Any]], images: dict[str, bytes], extractor, feature_path: str | Path,
                 label_path: str | Path, resume: bool = True, profile_images: int = 0) -> dict[str, int]:
    """Incrementally persist immutable feature rows and evaluator-only labels."""
    from io import BytesIO
    from PIL import Image
    feature_path, label_path = Path(feature_path), Path(label_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    def existing(path):
        if not resume or not path.exists():
            return {}
        lines = path.read_text(encoding='utf-8').splitlines()
        result = {}
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines)-1:
                    break  # recover a last record interrupted during writing
                raise
            result[value['image_id']] = value
        return result
    previous_features, previous_labels = existing(feature_path), existing(label_path)
    eligible = {row['image_id'] for row in manifest if not row['role'].startswith('excluded')}
    completed = set(previous_features) & set(previous_labels) & eligible
    # An interrupted feature/label pair is recomputed; deduplicate old appends.
    for path, records in ((feature_path, previous_features), (label_path, previous_labels)):
        with path.open('w', encoding='utf-8') as handle:
            for image_id in sorted(completed):
                handle.write(json.dumps(records[image_id], sort_keys=True, allow_nan=False)+'\n')
    counts = {"written": 0, "resumed": len(completed), "failed": 0}
    with feature_path.open("a", encoding="utf-8") as features, label_path.open("a", encoding="utf-8") as labels:
        for row in manifest:
            if row['role'].startswith('excluded'):
                continue
            if row["image_id"] in completed:
                if previous_labels[row['image_id']]['labels'] != row['labels'] or previous_labels[row['image_id']]['role'] != row['role']:
                    raise ValueError('cached label or role conflicts with immutable manifest')
                continue
            try:
                stored = images[row['image_id']]
                with (Image.open(BytesIO(stored)) if isinstance(stored, bytes) else Image.open(stored)) as opened:
                    image = canonical_image(opened)
                evidence = extractor.extract(image, row["image_id"])
                if counts['written']+counts['resumed'] < profile_images:
                    try:
                        evidence.raw['inference_costs'] = extractor.profile(image)
                    except Exception as error:
                        evidence.raw['cost_profile_error'] = f'{type(error).__name__}: {error}'
                serialized = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
                feature_row = {**serialized, "image_id": row["image_id"], "role": row["role"],
                               "official_split": row["official_split"], "duplicate_group": row["duplicate_group"],
                               "image_sha256": row["image_sha256"], "width": row["width"], "height": row["height"]}
                features.write(json.dumps(feature_row, sort_keys=True, allow_nan=False) + "\n")
                features.flush()
                label_row = {"image_id": row["image_id"], "official_split": row["official_split"],
                             "role": row["role"], "duplicate_group": row["duplicate_group"],
                             "labels": row["labels"]}
                labels.write(json.dumps(label_row, sort_keys=True, allow_nan=False) + "\n")
                labels.flush()
                counts["written"] += 1
                completed.add(row['image_id'])
                previous_features[row['image_id']] = feature_row
                previous_labels[row['image_id']] = label_row
                image.close()
                if counts['written'] == 1 or counts['written'] % 16 == 0:
                    print(f'extraction {getattr(extractor,"device","test")}: {counts["written"]} new, {counts["resumed"]} resumed', flush=True)
            except Exception as exc:
                counts["failed"] += 1
                error_path = feature_path.parent / "extraction_errors.jsonl"
                with error_path.open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({"image_id": row["image_id"], "error": repr(exc)}) + "\n")
                raise
    return counts


def run_worker(worker_manifest):
    from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import FrozenImageExtractor
    import torch
    torch.set_num_threads(2)
    payload = json.loads(Path(worker_manifest).read_text())
    folder = Path(payload['output'])
    extractor = FrozenImageExtractor(device=payload['device'])
    try:
        result = extract_rows(payload['rows'], payload['images'], extractor,
                              folder/'features.jsonl',folder/'labels.jsonl')
        from PIL import Image
        with (folder/'inference_costs.jsonl').open('w',encoding='utf-8') as handle:
            for row in payload['rows'][:4]:
                try:
                    with Image.open(payload['images'][row['image_id']]) as image:
                        cost = extractor.profile(canonical_image(image))
                    record = dict(image_id=row['image_id'],device=payload['device'],costs=cost)
                except Exception as error:
                    record = dict(image_id=row['image_id'],device=payload['device'],error=repr(error))
                handle.write(json.dumps(record,allow_nan=False)+'\n')
                handle.flush()
        return result
    finally:
        extractor.close()


def extract_two_gpus(rows, images, output_dir, seed_cache=None):
    """Independent batch-one inference on disjoint images; no model replication training."""
    import subprocess
    import sys
    output_dir = Path(output_dir)
    unique = {}
    for row in rows:
        if row['role'].startswith('excluded'):
            continue
        key = row['image_id']
        if key in unique and unique[key]['labels'] != row['labels']:
            raise ValueError('identical source bytes have conflicting native truth')
        unique[key] = row
    rows = sorted(unique.values(),key=lambda row:row['image_id'])
    seed_features, seed_labels = {}, {}
    if seed_cache is not None:
        seed_cache = Path(seed_cache)
        for name,target in (('features.jsonl',seed_features),('labels.jsonl',seed_labels)):
            for line in (seed_cache/name).read_text().splitlines():
                record = json.loads(line)
                target[record['image_id']] = record
        if set(seed_features) != set(seed_labels):
            raise ValueError('partial seed cache has unpaired feature/label records')
        from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import MODEL_REVISIONS
        for image_id,feature in seed_features.items():
            if image_id not in unique:
                continue
            expected = unique[image_id]
            if seed_labels[image_id]['labels'] != expected['labels'] or seed_labels[image_id]['role'] != expected['role']:
                raise ValueError('seed cache truth/role differs from pinned manifest')
            producer = feature.get('raw',{}).get('model_revisions',{})
            if any(producer.get(name) != revision for name,revision in MODEL_REVISIONS.items() if not name.startswith('Qwen/')):
                raise ValueError('seed cache image producer revisions are incompatible')
    processes, logs = [], []
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])+os.pathsep+environment.get('PYTHONPATH','')
    environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    try:
        for gpu in range(2):
            folder = output_dir/f'gpu{gpu}'
            folder.mkdir(parents=True,exist_ok=True)
            shard = rows[gpu::2]
            if not (folder/'features.jsonl').exists():
                seeded = [row['image_id'] for row in shard if row['image_id'] in seed_features]
                for name,records in (('features.jsonl',seed_features),('labels.jsonl',seed_labels)):
                    (folder/name).write_text(''.join(json.dumps(records[key])+'\n' for key in seeded),encoding='utf-8')
            manifest = folder/'worker_manifest.json'
            manifest.write_text(json.dumps(dict(rows=shard,images={row['image_id']:images[row['image_id']] for row in shard},
                                                device=f'cuda:{gpu}',output=str(folder))),encoding='utf-8')
            log = (folder/'worker.log').open('w',encoding='utf-8')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-m','Scheduler.training.decoder_v3_extract',
                                               '--worker-manifest',str(manifest)],env=environment,stdout=log,stderr=subprocess.STDOUT))
        import time
        positions = [0,0]
        while any(process.poll() is None for process in processes):
            for gpu,process in enumerate(processes):
                with (output_dir/f'gpu{gpu}'/'worker.log').open(encoding='utf-8',errors='replace') as reader:
                    reader.seek(positions[gpu])
                    lines = reader.readlines()
                    positions[gpu] = reader.tell()
                for line in lines:
                    if line.startswith('extraction '):
                        print(line.strip(),flush=True)
                if process.poll() not in (None,0):
                    raise RuntimeError(f'GPU worker {gpu} failed ({process.returncode}); inspect gpu{gpu}/worker.log')
            time.sleep(5)
        if any(process.returncode for process in processes):
            raise RuntimeError('GPU extraction worker failed; inspect worker logs')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    for name in ('features.jsonl','labels.jsonl'):
        merged = {}
        for gpu in range(2):
            for line in (output_dir/f'gpu{gpu}'/name).read_text().splitlines():
                record = json.loads(line)
                if record['image_id'] in merged:
                    raise ValueError('duplicate GPU-shard result')
                merged[record['image_id']] = record
        if set(merged) != set(unique):
            raise ValueError('GPU shard merge is incomplete')
        temporary = output_dir/(name+'.tmp')
        temporary.write_text(''.join(json.dumps(merged[key],allow_nan=False)+'\n' for key in sorted(merged)),encoding='utf-8')
        temporary.replace(output_dir/name)
    return dict(written=len(unique),seed_cache_records=len(set(seed_features)&set(unique)),gpu_workers=2,profiled_images_per_worker=4)


def run_extraction(output_dir, data_root, device='auto', qwen_images=32, require_two_gpus=True, seed_cache=None):
    """Run actual frozen inference, then bounded VLM proposals, with provenance."""
    import platform
    import shutil
    import time
    from collections import Counter
    from PIL import Image
    from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import FrozenImageExtractor, MODEL_REVISIONS
    from Scheduler.training.decoder_v3_ollama import OllamaQwenReviewer, QWEN_MODEL
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        target = output_dir/name
        temporary = target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)
    save('status.json', dict(stage='extract', status='running', dataset_revision=DATASET_REVISION))
    try:
        if require_two_gpus and torch.cuda.device_count() < 2:
            raise RuntimeError('Two GPUs required by this execution profile')
        Path(data_root).mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(data_root)
        save('disk_preflight.json', dict(free_bytes=disk.free, total_bytes=disk.total))
        if disk.free < 16 * 1024**3:
            raise RuntimeError('At least 16 GiB free temporary storage required for pinned dataset and frozen models')
        download_arrow(data_root, allow_network=True)
        rows, images = prepare_manifest(load_arrow_rows(data_root), Path(data_root)/'source_cache')
        save('label_conflicts.json', [dict(image_id=row['image_id'],
             original_role=row['original_role'], conflicting_fields=row['conflicting_fields'],
             labels=row['labels'], official_split=row['official_split'])
             for row in rows if row['role'] == 'excluded_conflicting_labels'])
        counts = Counter(row['official_split'] for row in rows)
        if counts != {'validation': 890, 'test': 1049}:
            raise RuntimeError(f'pinned dataset row-count mismatch: {dict(counts)}')
        save('manifest.json', rows)
        if require_two_gpus:
            result = extract_two_gpus(rows, images, output_dir, seed_cache)
        else:
            extractor = FrozenImageExtractor(device=device)
            result = extract_rows(rows, images, extractor, output_dir/'features.jsonl', output_dir/'labels.jsonl',profile_images=8)
            extractor.close()
        feature_rows = [json.loads(line) for line in (output_dir/'features.jsonl').read_text().splitlines()]
        component_counts = {name: sum(row.get('statuses', {}).get(name) == 'ok' for row in feature_rows)
                            for name in ('clip', 'ladi', 'detector')}
        if any(count < .95 * len(feature_rows) for count in component_counts.values()):
            raise RuntimeError(f'frozen-model success fraction below .95: {component_counts}/{len(feature_rows)}')
        save('extraction_summary.json', dict(result=result, roles=dict(Counter(row['role'] for row in rows)),
                                            dataset_counts=dict(counts), model_revisions={k:v for k,v in MODEL_REVISIONS.items() if not k.startswith('Qwen/')},
                                            component_success=component_counts,
                                            python=platform.python_version(), evidence_scope='real_source_images_frozen_model_predictions'))
        profiles = [dict(image_id=row['image_id'], costs=row.get('raw',{}).get('inference_costs'),
                         error=row.get('raw',{}).get('cost_profile_error')) for row in feature_rows
                    if 'inference_costs' in row.get('raw',{}) or 'cost_profile_error' in row.get('raw',{})]
        if require_two_gpus:
            profiles = [json.loads(line) for gpu in range(2)
                        for line in (output_dir/f'gpu{gpu}'/'inference_costs.jsonl').read_text().splitlines()]
        save('inference_cost_profiles.json',profiles)
        qwen_status = {'status': 'not_requested', 'count': 0}
        if qwen_images:
            candidates = sorted((row for row in rows if row['role'] == 'A'), key=lambda row: row['image_id'])
            rng = __import__('random').Random(20261022)
            rng.shuffle(candidates)
            selected = candidates[:int(qwen_images)]
            selection = [row['image_id'] for row in selected]
            selection_path = output_dir/'qwen_selection.json'
            if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
                raise ValueError('annotation selection changed during resume; create a new run')
            save('qwen_selection.json', selection)
            try:
                started = time.monotonic()
                budget_path = output_dir/'qwen_budget.json'
                previous_elapsed = json.loads(budget_path.read_text()).get('elapsed_seconds', 0) if budget_path.exists() else 0
                model_id = QWEN_MODEL
                reviewer = OllamaQwenReviewer(log_dir=output_dir/'ollama',require_two_gpus=require_two_gpus)
                count = 0
                proposal_path = output_dir/'qwen_proposals_ollama_4b.jsonl'
                prior = [json.loads(line) for line in proposal_path.read_text().splitlines()] if proposal_path.exists() else []
                complete = {row['image_id'] for row in prior if row.get('status') == 'ok'}
                with proposal_path.open('a', encoding='utf-8') as handle:
                    for row in selected:
                        if row['image_id'] in complete:
                            continue
                        if previous_elapsed+time.monotonic()-started > 4*3600:
                            break
                        with Image.open(images[row['image_id']]) as image:
                            assessment = reviewer.assess(canonical_image(image))
                        handle.write(json.dumps(dict(image_id=row['image_id'], role=row['role'], **assessment))+'\n')
                        handle.flush()
                        count += 1
                        if assessment.get('status') == 'ok':
                            complete.add(row['image_id'])
                        save('qwen_budget.json', dict(elapsed_seconds=previous_elapsed+time.monotonic()-started))
                valid = len(set(selection) & complete)
                qwen_status = {'status': 'completed' if valid == len(selection) else 'partial',
                               'requested': len(selection), 'valid': valid,
                               'attempted_this_run': count, 'prior_attempts': len(prior),
                               'annotation_tier': 'vlm_proposed', 'model_id': model_id}
            except Exception as error:
                qwen_status = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            finally:
                if 'reviewer' in locals():
                    reviewer.close()
        save('qwen_status.json', qwen_status)
        save('status.json', dict(stage='extract', status='completed_with_annotation_failure' if qwen_status['status'] in ('failed','partial') else 'completed', qwen=qwen_status))
        return result
    except Exception as error:
        save('status.json', dict(stage='extract', status='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-manifest',required=True)
    run_worker(parser.parse_args().worker_manifest)

