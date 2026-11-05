"""Auditable MEDIC study helpers. Real data required; no synthetic fallback.

Public labels establish damage ordering, not the novelty/age preference tradeoff.
All run outputs are written to an explicit caller-owned artifact directory.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import pandas as pd

from Archived.owners_before_simplification_20261027.SemParam.disaster_four_parameters import (
    CLASSES, FEATURE_NAMES, DisasterRequest, LocalNoveltyWindow, predicted_importance,
)
from Archived.owners_before_simplification_20261027.SemParam.disaster_evidence_header import (
    DisasterEvidenceHeader, EVIDENCE_FEATURE_NAMES, EvidenceNoveltyWindow,
    HUMANITARIAN_CLASSES,
)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def provenance(output_dir):
    import importlib.metadata
    import torch
    versions = {}
    for package in ("torch", "numpy", "pandas", "transformers", "scikit-learn", "lightgbm", "compressai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    data = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "cuda": torch.cuda.is_available(), "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "claim_scope": "public-label damage ordering; contextual novelty/age preference not established"}
    save_json(Path(output_dir) / "environment.json", data)
    return data


def label_code(value, names):
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return names.get(key, -1)


def load_medic(root, output_dir, caps=None):
    """Audit native multiclass TSVs; protect test hashes from train leakage."""
    root, output = Path(root).resolve(), Path(output_dir)
    caps = caps or {"train": 6000, "validation": 1000, "test": 1000}
    files = {}
    for split, filename in (("train", "MEDIC_train.tsv"), ("validation", "MEDIC_dev.tsv"), ("test", "MEDIC_test.tsv")):
        matches = sorted(root.rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"Need exactly one {filename}, found {len(matches)} in {root}")
        files[split] = matches[0]
    info_labels = {"informative": 1, "not_informative": 0}
    damage_labels = {"little_or_none": 0, "little_or_no_damage": 0, "mild": 1, "mild_damage": 1,
                     "severe": 2, "severe_damage": 2}
    humanitarian_labels = {label: i for i, label in enumerate(HUMANITARIAN_CLASSES)}
    class_labels = {label: i for i, label in enumerate(CLASSES[:-1])}
    rows, excluded = [], []
    manifest_hashes = {}
    for split, file in files.items():
        manifest_hashes[split] = digest(file)
        table = pd.read_csv(file, sep="\t", dtype=str).fillna("")
        required = {"image_path", "informative", "damage_severity", "disaster_types", "humanitarian"}
        if not required <= set(table):
            raise ValueError(f"Unexpected MEDIC schema: {list(table)}; no inferred label fallback")
        # Candidate choice is deterministic and label-independent.
        table["_order"] = table.image_path.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
        table = table.sort_values("_order")
        valid_images = 0
        for record in table.to_dict("records"):
            if valid_images >= caps[split]:
                break
            relative = Path(record["image_path"].replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe image path in dataset")
            candidates = [file.parent / relative, file.parent / "data" / relative, root / relative, root / "data" / relative]
            found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
            if not found:
                excluded.append({"split": split, "image_path": str(relative), "reason": "missing_image"})
                continue
            path = found[0]
            if not path.is_relative_to(root):
                raise ValueError("Image resolved outside dataset root")
            image_hash = digest(path)
            y_info = label_code(record["informative"], info_labels)
            y_sev = label_code(record["damage_severity"], damage_labels)
            y_class = label_code(record["disaster_types"], class_labels)
            y_humanitarian = label_code(record["humanitarian"], humanitarian_labels)
            # Noninformative + damage>little is a flagged conflict, not silently resolved.
            conflict = y_info == 0 and y_sev > 0
            grade = -1 if conflict else (0 if y_info == 0 else y_sev + 1 if y_info == 1 and y_sev >= 0 else -1)
            rows.append({"image_id": image_hash, "image_path": str(path), "source_split": split,
                         "event_name": record.get("event_name", ""), "y_info": y_info, "y_severity": y_sev,
                         "y_class": y_class, "y_humanitarian": y_humanitarian,
                         "raw_humanitarian_label": record["humanitarian"],
                         "grade": grade, "label_conflict": conflict})
            valid_images += 1
    if not rows:
        raise ValueError("No actual dataset images found")
    frame = pd.DataFrame(rows)
    contradictory_hashes = set()
    for image_id, group in frame.groupby("image_id"):
        if any(group.loc[group[column] >= 0, column].nunique() > 1
               for column in ("y_info", "y_severity", "y_class", "y_humanitarian")):
            contradictory_hashes.add(image_id)
    frame.loc[frame.image_id.isin(contradictory_hashes), "label_conflict"] = True
    frame.loc[frame.image_id.isin(contradictory_hashes), "grade"] = -1
    frame["_priority"] = frame.source_split.map({"test": 0, "validation": 1, "train": 2})
    before = len(frame)
    frame = frame.sort_values("_priority").drop_duplicates("image_id").drop(columns="_priority")
    frame["role"] = frame.source_split
    for index in frame.index[frame.source_split == "train"]:
        fold = int(frame.at[index, "image_id"][:8], 16) % 100
        frame.at[index, "role"] = "parameter_fit" if fold < 60 else "parameter_calibration" if fold < 70 else "ranking_train"
    frame = frame.sort_values(["role", "image_id"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "manifest.csv", index=False)
    pd.DataFrame(excluded, columns=["split", "image_path", "reason"]).to_csv(output / "excluded.csv", index=False)
    audit = {"manifest_sha256": manifest_hashes, "image_count": len(frame), "role_counts": frame.role.value_counts().to_dict(),
             "exact_duplicates_removed": before - len(frame), "label_conflicts": int(frame.label_conflict.sum()),
             "duplicate_hash_label_disagreements": len(contradictory_hashes),
             "label_counts": {name: frame[name].value_counts().to_dict() for name in ("y_info", "y_severity", "y_class", "y_humanitarian", "grade")},
             "humanitarian_taxonomy": list(HUMANITARIAN_CLASSES),
             "humanitarian_unmapped_count": int((frame.y_humanitarian < 0).sum()),
             "humanitarian_raw_counts": frame.raw_humanitarian_label.value_counts().to_dict(),
             "near_duplicate_audit": "NOT completed; exact hash protection only", "missing_images": len(excluded),
             "event_held_out": False, "dataset": "MEDIC native multiclass tables", "real_UE_timestamps": False}
    save_json(output / "dataset_audit.json", audit)
    for role in ("parameter_fit", "parameter_calibration", "ranking_train", "validation", "test"):
        if int((frame.role == role).sum()) < 10:
            raise ValueError(f"Insufficient actual images in {role}; inspect dataset_audit.json")
    return frame


def cache_clip_embeddings(frame, output_dir, model_name="openai/clip-vit-base-patch32", revision=None, batch_size=32):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from huggingface_hub import model_info
    output = Path(output_dir)
    revision = revision or model_info(model_name).sha
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name, revision=revision).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name, revision=revision, use_fast=False)
    rows, timings = [], []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size]
        images = []
        for path in batch.image_path:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        clock = time.perf_counter()
        with torch.inference_mode():
            values = model.get_image_features(**inputs)
            if not isinstance(values, torch.Tensor):
                values = values.image_embeds
            values = torch.nn.functional.normalize(values.float(), dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.extend([(time.perf_counter() - clock) * 1000 / len(images)] * len(images))
        rows.append(values.cpu().numpy())
        # Each batch is durable, allowing recovery even if a later batch fails.
        np.savez_compressed(output / f"clip_batch_{start:06d}.npz", image_ids=batch.image_id.to_numpy(dtype=str), embeddings=rows[-1])
        print(f"CLIP {min(start + len(images), len(frame))}/{len(frame)}", flush=True)
    embeddings = np.concatenate(rows)
    np.savez_compressed(output / "embeddings.npz", embeddings=embeddings, image_ids=frame.image_id.to_numpy(dtype=str),
                        extraction_ms=np.asarray(timings))
    save_json(output / "clip_provenance.json", {"model": model_name, "revision": revision,
              "device": device, "timing": "batched throughput per image, not measured UE latency", "frozen": True})
    return embeddings


def _checkpoint(path, model, optimizer, epoch, seed, **extra):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "seed": seed,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], **extra}, temporary)
    temporary.replace(path)


def fit_parameter_heads(embeddings, frame, output_dir, seed=11, epochs=30, batch_size=256):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = nn.Linear(embeddings.shape[1], 16).to(device)  # info 2 + damage 3 + type 7 + humanitarian 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    labels = torch.tensor(frame[["y_info", "y_severity", "y_class", "y_humanitarian"]].to_numpy(), device=device)
    train = np.flatnonzero(frame.role == "parameter_fit")
    calibration = np.flatnonzero(frame.role == "parameter_calibration")

    def loss_for(indices):
        logits, y = model(x[indices]), labels[indices]
        terms = []
        for column, left, right in ((0, 0, 2), (1, 2, 5), (2, 5, 12), (3, 12, 16)):
            valid = y[:, column] >= 0
            if column == 1:
                valid = valid & (y[:, 0] == 1)  # conditional severity, not an independence assumption
            if valid.any():
                terms.append(nn.functional.cross_entropy(logits[valid, left:right], y[valid, column]))
        if not terms:
            return None
        return sum(terms)

    best, wait, history, best_weights = float("inf"), 0, [], None
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for indices in np.array_split(generator.permutation(train), max(1, int(np.ceil(len(train) / batch_size)))):
            optimizer.zero_grad()
            loss = loss_for(indices)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            value = loss_for(calibration)
            if value is None:
                raise ValueError("Calibration partition contains no supported labels")
            value = float(value)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "calibration_loss": value})
        _checkpoint(output / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, seed)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if value < best - 1e-6:
            best, wait = value, 0
            best_weights = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            _checkpoint(output / "best.pt", model, optimizer, epoch, seed)
        else:
            wait += 1
        if wait >= 5:
            break
    if best_weights is None:
        raise RuntimeError("No finite parameter checkpoint")
    model.load_state_dict(best_weights)
    with torch.no_grad():
        logits = model(x).cpu().numpy()
    temperatures, probs = {}, []
    for name, column, left, right in (("info", 0, 0, 2), ("severity", 1, 2, 5),
                                     ("class", 2, 5, 12), ("humanitarian", 3, 12, 16)):
        y = frame[["y_info", "y_severity", "y_class", "y_humanitarian"]].to_numpy()[:, column]
        mask = (frame.role.to_numpy() == "parameter_calibration") & (y >= 0)
        if name == "severity":
            mask &= frame.y_info.to_numpy() == 1
        if not mask.any():
            raise ValueError(f"No calibration targets for {name}")
        options = []
        for temp in (.5, .75, 1., 1.5, 2., 3.):
            p = _softmax(logits[:, left:right] / temp)
            options.append((-np.log(np.maximum(p[mask, y[mask]], 1e-12)).mean(), temp, p))
        _, temp, p = min(options, key=lambda item: item[0])
        temperatures[name] = temp
        probs.append(p)
    importance = predicted_importance(probs[0][:, 1], probs[1])
    # Abstention threshold is validation-independent: report coverage/accuracy at this declared operating point.
    class_ids = np.where(probs[2].max(1) >= .6, probs[2].argmax(1), 7)
    np.savez_compressed(output / "predictions.npz", importance=importance, info=probs[0], severity=probs[1],
                        semantic_class=class_ids, class_probabilities=probs[2], humanitarian=probs[3],
                        image_ids=frame.image_id.to_numpy(dtype=str))
    save_json(output / "training_provenance.json", {"seed": seed, "device": device, "temperatures": temperatures,
              "fit_ids": frame.iloc[train].image_id.tolist(), "calibration_ids": frame.iloc[calibration].image_id.tolist(),
              "importance_definition": "P(info)*E[damage grade | info]/3", "class_abstention_threshold": .6})
    return importance, class_ids


def _softmax(values):
    values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def prepare_ranking_rows(frame, embeddings, importance, class_ids, output_dir, seed=42, query_size=32,
                         evidence_prediction_npz=None):
    """Declared synthetic UE streams. No invented urgency/preference labels."""
    evidence = evidence_prediction_npz is not None
    generator = np.random.default_rng(seed)
    history = EvidenceNoveltyWindow(20) if evidence else LocalNoveltyWindow(20)
    if evidence:
        with np.load(evidence_prediction_npz, allow_pickle=False) as saved:
            if saved["image_ids"].tolist() != frame.image_id.tolist():
                raise ValueError("Evidence predictions do not align with audited manifest")
            p_info = saved["info"][:, 1]
            p_damage = saved["severity"]
            p_humanitarian = saved["humanitarian"]
    records, features = [], []
    for role in ("ranking_train", "validation", "test"):
        history.reset()
        selected = np.flatnonzero((frame.role == role) & (frame.grade >= 0))
        selected = generator.permutation(selected)
        for position, index in enumerate(selected):
            if position % query_size == 0:
                last_position = min(len(selected) - 1, (position // query_size + 1) * query_size - 1)
                snapshot = float(last_position * 1000 + generator.uniform(0, 30_000))
            ue = str(position % 5)
            capture = float(position * 1000)
            novelty = history.observe(ue, capture, embeddings[index])
            if evidence:
                request = DisasterEvidenceHeader(
                    float(importance[index]), novelty, capture, CLASSES[int(class_ids[index])],
                    float(p_info[index]), tuple(float(v) for v in p_damage[index]),
                    tuple(float(v) for v in p_humanitarian[index]),
                )
            else:
                request = DisasterRequest(float(importance[index]), novelty, capture, CLASSES[int(class_ids[index])])
            wire = request.serialize()
            parsed = DisasterEvidenceHeader.parse(wire) if evidence else DisasterRequest.parse(wire)
            row = {"image_id": frame.iloc[index].image_id, "frame_index": int(index), "role": role,
                   "query_id": f"{role}:{position // query_size}", "grade": int(frame.iloc[index].grade),
                   "ue_id": ue, "capture_ms": capture, "snapshot_ms": snapshot,
                   "request_json": wire.decode(), "request_bytes": len(wire),
                   "source_sequence": position, "scenario": "synthetic UE allocation/capture/age; labels native"}
            features.append(parsed.features(snapshot))
            records.append(row)
    data, x = pd.DataFrame(records), np.asarray(features)
    if not len(data):
        raise ValueError("No supported ranking grades")
    output = Path(output_dir)
    data.to_csv(output / "ranking_rows.csv", index=False)
    np.savez_compressed(output / "ranking_features.npz", features=x,
                        names=np.asarray(EVIDENCE_FEATURE_NAMES if evidence else FEATURE_NAMES))
    return data, x


def ranking_metrics(grades, scores, query_ids):
    from sklearn.metrics import ndcg_score
    values = []
    query_ids = np.asarray(query_ids)
    for group in dict.fromkeys(query_ids):
        rows = np.flatnonzero(query_ids == group)
        if len(rows) < 2 or np.sum(np.asarray(grades)[rows]) == 0:
            continue
        g, s = np.asarray(grades)[rows], np.asarray(scores)[rows]
        values.append({"query_id": group, "ndcg5": float(ndcg_score([g], [s], k=5)),
                       "ndcg10": float(ndcg_score([g], [s], k=10))})
    return pd.DataFrame(values, columns=["query_id", "ndcg5", "ndcg10"])


def train_ranker_family(rows, features, output_dir, seed=11, epochs=30):
    """Native damage-grade experiment. Does NOT validate contextual preference."""
    import torch
    from torch import nn
    from Scheduler.disaster_rankers import ArrayScorer
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(rows.role == "ranking_train")
    validation = np.flatnonzero(rows.role == "validation")
    test = np.flatnonzero(rows.role == "test")
    if min(map(len, (train, validation, test))) < 2 or rows.iloc[train].grade.nunique() < 2:
        raise ValueError("Need independent ranking partitions and at least two training grades")
    mean, scale = features[train].mean(0), features[train].std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor((features - mean) / scale, dtype=torch.float32, device=device)
    y = torch.tensor(rows.grade.to_numpy(), dtype=torch.long, device=device)
    summaries, query_results, predictions = [], [], []
    for name in ("classifier", "pointwise_ordinal_regression", "linear_pairwise", "ranknet"):
        torch.manual_seed(seed)
        generator = np.random.default_rng(seed)
        width = features.shape[1]
        if name == "classifier":
            model = nn.Sequential(nn.Linear(width, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
        elif name == "linear_pairwise":
            model = nn.Sequential(nn.Linear(width, 1)).to(device)
        else:
            model = nn.Sequential(nn.Linear(width, 16), nn.ReLU(), nn.Linear(16, 1)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        destination = output / name
        destination.mkdir(exist_ok=True)
        best, wait, best_weights, history = -float("inf"), 0, None, []
        def scores():
            logits = model(x)
            if name == "classifier":
                return (logits.softmax(-1) * torch.arange(4, device=device)).sum(-1).detach().cpu().numpy()
            return logits.reshape(-1).detach().cpu().numpy()
        groups = [np.flatnonzero((rows.query_id == group) & (rows.role == "ranking_train"))
                  for group in rows.iloc[train].query_id.unique()]
        groups = [g for g in groups if len(g) > 1]
        if not groups:
            raise ValueError("No within-query training pairs")
        clock = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            losses = []
            batches = np.array_split(generator.permutation(train), max(1, int(np.ceil(len(train) / 256))))
            for batch in batches:
                optimizer.zero_grad()
                if name in ("ranknet", "linear_pairwise"):
                    pairs = [generator.choice(groups[int(generator.integers(len(groups)))], 2, replace=False) for _ in range(256)]
                    a, b = np.asarray(pairs).T
                    target = .5 + .5 * torch.sign((y[a] - y[b]).float())
                    loss = nn.functional.binary_cross_entropy_with_logits((model(x[a]) - model(x[b])).reshape(-1), target)
                elif name == "classifier":
                    loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
                else:
                    # Ordinal-grade regression only; numeric distances are an explicit baseline assumption.
                    loss = nn.functional.mse_loss(model(x[batch]).reshape(-1), y[batch].float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val_score = scores()
            metrics = ranking_metrics(rows.iloc[validation].grade.to_numpy(), val_score[validation], rows.iloc[validation].query_id)
            value = float(metrics.ndcg5.mean()) if len(metrics) else 0.
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_ndcg5": value})
            _checkpoint(destination / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, seed)
            pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
            if value > best + 1e-6:
                best, wait = value, 0
                best_weights = {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}
                _checkpoint(destination / "best.pt", model, optimizer, epoch, seed)
            else:
                wait += 1
            if wait >= 5:
                break
        model.load_state_dict(best_weights)
        model.eval()
        linear = [layer for layer in model if isinstance(layer, nn.Linear)]
        arrays = {"mean": mean, "scale": scale, "layers": np.asarray(len(linear))}
        for i, layer in enumerate(linear):
            arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
            arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
        if name == "classifier":
            arrays["classes"] = np.arange(4)
        np.savez_compressed(destination / "inference.npz", **arrays)
        with torch.no_grad():
            final = scores()
        exported = ArrayScorer(destination / "inference.npz")(features)
        np.testing.assert_allclose(exported, final, atol=2e-5, rtol=2e-5)
        result = ranking_metrics(rows.iloc[test].grade.to_numpy(), final[test], rows.iloc[test].query_id)
        result["policy"], result["seed"] = name, seed
        query_results.append(result)
        predictions.append(pd.DataFrame({"image_id": rows.image_id, "role": rows.role, "score": final, "policy": name, "seed": seed}))
        summaries.append({"policy": name, "seed": seed, "validation_ndcg5": best,
                          "test_ndcg5": float(result.ndcg5.mean()) if len(result) else None,
                          "training_seconds": time.perf_counter() - clock})
    # Boosted-tree contender: same rows, labels, groups, held-out partition and linear label gains.
    import lightgbm as lgb
    groups_train = rows.iloc[train].groupby("query_id", sort=False).size().to_numpy()
    groups_val = rows.iloc[validation].groupby("query_id", sort=False).size().to_numpy()
    destination = output / "lambdamart"
    destination.mkdir(exist_ok=True)
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=100, learning_rate=.05, num_leaves=7,
                          min_child_samples=10, random_state=seed, n_jobs=2, verbosity=-1, label_gain=[0, 1, 2, 3])
    def save_booster(env):
        env.model.save_model(str(destination / f"iteration_{env.iteration:03d}.txt"))
    model.fit(features[train], rows.iloc[train].grade, group=groups_train,
              eval_set=[(features[validation], rows.iloc[validation].grade)], eval_group=[groups_val], eval_at=[5, 10],
              callbacks=[lgb.early_stopping(10, verbose=False), save_booster])
    model.booster_.save_model(str(destination / "best.txt"), num_iteration=model.best_iteration_)
    final = model.predict(features)
    result = ranking_metrics(rows.iloc[test].grade.to_numpy(), final[test], rows.iloc[test].query_id)
    result["policy"], result["seed"] = "lambdamart", seed
    query_results.append(result)
    predictions.append(pd.DataFrame({"image_id": rows.image_id, "role": rows.role, "score": final, "policy": "lambdamart", "seed": seed}))
    val_result = ranking_metrics(rows.iloc[validation].grade.to_numpy(), final[validation], rows.iloc[validation].query_id)
    summaries.append({"policy": "lambdamart", "seed": seed, "validation_ndcg5": float(val_result.ndcg5.mean()),
                      "test_ndcg5": float(result.ndcg5.mean()) if len(result) else None})
    pd.DataFrame(summaries).to_csv(output / "model_summary.csv", index=False)
    pd.concat(query_results).to_csv(output / "per_query_metrics.csv", index=False)
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    save_json(output / "claim_boundary.json", {"supervision": "independent native damage/informativeness grades",
              "contextual_priority_labels": False, "novelty_age_tradeoff_validated": False,
              "synthetic_tests_are_research_results": False,
              "feature_names": (EVIDENCE_FEATURE_NAMES if features.shape[1] == len(EVIDENCE_FEATURE_NAMES) else FEATURE_NAMES),
              "rank_fit_ids": rows.iloc[train].image_id.tolist(), "validation_ids": rows.iloc[validation].image_id.tolist(),
              "test_ids": rows.iloc[test].image_id.tolist()})
    return pd.DataFrame(summaries)


def measure_test_payloads(frame, ranking_rows, output_dir, quality=3):
    from Codec.Models.scale_hyperprior_bundle import ScaleHyperpriorBundle
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # CPU entropy coding avoids assuming cross-platform bitstream reproducibility.
    codec, records = ScaleHyperpriorBundle(quality=quality, device="cpu"), []
    for index in ranking_rows.loc[ranking_rows.role == "test", "frame_index"].unique():
        row = frame.iloc[index]
        bundle = output / f"{row.image_id}.dhb"
        measurement = codec.roundtrip(row.image_path, bundle)
        records.append({"image_id": row.image_id, **measurement, "bundle_sha256": digest(bundle)})
        # Incremental measurement log persists even if the next codec operation fails.
        pd.DataFrame(records).to_json(output / "payload_measurements.jsonl", orient="records", lines=True)
    return pd.DataFrame(records)


def evaluate_service_replay(ranking_rows, payloads, model_dir, output_dir, seed=11, workloads=5, max_jobs=128):
    """One-shot digital, ideal link. Timing is a declared simulation, not MAC/PHY evidence."""
    from Scheduler.disaster_rankers import ArrayScorer, candidate_first_safe_mask
    from Scheduler.training.disaster_trace import LOAD_FACTORS, construct_trace, synthetic_lifetimes_from_ready
    import lightgbm as lgb
    model_dir, output = Path(model_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = ranking_rows[ranking_rows.role == "test"].merge(payloads, on="image_id", validate="one_to_one")
    if len(frame) < 2:
        raise ValueError("Measured test payloads required; no random byte-length substitute")
    scorers = {name: ArrayScorer(model_dir / name / "inference.npz") for name in (
        "classifier", "pointwise_ordinal_regression", "linear_pairwise", "ranknet")}
    booster = lgb.Booster(model_file=str(model_dir / "lambdamart" / "best.txt"))
    scorers["lambdamart"] = booster.predict
    if not 1 <= workloads <= len(LOAD_FACTORS):
        raise ValueError("Workloads must select the declared load grid")
    policies = ["fifo", "round_robin", "oldest_first", "sjf", "importance_only", "value_per_time", "fixed_formula", *scorers]
    summaries, curves = [], []
    control_rate = 100_000.  # explicitly separate logical control channel, bits/s
    for workload in range(workloads):
        generator = np.random.default_rng(1000 + workload)
        selected = frame.iloc[generator.permutation(len(frame))[:max_jobs]]
        # Choose the offered set first. Ordering by its declared source sequence
        # preserves the UE-local novelty history used by the old four-field pilot.
        selected = selected.sort_values("source_sequence").reset_index(drop=True)
        for mode in ("static", "dynamic"):
            requests = [DisasterEvidenceHeader.parse(value.encode()) if '"schema_version"' in value
                        else DisasterRequest.parse(value.encode()) for value in selected.request_json]
            bits = selected.payload_bits.to_numpy(float)
            # Policy-independent rate chosen to expose queueing; not a measured deployed link.
            data_rate = float(np.median(bits)) / 1.2
            service = bits / data_rate * 1000 + selected.decode_ms.to_numpy(float) + (2 * 24 * 8 / control_rate * 1000)
            preparation = selected.encode_ms.to_numpy(float)
            initial_header_bits = selected.request_bytes.to_numpy(float) * 8
            initial_trace = construct_trace(service, preparation, initial_header_bits, selected.ue_id.to_numpy(),
                                    mode=mode, load_factor=LOAD_FACTORS[workload],
                                    seed=1000 + workload, control_rate_bps=control_rate)
            captures = initial_trace["capture_ms"]
            for i, request in enumerate(requests):
                requests[i] = replace(request, capture_time_ms=float(captures[i]))
            # Retiming changes JSON timestamp length. Recompute control demand
            # from the *exact serialized headers subsequently parsed by policy*.
            header_bits = np.asarray([len(request.serialize()) * 8 for request in requests], float)
            trace = construct_trace(service, preparation, header_bits, selected.ue_id.to_numpy(),
                                    mode=mode, load_factor=LOAD_FACTORS[workload],
                                    seed=1000 + workload, control_rate_bps=control_rate)
            if not np.array_equal(trace["capture_ms"], captures):
                raise AssertionError("Header accounting changed capture generation")
            ready = trace["ready_ms"]
            save_json(output / f"trace_seed{seed}_{mode}_workload{workload}.json",
                      {**{key: value for key, value in trace.items() if key not in ("capture_ms", "ready_ms")},
                       "capture_ms": captures.tolist(), "ready_ms": ready.tolist(),
                       "offered_image_ids": selected.image_id.tolist(),
                       "source_local_history_may_include_unoffered_observations": True})
            horizon = float(ready.max() + service.sum())
            for guard in (False, True):
                # Deadline arm is separate; primary ordering arm has no assumed real deadline.
                # Synthetic lifetime is independent of grade. Static expiration
                # starts at batch release; no job is pre-expired by construction.
                deadlines = synthetic_lifetimes_from_ready(ready, service) if guard else np.full(len(selected), np.inf)
                for policy in policies + (["edf"] if guard else []):
                    pending = set(range(len(selected)))
                    window_start = float(ready.min()) if mode == "static" else 0.
                    now, utility, area, last_ue = window_start, 0., 0., None
                    events, curve, decision_times = [], [], []
                    total_grade = float(selected.grade.sum())
                    while pending:
                        available = sorted(i for i in pending if ready[i] <= now)
                        if not available:
                            next_time = min(ready[i] for i in pending)
                            area += utility * (next_time - now)
                            now = next_time
                            continue
                        start = time.perf_counter()
                        mask, overload = candidate_first_safe_mask(service[available], deadlines[available], now) if guard else (np.ones(len(available), bool), False)
                        allowed = [i for k, i in enumerate(available) if mask[k]]
                        if not allowed:
                            for i in available:
                                pending.remove(i)
                                events.append({"image_id": selected.iloc[i].image_id, "status": "deadline_infeasible", "at_ms": now,
                                               "grade": int(selected.iloc[i].grade), "overload": bool(overload)})
                            continue
                        x = np.stack([type(requests[i]).parse(requests[i].serialize()).features(now) for i in allowed])
                        if policy in scorers:
                            values = np.asarray(scorers[policy](x))
                        elif policy == "importance_only":
                            values = x[:, 0]
                        elif policy == "value_per_time":
                            values = x[:, 0] / service[allowed]
                        elif policy == "fixed_formula":
                            values = x[:, 0] + .25 * x[:, 1] + .05 * x[:, 2]
                        elif policy == "sjf":
                            values = -service[allowed]
                        elif policy == "oldest_first":
                            values = -captures[allowed]
                        elif policy == "edf":
                            values = -deadlines[allowed]
                        elif policy == "round_robin":
                            users = sorted(set(selected.iloc[i].ue_id for i in allowed))
                            next_ue = next((u for u in users if last_ue is None or u > last_ue), users[0])
                            values = np.asarray([1. if selected.iloc[i].ue_id == next_ue else 0. for i in allowed])
                            last_ue = next_ue
                        else:
                            values = -ready[allowed]
                        winner = min(range(len(allowed)), key=lambda k: (-float(values[k]), ready[allowed[k]], selected.iloc[allowed[k]].image_id))
                        chosen = allowed[winner]
                        decision_times.append((time.perf_counter() - start) * 1000)
                        finish = now + service[chosen]
                        area += utility * (finish - now)
                        utility += float(selected.iloc[chosen].grade)
                        events.append({"image_id": selected.iloc[chosen].image_id, "ue_id": selected.iloc[chosen].ue_id,
                                       "status": "decoded", "start_ms": now, "finish_ms": finish,
                                       "capture_ms": captures[chosen], "ready_ms": ready[chosen],
                                       "score": float(values[winner]), "grade": int(selected.iloc[chosen].grade),
                                       "deadline_met": bool(finish <= deadlines[chosen]), "overload": bool(overload)})
                        now = finish
                        curve.append({"time_ms": now, "grade_fraction": utility / total_grade if total_grade else 0.})
                        pending.remove(chosen)
                    area += utility * max(0., horizon - now)
                    completed = [row for row in events if row["status"] == "decoded"]
                    name = f"seed{seed}_{mode}_deadline{int(guard)}_workload{workload}_{policy}"
                    pd.DataFrame(events).to_csv(output / f"{name}.csv", index=False)
                    curves.extend({**point, "seed": seed, "mode": mode, "deadline_arm": guard, "workload": workload, "policy": policy} for point in curve)
                    summaries.append({"seed": seed, "mode": mode, "deadline_arm": guard, "workload": workload, "policy": policy,
                                      "grade_delivery_auc": area / ((horizon - window_start) * total_grade) if total_grade else None,
                                      "zero_grade_workload": total_grade == 0,
                                      "deadline_miss_rate": 1 - len(completed) / len(selected),
                                      "p95_queue_ms": float(np.percentile([r["start_ms"] - r["ready_ms"] for r in completed], 95)) if completed else None,
                                      "decision_ms_mean": float(np.mean(decision_times)) if decision_times else None,
                                      "actual_payload_bits": int(bits.sum()), "data_rate_bps": data_rate,
                                      "actual_header_bits": int(header_bits.sum()),
                                      "target_load_factor": LOAD_FACTORS[workload],
                                      "realized_load_factor": trace["realized_load_factor"],
                                      "fifo_max_ready_backlog": trace["fifo_max_ready_backlog"],
                                      "timing_scope": "codec-time surrogate; VLM and controller computation excluded from replay clock; no physical radio claim"})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "service_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "service_curves.csv", index=False)
    save_json(output / "replay_limits.json", {"model": "one-shot, ideal delivery, nonpreemptive complete bundles",
              "control_channel": "logical separate channel; request contention not modeled",
              "capture_age_and_novelty": "selected jobs retimed after sampling; UE-local novelty may include unoffered observations; not real UE sequence", "VLM_latency_included": False,
              "controller_latency_included_in_clock": False,
              "controller_latency_reporting": "measured separately as decision_ms_mean; queue and delivery metrics exclude it",
              "deadline_arm": "declared sensitivity only", "contextual_human_preferences": "not supplied by public damage labels"})
    return summary


def download_medic_archive(url, archive_path, chunk_bytes=64 * 1024 * 1024,
                           retries=5, backoff_seconds=1, timeout=(30, 180)):
    """Resume identity-checked bounded ranges; SHA256 is local integrity, not authenticity.

    The .part and identity JSON survive failures. Never discard a conflicting
    resource or corrupt cache automatically: the caller must inspect those files.
    """
    import requests
    import shutil
    import os
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    part = archive_path.with_name(archive_path.name + ".part")
    metadata = archive_path.with_name(archive_path.name + ".download.json")
    if chunk_bytes < 1 or retries < 0:
        raise ValueError("Invalid download chunk/retry settings")

    def persist(data):
        temporary = metadata.with_name(metadata.name + ".tmp")
        save_json(temporary, data)
        temporary.replace(metadata)

    state = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None
    if state and state.get("url") != url:
        raise ValueError("Download URL conflicts with retained identity metadata")
    if archive_path.exists():
        if (not state or not state.get("complete") or
                archive_path.stat().st_size != state["total_bytes"] or
                digest(archive_path) != state.get("sha256")):
            raise ValueError("Completed MEDIC cache failed size/SHA256 verification; retained for inspection")
        return archive_path, state
    if part.exists() and not state:
        raise ValueError("Partial MEDIC archive has no identity metadata; retained for inspection")
    if state and state.get("complete"):
        raise ValueError("Completed archive is missing; inspect retained download metadata")
    failures = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if state and offset > state["total_bytes"]:
            raise ValueError("Partial archive exceeds declared resource size")
        if state and offset == state["total_bytes"]:
            state.update(sha256=digest(part), complete=True, downloaded_bytes=offset,
                         hash_scope="local integrity only; no publisher checksum supplied")
            # Publish data before completion metadata; interrupted publication fails safely.
            part.replace(archive_path)
            persist(state)
            return archive_path, state
        end = offset + chunk_bytes - 1
        if state:
            end = min(end, state["total_bytes"] - 1)
        headers = {"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}
        if state:
            headers["If-Range"] = state["etag"]
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code != 206:
                    raise ValueError(f"Server did not honor MEDIC Range request (HTTP {response.status_code}); no bytes appended")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match:
                    raise ValueError("Missing or invalid Content-Range")
                first, last, total = map(int, match.groups())
                etag = response.headers.get("ETag")
                if (first != offset or last != min(end, total - 1) or total <= last or
                        not etag or etag.startswith("W/") or
                        response.headers.get("Content-Encoding", "identity") != "identity" or
                        response.headers.get("Content-Length") != str(last - first + 1)):
                    raise ValueError("Inconsistent MEDIC range, length, encoding or strong ETag")
                if state and (etag != state["etag"] or total != state["total_bytes"]):
                    raise ValueError("MEDIC resource identity changed; partial bytes retained, no bytes appended")
                if not state:
                    state = {"url": url, "etag": etag, "total_bytes": total, "complete": False,
                             "last_modified": response.headers.get("Last-Modified"), "downloaded_bytes": 0}
                    persist(state)
                remaining = total - offset
                if shutil.disk_usage(archive_path.parent).free < remaining:
                    raise OSError(f"Insufficient disk space for MEDIC: need {remaining} additional bytes")
                received = 0
                with part.open("ab") as stream:
                    try:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if received + len(block) > last - first + 1:
                                raise ValueError("MEDIC response exceeds requested range")
                            stream.write(block)
                            received += len(block)
                    finally:
                        stream.flush()
                        os.fsync(stream.fileno())
                        state["downloaded_bytes"] = stream.tell()
                        persist(state)
                if received != last - first + 1:
                    raise requests.exceptions.ChunkedEncodingError("Short MEDIC range response")
                failures = 0
                print(f"MEDIC downloaded {state['downloaded_bytes'] / 2**30:.3f}/{total / 2**30:.3f} GiB", flush=True)
        except requests.exceptions.RequestException as error:
            failures += 1
            if failures > retries:
                raise RuntimeError(f"MEDIC download interrupted; resumable bytes retained at {part}") from error
            print(f"MEDIC retry {failures}/{retries}; resuming retained bytes: {error}", flush=True)
            time.sleep(min(backoff_seconds * 2 ** (failures - 1), 30))


def fetch_medic_subset(destination, sample_modulus=5, url="https://crisisnlp.qcri.org/data/MEDIC/MEDIC.tar.gz",
                       archive_cache_dir=None):
    """Cache archive outside extraction output, then retain deterministic sample.

    On Kaggle pass /kaggle/temp/medic-download to avoid publishing the archive
    as a working-directory output. Scratch partials survive calls, not fresh VMs.
    """
    import tarfile
    import shutil
    import gzip
    destination = Path(destination).resolve()
    if sample_modulus < 1:
        raise ValueError("Invalid sampling modulus")
    destination.mkdir(parents=True, exist_ok=True)
    count, saved_bytes = 0, 0
    cache_root = Path(archive_cache_dir) if archive_cache_dir is not None else destination.parent / ".medic_download_cache"
    cache = cache_root / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".tar.gz")
    archive_path, download = download_medic_archive(url, cache)
    # Reserve space for the retained sample; model/payload requirements remain
    # caller-dependent and cannot be inferred from compressed archive size.
    sample_reserve = max(3 * 1024**3, download["total_bytes"] // sample_modulus)
    if shutil.disk_usage(destination).free < sample_reserve:
        raise OSError(f"Insufficient disk space for MEDIC extraction: reserve {sample_reserve} bytes plus model/run outputs")
    log = dict(download, archive_path=str(archive_path), download_complete=True, extraction_complete=False,
               sample_rule=f"sha256(archive member name) modulo {sample_modulus} == 0")
    save_json(destination / "download_provenance.json", log)
    with gzip.open(archive_path, "rb") as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("Unsafe archive member")
                metadata = name.name in {"MEDIC_train.tsv", "MEDIC_dev.tsv", "MEDIC_test.tsv"} or name.suffix.lower() == ".txt"
                image = name.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                selected = int(hashlib.sha256(member.name.encode()).hexdigest()[:12], 16) % sample_modulus == 0
                if not metadata and not (image and selected):
                    continue
                if member.size > 100 * 1024 * 1024:
                    raise ValueError("Unexpectedly large retained archive member")
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Archive path escaped destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
                count += 1
                saved_bytes += member.size
                if count % 500 == 0:
                    log.update(retained_members=count, retained_bytes=saved_bytes)
                    save_json(destination / "download_provenance.json", log)
                    print(f"MEDIC retained {count} members, {saved_bytes / 1e9:.2f} GB", flush=True)
        # tar EOF precedes gzip trailer; explicitly drain to validate CRC/truncation.
        while decompressed.read(1024 * 1024):
            pass
    log.update(extraction_complete=True, gzip_crc_verified=True, retained_members=count, retained_bytes=saved_bytes)
    save_json(destination / "download_provenance.json", log)
    return destination


def extractor_metrics(frame, prediction_npz, output_dir):
    """Record classifier error/calibration diagnostics independently of ranking."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
    with np.load(prediction_npz, allow_pickle=False) as saved:
        probabilities = {name: saved[key] for name, key in (("info", "info"), ("severity", "severity"), ("class", "class_probabilities"))}
        if "humanitarian" in saved and "y_humanitarian" in frame:
            probabilities["humanitarian"] = saved["humanitarian"]
    results = []
    tasks = [("info", "y_info"), ("severity", "y_severity"), ("class", "y_class")]
    if "humanitarian" in probabilities:
        tasks.append(("humanitarian", "y_humanitarian"))
    for name, label in tasks:
        mask = (frame.role.to_numpy() == "test") & (frame[label].to_numpy() >= 0)
        if name == "severity":
            mask &= frame.y_info.to_numpy() == 1
        p, y = probabilities[name][mask], frame.loc[mask, label].to_numpy()
        if not len(y):
            continue
        labels = np.arange(p.shape[1])
        result = {"task": name, "n": len(y), "accuracy": accuracy_score(y, p.argmax(1)),
                        "macro_f1": f1_score(y, p.argmax(1), labels=labels, average="macro", zero_division=0),
                        "class_recall": recall_score(y, p.argmax(1), labels=labels, average=None, zero_division=0).tolist(),
                        "log_loss": log_loss(y, p, labels=labels)}
        if name == "class":
            # Match request generation's deployed confidence gate, not just argmax.
            accepted = p.max(1) >= .6
            predictions = p.argmax(1)
            accepted_accuracy = float(accuracy_score(y[accepted], predictions[accepted])) if accepted.any() else None
            deployed_predictions = np.where(accepted, predictions, -1)
            result.update({"abstention_threshold": .6,
                           "accepted_count": int(accepted.sum()),
                           "accepted_coverage": float(accepted.mean()),
                           "unknown_rate": float((~accepted).mean()),
                           "accepted_accuracy": accepted_accuracy,
                           "accepted_error_rate": 1 - accepted_accuracy if accepted_accuracy is not None else None,
                           "deployed_class_recall_including_abstention": recall_score(
                               y, deployed_predictions, labels=labels, average=None, zero_division=0).tolist()})
        results.append(result)
    save_json(Path(output_dir) / "extractor_test_metrics.json", results)
    return results


def render_study_outputs(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    models = pd.concat(pd.read_csv(p) for p in sorted(root.glob("seed_*/rankers/model_summary.csv")))
    replay = pd.concat(pd.read_csv(p) for p in sorted(root.glob("seed_*/replay/service_summary.csv")))
    models.to_csv(root / "ranking_comparison.csv", index=False)
    replay.to_csv(root / "service_comparison.csv", index=False)
    summary = models.groupby("policy").agg(validation_ndcg5=("validation_ndcg5", "mean"), test_ndcg5=("test_ndcg5", "mean"),
                                            seed_std=("test_ndcg5", "std"), seeds=("seed", "nunique")).reset_index()
    summary.to_csv(root / "comparison_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary.policy, summary.test_ndcg5, yerr=summary.seed_std.fillna(0), capsize=3)
    ax.set_ylabel("Held-out damage nDCG@5")
    ax.set_title("Public-label ranking; bars: seed variation, not population confidence")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(root / "01_ranking_comparison.png", dpi=180)
    plt.close(fig)
    subset = replay[(replay["mode"] == "dynamic") & ~replay.deadline_arm]
    grouped = subset.groupby("policy").grade_delivery_auc.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(grouped.index, grouped.values)
    ax.set_xlabel("Grade-weighted delivery AUC")
    ax.set_title("Dynamic digital-bundle replay: declared timing model")
    fig.tight_layout()
    fig.savefig(root / "02_dynamic_delivery.png", dpi=180)
    plt.close(fig)
    # Choose by validation only. A test winner is not silently selected as the final algorithm.
    selected = summary.sort_values(["validation_ndcg5", "policy"], ascending=[False, True]).iloc[0].policy
    save_json(root / "selection.json", {"validation_selected_method": selected,
              "criterion": "mean validation nDCG@5, declared alphabetical tie break",
              "test_used_for_selection": False, "novelty_proven": False,
              "final_four_parameter_contextual_priority_validated": False})
    return summary
