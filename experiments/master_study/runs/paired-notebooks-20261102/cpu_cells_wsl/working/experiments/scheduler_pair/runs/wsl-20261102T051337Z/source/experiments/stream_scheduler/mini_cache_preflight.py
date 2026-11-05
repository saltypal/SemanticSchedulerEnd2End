"""WSL mini-pool integrity and received-feature preflight; no teacher or model inference."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tarfile
import time
import types
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contract(root):
    # Only pure contract modules; upstream package __init__ imports vision stacks.
    package = types.ModuleType("_mini_preflight_contract")
    package.__path__ = [str(root / "parameters")]
    sys.modules[package.__name__] = package
    name = package.__name__ + ".scene_arrays"
    spec = importlib.util.spec_from_file_location(name, root / "parameters/scene_arrays.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    log = (out / "run.log").open("w", buffering=1)
    start = time.perf_counter()
    def emit(message):
        line = f"{time.perf_counter()-start:.2f}s {message}"
        print(line, flush=True)
        log.write(line + "\n")
    files = [root / "parameters/scene_arrays.py", root / "parameters/scene_vocabulary.py"]
    before = {str(p): sha(p.read_bytes()) for p in files}
    manifest = {"scope": "real cache integrity + received-feature export; development only",
                "interpreter": sys.executable, "python": platform.python_version(),
                "platform": platform.platform(), "numpy": np.__version__,
                "source_hashes": before, "script_sha256": sha(Path(__file__).read_bytes()),
                "archive": str(args.archive.resolve()), "output": str(out)}
    write(out / "manifest.json", manifest)
    emit("START archive SHA256 verification")
    h = hashlib.sha256()
    with args.archive.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    archive_sha = h.hexdigest()
    receipt_path = args.archive.parent / "Mini_dataset_6000_receipt.json"
    expected = json.loads(receipt_path.read_text())["sha256"].lower()
    if archive_sha != expected:
        raise ValueError("archive checksum mismatch")
    emit("PASS archive checksum; inspecting metadata without extracting images")
    m = contract(root)
    with tarfile.open(args.archive) as tar:
        members = {x.name: x for x in tar.getmembers()}
        def read(name):
            return tar.extractfile("Mini_dataset/" + name).read()
        rows = [json.loads(x) for x in read("handoff/image_parameters.jsonl").splitlines() if x.strip()]
        index = json.loads(read("portable_image_index.json"))
        refs = json.loads(read("handoff/reference_records.json"))
        training = json.loads(read("student/training_records.json"))
        registry = json.loads(read("scene/registry.json"))
        if registry != m.REGISTRY:
            raise ValueError("cached registry differs from current pure contract")
        byid = {r["image_id"]: r for r in index}
        refids = {r["image_id"] for r in refs}
        if len(rows) != 6000 or len(index) != 6000 or len(byid) != 6000 or len(refids) != 6000:
            raise ValueError("expected 6000 unique aligned image/index/reference rows")
        exposed_hashes = {r["image_sha256"] for r in training if r["split"] == "train"}
        features, ids, provenance, counts = [], [], [], Counter()
        groups, hashes, producer_ids, sparse_sizes = {}, {}, set(), []
        levels = ("Critical", "High", "Medium", "Low")
        for i, row in enumerate(rows):
            ident = row["image_id"]
            if ident in ids or ident not in refids or ident not in byid:
                raise ValueError("duplicate or unmatched handoff identity")
            record = byid[ident]
            relative = Path(record["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe archive image index")
            member = "Mini_dataset/" + relative.as_posix()
            raw_hash = sha(tar.extractfile(members[member]).read())
            if raw_hash != row["image_sha256"] or raw_hash != record["image_sha256"]:
                raise ValueError("raw image SHA256 mismatch: " + ident)
            scene = m.SceneArray.from_record(row["scene"])
            wire = bytes.fromhex(row["scene_wire_hex"])
            decoded = m.SceneArray.deserialize(wire, scene.producer_sha256)
            if scene.serialize() != wire or decoded.serialize() != wire:
                raise ValueError("scene wire/record mismatch")
            spi = row["spi"]
            available = spi.get("available")
            if type(available) is not bool or available and spi.get("level") not in levels:
                raise ValueError("invalid received SPI")
            values = [float(available and spi.get("level") == z) for z in levels] + [0., 0.]
            # Relevance and novelty are unavailable, not fabricated semantic scores.
            masks = [float(available)] * 4 + [0., 0.]
            x = decoded.features(scene.producer_sha256) + values + masks + [0.]
            if len(x) != 387 or not np.isfinite(x).all():
                raise ValueError("invalid387-entry received-feature row")
            features.append(x); ids.append(ident); producer_ids.add(scene.producer_sha256)
            group, split = row["group"], row["split"]
            groups.setdefault(group, set()).add(split)
            hashes.setdefault(raw_hash, set()).add(split)
            l = sum(n != 0 or round(a * 255) != 0 for n, a in zip(decoded.counts, decoded.largest_area))
            sparse_sizes.append(min(307, 29 + 4*l))
            counts["source/" + row["dataset"]] += 1
            counts["spi/" + spi.get("level", "Unknown")] += 1
            counts["split/" + split] += 1
            counts["training_exposed_hash"] += raw_hash in exposed_hashes
            counts["scene_ok"] += decoded.detector_ok
            counts["scene_saturated"] += decoded.saturated
            provenance.append({"image_id": ident, "sha256": raw_hash, "group": group,
                "split": split, "dataset": row["dataset"], "spi_training_hash_exposed": raw_hash in exposed_hashes,
                "declared_parameter_training_exposed": row.get("parameter_training_exposed"),
                "near_duplicate_status": row.get("near_duplicate_status"), "scope": "development only"})
            if (i+1) % 500 == 0:
                emit(f"validated {i+1}/{len(rows)} image hashes, wire blocks and features")
        crossings = {g: sorted(v) for g,v in groups.items() if len(v)>1}
        raw_crossings = {g: sorted(v) for g,v in hashes.items() if len(v)>1}
        if crossings or raw_crossings:
            write(out / "split_conflicts.json", {"groups": crossings, "raw_hashes": raw_crossings})
            raise ValueError("parent or exact duplicates cross scheduler splits")
        np.savez_compressed(out / "received_features.npz", image_ids=np.asarray(ids), features=np.asarray(features,dtype=np.float32))
        (out / "provenance.jsonl").write_text("".join(json.dumps(r)+"\n" for r in provenance))
        sizes = np.asarray(sparse_sizes)
        result = {"status": "PASS", "images_verified": len(ids), "features": [len(ids),387],
            "archive_sha256": archive_sha, "counts": dict(counts), "scene_classes": len(registry["classes"]),
            "producer_ids": sorted(producer_ids), "reference_urgency_available": sum(bool(r.get("urgency_truth_available")) for r in refs),
            "mission_relevance_available": sum(bool(r.get("mission_relevance_available")) for r in rows),
            "novelty_available": sum(bool(r.get("novelty_available")) for r in rows),
            "split_conflicts": 0, "unique_raw_images": len(hashes),
            "proposed_lossless_scene_bytes": {"mean": float(sizes.mean()), "p95": float(np.quantile(sizes,.95)), "max": int(sizes.max()),
                "scope": "calculated from real cached sparsity; compact serializer unimplemented; excludes scalar/routing/profile provisioning"},
            "training": "not run; qualified mission/context preference labels absent",
            "limitations": ["SPI is a prediction, not an independent target", "development pool is not fresh final evaluation", "near-duplicate grouping is not independently audited", "no real temporal sequence claim from randomized still images"],
            "elapsed_seconds": time.perf_counter()-start}
    after = {str(p): sha(p.read_bytes()) for p in files}
    if before != after:
        raise RuntimeError("upstream source changed during run; result must be revalidated")
    result["producer_sources_unchanged"] = True
    write(out / "summary.json", result)
    write(out / "artifact_manifest.json", {p.name: sha(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name != "run.log"})
    emit("COMPLETE cache ingress preflight; no inference, training, codec or network")
    log.close()


if __name__ == "__main__":
    main()
