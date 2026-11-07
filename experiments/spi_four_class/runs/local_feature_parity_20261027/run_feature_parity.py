from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path("parameters/results/spi_inputs_v2_20261027b")
OUT = Path("experiments/spi_four_class/runs/local_feature_parity_20261027")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path.cwd()))
started = time.perf_counter()
print("START offline CPU CLIP feature parity; no network/model download", flush=True)

import zipfile
from parameters.contracts import MissionSpec
from parameters.extractor import embedding_signature
from parameters.minimal_vision import MinimalVisionProvider

rows = json.loads((ROOT / "panel.json").read_text(encoding="utf-8"))
row = next(r for r in rows if r["split"] == "train")
with np.load(ROOT / "features.npz", allow_pickle=False) as cached:
    positions = {image_id: index for index, image_id in enumerate(cached["image_ids"].tolist())}
    if row["image_id"] not in positions:
        raise ValueError("selected image ID missing from cached feature rows")
    cached_index = positions[row["image_id"]]
    cached_embedding = np.asarray(cached["embeddings"][cached_index], dtype=np.float32)
    expected_signature = str(cached["feature_signature"].item())

with zipfile.ZipFile(ROOT / "images.zip") as archive:
    image_bytes = archive.read(row["image_path"])
image_sha = hashlib.sha256(image_bytes).hexdigest()
if image_sha != row["image_sha256"]:
    raise ValueError("selected image bytes do not match panel SHA-256")
with Image.open(io.BytesIO(image_bytes)) as opened:
    opened.load()
    original_size = list(opened.size)
    image = opened.copy()

mission = MissionSpec(
    "spi_feature_parity",
    "Assess visible conditions in a field photograph.",
    (("visible", "a photograph"),),
)
checkpoint = "models/priority_classifiers/clip"
provider = MinimalVisionProvider(checkpoint, {mission.mission_id: mission}, device="cpu")
try:
    print("START loading cached CLIP snapshot in local_files_only mode", flush=True)
    live = provider.extract_batch([image], [mission])[0]
finally:
    provider.close()

live_embedding = np.asarray(live.embedding, dtype=np.float32)
live_signature = embedding_signature(live)
if live_embedding.shape != cached_embedding.shape:
    raise ValueError(f"embedding shape mismatch: {live_embedding.shape} vs {cached_embedding.shape}")
if not np.isfinite(live_embedding).all() or not np.isfinite(cached_embedding).all():
    raise ValueError("nonfinite feature value")
max_absolute_difference = float(np.max(np.abs(live_embedding - cached_embedding)))
cosine = float(np.dot(live_embedding, cached_embedding) / (
    np.linalg.norm(live_embedding) * np.linalg.norm(cached_embedding)
))
provenance = dict(live.provenance)
result = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "project": str(Path.cwd()),
    "interpreter": __import__("sys").executable,
    "python": __import__("platform").python_version(),
    "device": "cpu",
    "network_access": "disabled by local_files_only loader",
    "input_archive": str(ROOT / "images.zip"),
    "input_archive_sha256": hashlib.sha256((ROOT / "images.zip").read_bytes()).hexdigest(),
    "features_file_sha256": hashlib.sha256((ROOT / "features.npz").read_bytes()).hexdigest(),
    "image_id": row["image_id"],
    "image_path": row["image_path"],
    "image_sha256": image_sha,
    "split": row["split"],
    "development_exposed": row["development_exposed"],
    "original_image_size": original_size,
    "checkpoint": checkpoint,
    "live_provenance": provenance,
    "cached_feature_signature": expected_signature,
    "live_embedding_signature": live_signature,
    "signature_match": expected_signature == live_signature,
    "embedding_dimension": int(live_embedding.size),
    "cached_l2_norm": float(np.linalg.norm(cached_embedding)),
    "live_l2_norm": float(np.linalg.norm(live_embedding)),
    "cosine_similarity": cosine,
    "max_absolute_difference": max_absolute_difference,
    "parity_threshold_max_abs": 1e-4,
    "embedding_match_within_threshold": max_absolute_difference <= 1e-4,
    "elapsed_seconds": time.perf_counter() - started,
    "model_weights_downloaded": False,
    "cache_or_source_modified": False,
}
(OUT / "feature_parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Feature parity result", json.dumps({
    key: result[key] for key in (
        "image_id", "image_sha256", "signature_match", "embedding_dimension",
        "cosine_similarity", "max_absolute_difference", "embedding_match_within_threshold",
        "elapsed_seconds",
    )
}, indent=2), flush=True)
print("END offline CPU feature parity", flush=True)
