"""Encode whole SwinJSCC latents once and replay selected scheduled jobs.

Latent storage bytes, payload bits and complex channel uses remain separate.
This module runs the real configured checkpoint or raises; it has no synthetic
quality fallback.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Iterable, Mapping


def _load_components(model_dir: str | Path, upstream_root: str | Path, device: str):
    from EncDec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SwinJSCC but is unavailable")
    adapter = load_stage1_swinjscc(model_dir, device=device, upstream_root=upstream_root)
    for model in adapter._require_built():
        model.eval()
    checkpoint = Path(model_dir) / "stage1_swinjscc_full.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest_builder = sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    return adapter, device, digest


def _input_tensor(image_path: str | Path, device: str, *, size: int):
    from PIL import Image, ImageOps
    from EncDec.Evaluation.kodak_benchmark import pil_to_tensor

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        preprocessed = ImageOps.fit(opened.convert("RGB"), (size, size),
                                    method=Image.Resampling.BICUBIC)
    return pil_to_tensor(preprocessed, device), preprocessed


def cache_swinjscc_payloads(
    image_rows: Iterable[Mapping[str, object]], output_dir: str | Path,
    *, model_dir: str | Path, upstream_root: str | Path,
    snr_db: float = 10.0, rate: int = 96, device: str = "auto",
) -> Path:
    """Encode each distinct image once and persist tensors outside Git source."""
    import torch
    from INFRA.Artifacts import ImageArtifact

    adapter, device, checkpoint_sha = _load_components(model_dir, upstream_root, device)
    destination = Path(output_dir)
    latent_dir = destination / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen: set[str] = set()
    for row in image_rows:
        image_id = str(row["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate image ID in payload cache request: {image_id}")
        seen.add(image_id)
        image_size = tuple(adapter.config.initial_image_size)
        if image_size[0] != image_size[1]:
            raise ValueError("Scheduler replay currently needs a square codec input")
        tensor, _ = _input_tensor(str(row["image_path"]), device, size=image_size[0])
        artifact = ImageArtifact(tensor=tensor, sample_ids=[image_id], source="LADI-v2")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = perf_counter_ns()
        with torch.no_grad():
            latent = adapter.encode(artifact, snr_db=snr_db, rate=rate)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        encode_ms = (perf_counter_ns() - started) / 1_000_000.0
        tensor_bytes = int(latent.tensor.numel() * latent.tensor.element_size())
        mask_bytes = 0 if latent.rate_mask is None else int(
            latent.rate_mask.numel() * latent.rate_mask.element_size())
        active_real_values = latent.active_real_values()
        channel_uses = math.ceil(active_real_values / 2)
        saved = latent_dir / f"{image_id}.pt"
        torch.save({
            "tensor": latent.tensor.detach().cpu(),
            "rate_mask": None if latent.rate_mask is None else latent.rate_mask.detach().cpu(),
            "source_model": latent.source_model,
            "rate_tokens": latent.rate_tokens,
            "metadata": latent.metadata,
            "checkpoint_sha256": checkpoint_sha,
        }, saved)
        records.append({
            "image_id": image_id, "latent_path": saved.relative_to(destination).as_posix(),
            "latent_shape": "x".join(str(value) for value in latent.tensor.shape),
            "latent_storage_bytes": tensor_bytes,
            "rate_mask_storage_bytes": mask_bytes,
            "channel_uses": channel_uses,
            "resource_unit": "complex_channel_uses",
            "payload_bits": "unavailable",
            "encode_latency_ms": round(encode_ms, 6),
            "checkpoint_sha256": checkpoint_sha,
            "snr_db": snr_db, "rate": rate,
            "preprocessing": f"RGB center-fit bicubic {image_size[0]}x{image_size[1]}",
        })
    if not records:
        raise ValueError("No images were supplied for SwinJSCC caching")
    output = destination / "payload_costs.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


def replay_selected_jobs(
    selected_image_ids: Iterable[str], image_paths: Mapping[str, str | Path],
    payload_costs_csv: str | Path, output_dir: str | Path,
    *, model_dir: str | Path, upstream_root: str | Path,
    snr_db: float = 10.0, seed: int = 42, device: str = "auto",
) -> Path:
    """Transmit saved latents through AWGN and decode selected images."""
    import torch
    from EncDec.Evaluation.Image.psnr import psnr
    from EncDec.Evaluation.Image.ssim import ssim
    from EncDec.Evaluation.kodak_benchmark import tensor_to_pil
    from INFRA.Artifacts import LatentArtifact
    from INFRA.Channels.awgn import AWGNChannel

    adapter, device, checkpoint_sha = _load_components(model_dir, upstream_root, device)
    with Path(payload_costs_csv).open(newline="", encoding="utf-8") as handle:
        cached = {row["image_id"]: row for row in csv.DictReader(handle)}
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    channel = AWGNChannel()
    records = []
    for image_id in selected_image_ids:
        if image_id not in cached or image_id not in image_paths:
            raise KeyError(f"Missing real payload or source image for {image_id}")
        record = cached[image_id]
        if record["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Cached latent checkpoint differs from replay checkpoint")
        latent_path = Path(record["latent_path"])
        if not latent_path.is_absolute():
            latent_path = Path(payload_costs_csv).parent / latent_path
        saved = torch.load(latent_path, map_location=device, weights_only=True)
        latent = LatentArtifact(
            tensor=saved["tensor"], rate_mask=saved["rate_mask"],
            source_model=saved["source_model"],
            rate_tokens=saved["rate_tokens"], metadata=saved["metadata"],
        )
        image_size = tuple(adapter.config.initial_image_size)
        if image_size[0] != image_size[1]:
            raise ValueError("Scheduler replay currently needs a square codec input")
        reference, original_pil = _input_tensor(
            image_paths[image_id], device, size=image_size[0])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = perf_counter_ns()
        with torch.no_grad():
            transmission = channel.transmit(latent, snr_db)
            reconstruction = adapter.reconstruct_received(transmission, latent).tensor
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        replay_ms = (perf_counter_ns() - started) / 1_000_000.0
        original_path = destination / f"{image_id}_input.png"
        reconstructed_path = destination / f"{image_id}_reconstruction.png"
        original_pil.save(original_path)
        tensor_to_pil(reconstruction).save(reconstructed_path)
        records.append({
            "image_id": image_id,
            "status": "measured_checkpoint_inference",
            "psnr_db": psnr(reference, reconstruction),
            "ssim": ssim(reference, reconstruction),
            "replay_latency_ms": round(replay_ms, 6),
            "channel_uses": transmission.channel_uses_per_image[0],
            "checkpoint_sha256": checkpoint_sha,
            "snr_db": snr_db, "input_path": str(original_path),
            "reconstruction_path": str(reconstructed_path),
        })
    if not records:
        raise ValueError("No selected jobs were requested for codec replay")
    output = destination / "reconstruction_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output
