"""Prepared SwinJSCC transport callable for the whole-latent engine.

The checkpoint is loaded once, not once per scheduling decision. Optional
receiver/task extractors evaluate reconstructions, never original images.
"""
from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter_ns

from INFRA.latent.contracts import DeliveryEvidence


class CachedSwinLatentTransport:
    def __init__(self, payload_costs, output_dir, *, model_dir, upstream_root,
                 snr_db=10., seed=42, device="auto", receiver_extractor=None,
                 task_evaluator=None):
        import torch
        from EncDec.integration.scheduler_replay import _load_components
        from INFRA.Channels.awgn import AWGNChannel
        self.adapter, self.device, self.checkpoint_sha = _load_components(model_dir, upstream_root, device)
        self.torch = torch
        self.channel = AWGNChannel()
        self.profile = Path(payload_costs)
        with self.profile.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.costs = {row["image_id"]: row for row in rows}
        if len(self.costs) != len(rows):
            raise ValueError("Duplicate cached payload identity")
        self.destination = Path(output_dir)
        self.destination.mkdir(parents=True, exist_ok=True)
        self.snr_db = snr_db
        self.receiver_extractor = receiver_extractor
        self.task_evaluator = task_evaluator
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        self.step = 0
        self.records = []

    def __call__(self, job):
        from INFRA.Artifacts import LatentArtifact
        from EncDec.Evaluation.kodak_benchmark import tensor_to_pil
        record = self.costs[job.image_id]
        if (job.resource_unit != "complex_channel_uses" or
                job.resource_units != int(record["channel_uses"]) or
                record["checkpoint_sha256"] != self.checkpoint_sha or
                job.payload_provenance != "swinjscc:" + self.checkpoint_sha):
            raise ValueError("Selected latent demand/checkpoint differs from its cached profile")
        path = Path(record["latent_path"])
        if not path.is_absolute():
            path = self.profile.parent / path
        saved = self.torch.load(path, map_location=self.device, weights_only=True)
        if saved["checkpoint_sha256"] != self.checkpoint_sha:
            raise ValueError("Saved tensor checkpoint identity differs")
        latent = LatentArtifact(tensor=saved["tensor"], rate_mask=saved["rate_mask"],
            source_model=saved["source_model"], rate_tokens=saved["rate_tokens"], metadata=saved["metadata"])
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = perf_counter_ns()
        with self.torch.no_grad():
            transmission = self.channel.transmit(latent, self.snr_db)
            reconstructed = self.adapter.reconstruct_received(transmission, latent).tensor
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        elapsed = (perf_counter_ns() - started) / 1_000_000
        if transmission.channel_uses_per_image[0] != job.resource_units:
            raise ValueError("Measured transmitted demand differs from the selected payload")
        if not self.torch.isfinite(reconstructed).all().item():
            return DeliveryEvidence(job.job_id, False, "swinjscc_nonfinite_reconstruction")
        self.step += 1
        output = self.destination / f"{self.step:06d}_reconstruction.png"
        tensor_to_pil(reconstructed).save(output)
        embedding = (() if self.receiver_extractor is None else
                     tuple(float(value) for value in self.receiver_extractor(output)))
        quality = None if self.task_evaluator is None else float(self.task_evaluator(job, output))
        self.records.append({"job_id": job.job_id, "payload_id": job.payload_id,
            "reconstruction_path": str(output), "channel_uses": job.resource_units,
            "decode_transport_wall_ms": elapsed, "snr_db": self.snr_db,
            "checkpoint_sha256": self.checkpoint_sha})
        import json
        (self.destination / "transport_metrics.json").write_text(
            json.dumps(self.records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return DeliveryEvidence(job.job_id, True, "measured_swinjscc:" + self.checkpoint_sha,
                                task_quality=quality, receiver_embedding=embedding)
