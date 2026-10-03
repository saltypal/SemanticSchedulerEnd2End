"""Transport boundary checks with an explicit fake decoder, not a checkpoint."""
import csv
from types import SimpleNamespace

import pytest

from INFRA.latent_contracts import LatentJob


def test_cached_tensor_transport_loads_once_and_checks_identity(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from EncDec.integration import scheduler_replay
    from EncDec.integration.latent_transport import CachedSwinLatentTransport
    checkpoint = "a" * 64
    calls = []
    class Decoder:
        def reconstruct_received(self, transmission, latent):
            return SimpleNamespace(tensor=torch.ones(1, 3, 2, 2) * .5)
    def load(*args):
        calls.append(args)
        return Decoder(), "cpu", checkpoint
    monkeypatch.setattr(scheduler_replay, "_load_components", load)
    torch.save({"tensor": torch.ones(1, 4), "rate_mask": None,
        "source_model": "fake-test", "rate_tokens": 1, "metadata": {},
        "checkpoint_sha256": checkpoint}, tmp_path / "latent.pt")
    profile = tmp_path / "payload_costs.csv"
    with profile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "channel_uses", "checkpoint_sha256", "latent_path"))
        writer.writeheader()
        writer.writerow({"image_id":"image", "channel_uses":2,
                         "checkpoint_sha256":checkpoint, "latent_path":"latent.pt"})
    transport = CachedSwinLatentTransport(profile, tmp_path / "out", model_dir="unused", upstream_root="unused")
    job = LatentJob("job", "image", "task", "payload", 2, "complex_channel_uses", "swinjscc:" + checkpoint)
    first, second = transport(job), transport(job)
    assert len(calls) == 1
    assert first.transport_success and second.transport_success
    assert first.task_quality is None and first.receiver_embedding == ()
    assert (tmp_path / "out/000001_reconstruction.png").exists()
    from dataclasses import replace
    with pytest.raises(ValueError):
        transport(replace(job, resource_units=3))
