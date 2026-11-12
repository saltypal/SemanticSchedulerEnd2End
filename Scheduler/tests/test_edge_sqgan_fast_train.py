"""CPU smoke of Codec/edge/sqgan_fast_train.execute_fast with a tiny fake SQ-GAN (no upstream source, no GPU)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn


def _fake_src(monkeypatch, n_images=6):
    class Images(torch.utils.data.Dataset):
        def __init__(self, manifest, split, size):
            pass

        def __len__(self):
            return n_images

        def __getitem__(self, i):
            return {"image": torch.rand(3, 8, 16)}

    class EMA:
        def __init__(self, modules):
            pass

        def update(self):
            return {"image": 1, "semantic": 1}

        def close(self):
            pass

    src = types.ModuleType("src")
    mods = {"src": src, "src.dataset": types.SimpleNamespace(ParentImages=Images),
            "src.ema": types.SimpleNamespace(ReplicaEMA=EMA),
            "src.runtime": types.SimpleNamespace(write_json=lambda p, o: Path(p).write_text(json.dumps(o))),
            "src.evaluate": types.SimpleNamespace(evaluate=lambda *a, **k: None)}
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)


class _Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Linear(3, 1)

    def forward(self, codebook_loss, semantic, inputs, reconstructions, optimizer_idx, global_step, last_layer, split):
        if optimizer_idx == 0:
            fake = self.discriminator(reconstructions.mean((2, 3)))
            return ((inputs - reconstructions) ** 2).mean() - fake.mean() + codebook_loss, {}
        real = self.discriminator(inputs.detach().mean((2, 3)))
        fake = self.discriminator(reconstructions.detach().mean((2, 3)))
        return torch.relu(1 - real).mean() + torch.relu(1 + fake).mean(), {}


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 3, 1)
        self.loss = _Loss()
        self.quantize = types.SimpleNamespace(codebook=None)
        self.quantize_semantic = types.SimpleNamespace(codebook=None)

    def get_last_layer(self):
        return self.net.weight

    def configure_optimizers(self):
        g = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        d = torch.optim.Adam(self.loss.discriminator.parameters(), lr=1e-3)
        return [g, d], [torch.optim.lr_scheduler.LambdaLR(g, lambda s: 1.0),
                        torch.optim.lr_scheduler.LambdaLR(d, lambda s: 1.0)]


class _View(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()
        self.metadata = {}

    def forward(self, image):
        rec = self.core.net(image)
        return {"inputs": image, "xrec": rec, "semantic": image, "qloss": rec.new_zeros(len(image))}


def _module():
    spec = importlib.util.spec_from_file_location("sqgan_fast_train",
                                                  Path(__file__).parents[2] / "Codec/edge/sqgan_fast_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(tmp_path, monkeypatch, stage, **opts):
    _fake_src(monkeypatch)
    fake_train = types.SimpleNamespace(build=lambda o: _View(), execute=None)
    options = {"manifest": "x", "batch_size": 2, "data_parallel": False, "num_workers": 0, "epochs": 2,
               "max_seconds": 60, **opts}
    _module().execute_fast(fake_train, stage, options, tmp_path)
    return tmp_path


def test_pilot_two_updates_and_checkpoint(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, "pilot")
    assert (run / "pilot" / "checkpoints" / "last.pt").exists()
    s = json.loads((run / "pilot" / "training_summary.json").read_text())
    assert s["steps"] == 2 and s["fast_loop"] is True
    assert (run / "pilot_passed.json").exists()          # the train stage requires this marker
    assert json.loads((run / "pilot" / "fast_loop.json").read_text())["amp_fp16"] is False   # cpu: amp off


def test_train_full_epochs_reuse_and_second_forward(tmp_path, monkeypatch):
    (tmp_path / "pilot_passed.json").write_text("{}")
    run = _run(tmp_path, monkeypatch, "train")
    assert json.loads((run / "training_summary.json").read_text())["steps"] == 6      # 3 batches x 2 epochs
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "pilot_passed.json").write_text("{}")
    run2 = _run(tmp_path / "b", monkeypatch, "train", reuse_forward=False)
    rows = [json.loads(x) for x in (run2 / "training.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(r["discriminator_loss"] == r["discriminator_loss"] for r in rows)   # finite
