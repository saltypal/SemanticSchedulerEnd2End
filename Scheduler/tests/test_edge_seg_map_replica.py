"""wrap_teacher must survive DataParallel replication (fine-tune crash 2026-11-12: replica on cuda:1 ran cuda:0 weights)."""
import types

import torch

from Codec.edge import seg_map as sm


class _Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(id2label={})

    def forward(self, x):
        return types.SimpleNamespace(logits=x + self.w * torch.arange(20.).view(1, 20, 1, 1))


def test_replica_runs_its_own_weights(monkeypatch):
    monkeypatch.setattr(sm, "check_teacher_labels", lambda cfg, labels: None)
    monkeypatch.setattr(sm, "build_index", lambda cfg: list(range(20)))
    t = sm.wrap_teacher(_Teacher(), None)
    rep = t._replicate_for_data_parallel()                  # what DataParallel.replicate does per device
    rep._parameters = dict(rep._parameters)
    rep._parameters["w"] = torch.nn.Parameter(torch.ones(1))
    x = torch.zeros(1, 20, 2, 2)
    assert not torch.equal(t(x).logits, rep(x).logits)
    assert t(x).logits.shape == (1, 20, 2, 2) and list(t.state_dict()) == ["w"]   # keys unchanged
