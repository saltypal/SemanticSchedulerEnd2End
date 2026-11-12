"""True batched decode: equals per-packet decode (fake torch modules, CPU), profile uses it, latency notebook builder."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from Codec.edge import sqgan_profile as sp  # noqa: E402
from Codec.integration.sqgan_payload import SQGANDigitalCodec  # noqa: E402
from scripts.edge import make_latency_profile_notebook as mk  # noqa: E402


class _Q(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(16, 4)
    def get_codebook_entry(self, idx):
        return self.emb(idx)                         # (B,k,4)


class _Demask(nn.Module):
    def forward(self, vec, _empty, sel, _none, mask):   # (B,4,k) -> (B,4,512) scattered
        out = vec.new_zeros(vec.shape[0], vec.shape[1], 512)
        return out.scatter(2, sel[:, None, :].expand(-1, vec.shape[1], -1), vec) + mask[:, None, :]


class _Dec(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.lin = nn.Linear(512, 256 * 512 // 64); self.ch = ch
    def forward(self, z, cond=None):
        y = self.lin(z).reshape(z.shape[0], z.shape[1], 256, 8)
        y = nn.functional.interpolate(y, size=(256, 512))
        return y[:, :self.ch] if z.shape[1] >= self.ch else y.repeat(1, self.ch, 1, 1)[:, :self.ch]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantize, self.quantize_semantic = _Q(), _Q()
        self.demasker = self.demasker_semantic = _Demask()
        self.decoder_semantic, self.decoder = _Dec(20), _Dec(3)
    def decode(self, packet):                         # single-packet path, same maths
        raise AssertionError("decode_batch must not call the single-packet decoder")


class _Packet:
    @staticmethod
    def decode_packet(p):
        k = p[0]
        sec = lambda off: (list(range(off, off + k)), [(off + i) % 16 for i in range(k)])
        return sec(p[1]), sec(p[2])


def _codec():
    torch.manual_seed(0)
    c = SQGANDigitalCodec.__new__(SQGANDigitalCodec)
    c.model, c.packet = _Model().eval(), _Packet()
    return c


def test_decode_batch_equals_single():
    c = _codec()
    ps = [bytes([3, 1, 5]), bytes([3, 7, 2]), bytes([5, 0, 9]), bytes([3, 4, 4])]   # two distinct kept-token counts
    batch = c.decode_batch(ps)
    assert len(batch) == 4 and all(t.shape == (1, 3, 256, 512) for t in batch)
    for p, t in zip(ps, batch):
        assert torch.allclose(c.decode_batch([p])[0], t, atol=1e-5)
    assert c.decode_batch([]) == []
    with pytest.raises(TypeError):
        c.decode_batch(["x"])


class _FakeCodec:
    def __init__(self, with_batch):
        self.calls = []
        if with_batch:
            self.decode_batch = lambda ps: self.calls.append(("batch", len(ps))) or [0] * len(ps)
    def encode(self, img):
        return b"payload"
    def decode(self, p):
        self.calls.append(("single", 1))


@pytest.mark.parametrize("with_batch", [True, False])
def test_profile_prefers_decode_batch(with_batch):
    c = _FakeCodec(with_batch)
    r = sp.profile_codec(c, [None], [1, 2], 2)
    assert r["emulated_batching"] is (not with_batch)
    assert {k for k, _ in c.calls} == ({"batch"} if with_batch else {"single"})
    assert set(r["latency_samples"]) == {1, 2}


def test_notebook_builder():
    nb, meta = mk.build()
    assert meta["id"] == "satyapaladuguplus/edge-v1-latency" and meta["title"] == "Edge V1 Latency"
    assert meta["kernel_sources"] == ["satyapaladuguplus/edge-sqgan-finetune-v1"]
    assert "satyapaladuguplus/scheduler-dataset-v1-images" in meta["dataset_sources"]
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["machine_shape"] == "NvidiaTeslaT4"
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "latency_v1_true.json" in src and "decode_batch" in src and '"emulated_batching": False' in src
    assert "@OUT@" not in src
