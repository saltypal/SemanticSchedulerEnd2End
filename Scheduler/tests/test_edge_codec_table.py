import json
import numpy as np
from Codec.edge import codec_table as ct
from Codec.edge import jpeg_ladder as jl
from Codec.edge import sqgan_profile as sp
from Evaluation.edge.contracts import LatencyProfile


class FakeCodec:
    def encode(self, im):
        return b"x" * 10

    def decode(self, p):
        return p


def fake_kw():
    return dict(encode_fn=lambda im: (b"p" * (im + 1), [0, 1], np.zeros((256, 512), int)),
                decode_fn=lambda p: p, judge_fn=lambda a, b: 1, sts_fn=lambda s, k: [0.5] * 13)


def test_profile_shape_and_latency_profile():
    p = sp.profile_codec(FakeCodec(), [0, 1], [1, 2, 4], 3, "cpu")
    assert sorted(p["latency_samples"]) == [1, 2, 4] and all(len(v) == 3 for v in p["latency_samples"].values())
    lp = LatencyProfile({b: [max(x, 1e-9) for x in v] for b, v in p["latency_samples"].items()})
    assert lp.batch_sizes == (1, 2, 4) and p["emulated_batching"]


def test_save_load(tmp_path):
    p = sp.profile_codec(FakeCodec(), [0], [1, 2], 2)
    sp.save_profile(p, tmp_path / "p.json")
    assert sorted(sp.load_profile(tmp_path / "p.json")["latency_samples"]) == [1, 2]


def test_resumable_one_row_per_id(tmp_path):
    out = tmp_path / "t.jsonl"
    items = [("a%d" % i, i) for i in range(6)]
    ct.run_table(items[:3], out, **fake_kw())
    ct.run_table(items, out, **fake_kw())
    ids = [json.loads(l)["image_id"] for l in out.read_text().splitlines()]
    assert sorted(ids) == sorted(k for k, _ in items) and len(ids) == 6


def test_shards_disjoint_cover(tmp_path):
    items = [("img%d" % i, i) for i in range(40)]
    got = []
    for s in range(2):
        out = tmp_path / ("s%d.jsonl" % s)
        ct.run_table(items, out, shard=(s, 2), **fake_kw())
        got.append({json.loads(l)["image_id"] for l in out.read_text().splitlines()})
    assert not got[0] & got[1] and got[0] | got[1] == {k for k, _ in items}


def test_jpeg_ladder_monotone():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    rows = jl.ladder_rows([("a", img)], lambda a, b: 1)
    sizes = [r["payload_bytes"] for r in sorted(rows, key=lambda r: r["quality"])]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    assert rows[0]["sts"] == [] and rows[0]["sts_available"] is False
