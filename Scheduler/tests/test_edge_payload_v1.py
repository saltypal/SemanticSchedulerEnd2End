"""Payload reader: fine-tuned v1 bundles (seg_classes.yaml) project 150 ADE channels to 20; v0 bundles are untouched."""
import inspect
import shutil

import pytest

torch = pytest.importorskip("torch")

from Codec.edge import seg_map  # noqa: E402
from Codec.integration.sqgan_payload import SQGANDigitalCodec  # noqa: E402


def _codec(bundle):
    c = SQGANDigitalCodec.__new__(SQGANDigitalCodec)      # no model, no weights: only the semantic-map path
    c.bundle = bundle
    return c


def test_v1_bundle_projects_ade_logits_to_codec_classes(tmp_path):
    shutil.copy(seg_map.default_path(), tmp_path / "seg_classes.yaml")
    cfg = seg_map.load_class_map()
    index = seg_map.build_index(cfg)
    sky_src = next(r["id"] for r in cfg["ade_to_project"] if r["to"] == "sky")
    logits = torch.zeros(1, 150, 8, 16)
    logits[:, sky_src] = 10.0
    out = _codec(tmp_path)._one_hot_from_logits(logits)
    assert out.shape == (1, 20, 256, 512) and torch.all(out.sum(1) == 1)
    sky = [c["id"] for c in cfg["classes"] if c["name"] == "sky"][0]
    assert index[sky_src] == sky and torch.all(out.argmax(1) == sky)


def test_v1_wrong_channel_count_is_rejected(tmp_path):
    shutil.copy(seg_map.default_path(), tmp_path / "seg_classes.yaml")
    with pytest.raises(ValueError):
        _codec(tmp_path)._one_hot_from_logits(torch.zeros(1, 20, 8, 16))


def test_v0_bundle_argmaxes_the_20_channel_logits_unchanged(tmp_path):
    logits = torch.zeros(1, 20, 8, 16)
    logits[:, 5] = 3.0
    out = _codec(tmp_path)._one_hot_from_logits(logits)
    assert out.shape == (1, 20, 256, 512) and torch.all(out.argmax(1) == 5)


def test_reader_source_mentions_project_logits_for_the_pilot_check():
    assert "project_logits" in inspect.getsource(SQGANDigitalCodec._one_hot_from_logits)
