"""Pilot checks: pass / fail / skip paths with injected fakes (no GPU, no network), registry, report table."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from Codec.edge import pilot_checks as pc
from parameters.edge import pilot

IMGS = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (300, 520, 3), dtype=np.uint8)) for i in range(8)]


def ctx(tmp_path, **inject):
    return pc.Ctx(out=tmp_path, input_base=str(tmp_path / "input"), bundle_root=str(tmp_path / "bundle"),
                  inject=dict(images=IMGS, **inject))


def test_registry_has_no_duplicates_and_every_item_maps_to_an_existing_check():
    assert len(pc.CHECKS) == len(set(pc.CHECKS))
    for item_id, (desc, source, name) in pc.UNVERIFIED_ITEMS.items():
        assert desc and source and name in pc.CHECKS, item_id
    assert sorted(pc.CHECK_ORDER) == sorted(pc.CHECKS)
    for must in ("gpu_count", "dataparallel_t4x2", "bundle_flat_restore", "real_masker_output", "trainer_launch",
                 "payload_reader_gap", "dataset_a_size_estimate", "api_token_env", "kaggle_input_discovery",
                 "load_embedder_convnextv2", "feature_pass_m_mapping", "feature_pass_disk_projection"):
        assert must in pc.UNVERIFIED_ITEMS
    for name in pc.CHECKS:
        assert pc.covers(name), "check %s covers nothing" % name


def test_run_check_writes_json_and_converts_exceptions(tmp_path):
    r = pc.run_check("env_gpu", ctx(tmp_path))
    saved = json.loads((tmp_path / "checks" / "env_gpu.json").read_text())
    assert saved["name"] == "env_gpu" and saved["status"] == r["status"] and "seconds" in saved
    assert saved["items"]["dataparallel_t4x2"]["status"] in ("skip", "pass")
    pc.CHECKS["boom"] = lambda c: 1 / 0
    try:
        r = pc.run_check("boom", ctx(tmp_path))
    finally:
        del pc.CHECKS["boom"]
    assert r["status"] == "fail" and "ZeroDivisionError" in r["error"]


def test_skip_when_nothing_mounted(tmp_path):
    for name in ("bundle_restore", "discovery_note", "dataset_scale", "feature_pass_smoke_verify", "source_root"):
        r = pc.run_check(name, ctx(tmp_path))
        assert r["status"] == "skip", name
        assert all(v["status"] == "skip" for v in r["items"].values())


class FakeMasker:
    def __init__(self):
        self.fns = []

    def register_forward_hook(self, fn):
        self.fns.append(fn)
        return SimpleNamespace(remove=lambda: self.fns.remove(fn))


def fake_codec(kept_fn=lambda m: list(range(round(m * 512)))):
    import torch
    codec = SimpleNamespace(model=SimpleNamespace(masker=FakeMasker()), bundle=".")

    def encode(image, semantic=None):
        for fn in list(codec.model.masker.fns):
            fn(None, (), {"sample_index": torch.tensor(kept_fn(0.25)), "alpha": torch.zeros(3)})
        return b"x" * 100

    codec.encode = encode
    codec.semantic_map = lambda image: torch.nn.functional.one_hot(
        torch.zeros(1, 256, 512, dtype=torch.long), 20).permute(0, 3, 1, 2).float()
    return codec


def test_masker_format_pass_and_fail(tmp_path):
    pytest.importorskip("torch")
    r = pc.run_check("masker_format", ctx(tmp_path, make_codec=lambda m: fake_codec(), device="cpu"))
    assert r["status"] == "pass", r["error"] or r["details"]
    assert r["details"]["sample_index_present"] and r["details"]["alpha_exposed"]
    r = pc.run_check("masker_format", ctx(tmp_path, make_codec=lambda m: fake_codec(lambda m: list(range(10))), device="cpu"))
    assert r["status"] == "fail"


def test_payload_reader_gap_skip_for_v0_bundle(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "emptybundle").mkdir()
    codec = fake_codec()
    codec.bundle = str(tmp_path / "emptybundle")
    r = pc.run_check("payload_semantic_map", ctx(tmp_path, make_codec=lambda m: codec, device="cpu"))
    assert r["status"] == "pass", r["error"] or r["details"]
    assert r["items"]["payload_reader_gap"]["status"] == "skip"
    assert r["details"]["shape"] == [1, 20, 256, 512]


def test_payload_reader_gap_fails_when_seg_classes_in_bundle(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "seg_classes.yaml").write_text("x: 1")
    codec = fake_codec()
    codec.bundle = str(tmp_path / "bundle")
    r = pc.run_check("payload_semantic_map", ctx(tmp_path, make_codec=lambda m: codec, device="cpu"))
    assert r["items"]["payload_reader_gap"]["status"] == "fail" and r["status"] == "fail"
    assert "KNOWN GAP" in r["items"]["payload_reader_gap"]["detail"]


def test_embedders_one_failure_does_not_stop_the_others(tmp_path):
    grid = {"encoders": [{"name": "clip_b32", "family": "clip", "dim": 4}, {"name": "dinov2_small", "family": "dino", "dim": 4},
                         {"name": "convnextv2_tiny", "family": "convnext", "dim": 4}]}

    def loader(name):
        if name == "dinov2_small":
            raise OSError("no network")
        dim = 5 if name == "convnextv2_tiny" else 4
        return lambda imgs: np.ones((len(imgs), dim), np.float32)
    r = pc.run_check("embedders", ctx(tmp_path, load_embedder=loader, grid_cfg=grid, ue_probe=lambda n, imgs: {"params": 1}))
    assert r["status"] == "fail" and r["details"]["failed"] == ["dinov2_small", "convnextv2_tiny"]
    assert r["items"]["load_embedder_clip"]["status"] == "pass"
    assert r["items"]["load_embedder_dinov2_small"]["status"] == "fail"
    assert r["items"]["load_embedder_convnextv2"]["status"] == "fail" and "dim 5" in r["items"]["load_embedder_convnextv2"]["detail"]
    assert r["details"]["clip_b32"]["ue_cost_probe"] == {"params": 1}
    ok = pc.run_check("embedders", ctx(tmp_path, load_embedder=lambda n: (lambda imgs: np.ones((len(imgs), 4), np.float32)),
                                      grid_cfg={"encoders": grid["encoders"][:1]}, ue_probe=lambda n, i: {}))
    assert ok["status"] == "pass"


def test_dataset_scale_estimate_and_rate(tmp_path):
    a = tmp_path / "dataset-a"
    a.mkdir()
    rows = [{"source": "crisismmd", "bytes": 100_000}, {"source": "places365", "bytes": 20_000},
            {"source": "coco_people", "bytes": 50_000}]
    (a / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "stage_results.json").write_text(json.dumps([{"name": "a_download_images", "seconds": 1.5, "peak_rss_mb": 321.0}]))
    r = pc.run_check("dataset_scale", ctx(tmp_path))
    assert r["details"]["images_per_second"] == 2.0 and r["details"]["peak_rss_mb"] == 321.0
    assert r["items"]["workers16_memory"]["status"] == "pass"
    assert r["items"]["dataset_a_size_estimate"]["status"] == "pass" and r["details"]["estimate_gb"] < 3.0


def test_dataset_scale_fails_when_estimate_exceeds_limit(tmp_path):
    a = tmp_path / "dataset-a"
    a.mkdir()
    rows = [{"source": s, "bytes": 5_000_000} for s in ("crisismmd", "places365", "coco_people")]
    (a / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    r = pc.run_check("dataset_scale", ctx(tmp_path))
    assert r["items"]["dataset_a_size_estimate"]["status"] == "fail" and r["details"]["estimate_gb"] > 90
    assert r["items"]["workers16_memory"]["status"] == "fail"             # no stage_results.json


def test_publish_roundtrip_paths(tmp_path):
    calls = []

    def publish(folder, slug, title, message=""):
        calls.append(len(list(Path(folder).rglob("*.txt"))))
        return {"status": "created" if len(calls) == 1 else "versioned"}
    r = pc.run_check("publish_roundtrip", ctx(tmp_path, publish=publish, cli_publish=True))
    assert r["status"] == "pass", r["items"]
    assert calls[0] == 300 and r["items"]["datasets_status_parse"]["status"] == "pass"
    bad = pc.run_check("publish_roundtrip", ctx(tmp_path, publish=lambda *a, **k: {"status": "created"}, cli_publish=True))
    assert bad["items"]["datasets_status_parse"]["status"] == "fail" and bad["status"] == "fail"
    skip = pc.run_check("publish_roundtrip", ctx(tmp_path, publish=lambda *a, **k: {}, creds={}))
    assert skip["status"] == "skip" and "KAGGLE_API_TOKEN" in skip["error"]


def test_discovery_note(tmp_path):
    d = tmp_path / "input" / "ds" / "x"
    d.mkdir(parents=True)
    (d / "index.jsonl").write_text("{}")
    (d / "labels_raw.jsonl").write_text("{}")
    r = pc.run_check("discovery_note", ctx(tmp_path))
    assert r["status"] == "pass", r["items"]
    assert r["details"]["discover_images_root"].endswith("x")


def test_judge_dino_with_fake_extractor(tmp_path):
    pytest.importorskip("sklearn")
    emb = np.random.default_rng(0).normal(size=(8, 6))
    r = pc.run_check("judge_dino", ctx(tmp_path, extract_dino=lambda imgs: emb))
    assert r["status"] == "pass", r["error"] or r["items"]


def make_smoke(root, good=True):
    from Codec.edge.feature_pass import m_tag
    cache = root / "cache"
    s = cache / "shards" / "shard_0000"
    s.mkdir(parents=True)
    tags = [m_tag(m) for m in pc._cfg("feature_pass")["sqgan"]["m_levels"]]
    (s / "done.json").write_text("{}")
    (s / "failures.json").write_text("[]")
    for f in ("phash64", "simhash64", "sqgan_ok", "class_grid", "prefeat_meanmax"):
        np.save(s / (f + ".npy"), np.zeros(2))
    for t in tags:
        for a in ("offsets", "kept_bitmap", "sts"):
            np.save(s / ("%s_%s.npy" % (a, t)), np.zeros(2))
        (s / ("recon_" + t)).mkdir()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(s / ("recon_" + t) / "a.jpg")
    stages = ["load", "phash", "sqgan_segment", "recon_save"] + ["sqgan_encode_" + t for t in tags] + ["decode_" + t for t in tags]
    rows = [{"uid": "u", "stage": st, "ms": (50.0 if st.startswith("decode") else 1.0) if good else -1.0, "warmup": False} for st in stages]
    (s / "timings.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (cache / "summary.json").write_text(json.dumps({"counts": {"ok": 20}}))
    (cache / "compute_profile.json").write_text(json.dumps({"sqgan_encoder": {"flops": 5}, "phash": {"cpu_ms_mean": 1}}))


def test_smoke_verify_pass_and_fail(tmp_path):
    good, bad = tmp_path / "g", tmp_path / "b"
    good.mkdir()
    bad.mkdir()
    make_smoke(good)
    r = pc.run_check("feature_pass_smoke_verify", ctx(good))
    assert r["status"] == "pass", r["items"]
    make_smoke(bad, good=False)
    r = pc.run_check("feature_pass_smoke_verify", ctx(bad))
    assert r["status"] == "fail" and r["items"]["feature_pass_timings_stages"]["status"] == "fail"


def test_report_table_has_correct_statuses(tmp_path):
    folder = tmp_path / "checks"
    folder.mkdir()
    (folder / "env_gpu.json").write_text(json.dumps({"name": "env_gpu", "status": "fail", "items": {
        "gpu_count": {"status": "pass", "detail": "2 GPU(s)"}, "dataparallel_t4x2": {"status": "skip", "detail": "only 1"}}}))
    (folder / "ade_teacher.json").write_text(json.dumps({"name": "ade_teacher", "status": "skip", "error": "no network", "items": {}}))
    checks = pilot.read_checks(tmp_path)
    assert pilot.item_row("gpu_count", checks) == ("PASS", "2 GPU(s)")
    assert pilot.item_row("dataparallel_t4x2", checks)[0] == "SKIP"
    assert pilot.item_row("torch_cuda", checks)[0] == "FAIL"                  # check failed, item inherits
    assert pilot.item_row("ade_id2label_match", checks) == ("SKIP", "no network")
    assert pilot.item_row("judge_dino_embeddings", checks)[0] == "FAIL"       # no json at all
    report = pilot.run_pilot(tmp_path, 3, runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
                             stages=[], cwd=str(tmp_path))
    md = (tmp_path / "PILOT_REPORT.md").read_text()
    assert "## Unverified items" in md and "| gpu_count | env_gpu | PASS | 2 GPU(s) |" in md
    assert "| ade_id2label_match | ade_teacher | SKIP | no network |" in md
    assert report["checks"]["env_gpu"]["status"] == "fail"


def test_notebook_builder_emits_collect_cell_and_metadata():
    from scripts.edge import make_pilot_notebook as mk
    nb, meta = mk.build()
    text = json.dumps(nb)
    assert "parameters.edge.pilot" in text and "PILOT_REPORT.md" in text and "checks" in text and "--per-source" in text
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["id"].endswith("/edge-pilot")
    assert meta["machine_shape"] == "NvidiaTeslaT4" and len(meta["dataset_sources"]) == 3


# ---------------------------------------------------------------- dataset discovery / input tree

def _touch(p, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.mark.parametrize("prefix", ["", "datasets/owner"])
def test_find_dataset_layouts(tmp_path, prefix):
    base = tmp_path / "input" / prefix
    _touch(base / "Semantic-Scheduler-SQGAN-Bundle-2026" / "x_flat_manifest.json")
    _touch(base / "private-sqgan-source" / "sqgan-pinned-source.zip")
    c = pc.Ctx(out=tmp_path, input_base=str(tmp_path / "input"))
    assert pc.find_dataset(c, "*sqgan-bundle*") == base / "Semantic-Scheduler-SQGAN-Bundle-2026"
    assert pc.find_dataset(c, "*sqgan-source*") == base / "private-sqgan-source"
    with pytest.raises(pc.Skip, match="sqgan-assets"):
        pc.find_dataset(c, "*sqgan-assets*")


def test_find_dataset_falls_back_to_manifest_folder(tmp_path):
    _touch(tmp_path / "input" / "weird-name" / "bundle_manifest.json")
    c = pc.Ctx(out=tmp_path, input_base=str(tmp_path / "input"))
    assert pc.find_dataset(c, "*sqgan-bundle*") == tmp_path / "input" / "weird-name"


def test_discover_datasets_lists_dirs_and_files(tmp_path):
    base = tmp_path / "input"
    _touch(base / "datasets" / "o" / "a-sqgan-assets" / "f.txt")
    _touch(base / "datasets" / "o" / "b" / "k_flat_manifest.json")
    _touch(base / "datasets" / "o" / "b" / "sqgan-pinned-source.zip")
    _touch(base / "d1" / "d2" / "d3" / "d4" / "d5" / "bundle_manifest.json")      # deeper than depth 4
    found = pc.discover_datasets(base)
    assert found["dirs"]["sqgan-assets"] == [str(base / "datasets/o/a-sqgan-assets")]
    assert found["files"]["*flat_manifest.json"] == [str(base / "datasets/o/b/k_flat_manifest.json")]
    assert found["files"]["sqgan-pinned-source.zip"] and found["files"]["bundle_manifest.json"] == []


def test_input_tree_depth_and_limit(tmp_path):
    base = tmp_path / "input"
    _touch(base / "a" / "b" / "c" / "deep.txt")
    for i in range(10):
        _touch(base / "many" / ("f%d.txt" % i))
    tree = pc.input_tree(base, depth=3, limit=200)
    assert "a/" in tree and "a/b/" in tree and "a/b/c/" in tree and "a/b/c/deep.txt" not in tree
    assert len(pc.input_tree(base, 3, 5)) == 5
    assert pc.input_tree(tmp_path / "missing") == []


def test_env_gpu_records_input_tree(tmp_path):
    pytest.importorskip("torch")
    _touch(tmp_path / "input" / "datasets" / "o" / "x-sqgan-bundle" / "m_flat_manifest.json")
    out = pc.check_env_gpu(pc.Ctx(out=tmp_path, input_base=str(tmp_path / "input")))
    tree = out["details"]["input_tree"]
    assert "datasets/o/x-sqgan-bundle/" in tree and len(tree) <= 200
