import json
import types
import zipfile
from pathlib import Path

import pytest
import yaml

from Codec.edge import bundle_check as bc
from Codec.edge import seg_map as sm
from Codec.edge import sqgan_finetune as ft
from Codec.integration.sqgan_payload import COMPONENTS
from scripts.edge import make_sqgan_finetune_notebook as mk

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "config/edge/sqgan_finetune.yaml"
SEG = ROOT / "config/edge/seg_classes.yaml"


def _jpeg(path, size):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path, "JPEG")


def make_world(tmp_path):
    """images dataset + labels dataset; val/test rows point at files that do NOT exist."""
    imgs, labs = tmp_path / "in" / "imgs", tmp_path / "in" / "labels"
    rows = []
    for i, ev, split in [(0, "hurricane_harvey", "train"), (1, "hurricane_harvey", "train"),
                         (2, "california_wildfires", "val"), (3, "hurricane_maria", "test")]:
        rel = "images/crisismmd/a/%d.jpg" % i
        if split == "train":
            _jpeg(imgs / rel, (600, 300))
        rows.append({"uid": "crisismmd:%d" % i, "file": rel, "event": ev, "source": "crisismmd",
                     "image_path": "a/%d.jpg" % i})
    _jpeg(imgs / "images/crisismmd/a/small.jpg", (100, 100))
    rows.append({"uid": "crisismmd:small", "file": "images/crisismmd/a/small.jpg", "event": "hurricane_harvey",
                 "source": "crisismmd"})
    (imgs / "index.jsonl").write_text("{}\n")
    labs.mkdir(parents=True)
    (labs / "labels_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return imgs, labs, rows


def test_config_loads_and_overrides():
    cfg = ft.load_config(CFG, ["train.epochs=2"])
    assert cfg["train"]["epochs"] == 2 and cfg["source"]["pinned_commit"] and cfg["codec"]["sts_profile"] == 2
    with pytest.raises(KeyError):
        ft.load_config(CFG, ["train.nope=1"])


def test_discover_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        ft.discover_root(tmp_path, "labels_raw.jsonl")
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "labels_raw.jsonl").write_text("")
    assert ft.discover_root(tmp_path, "labels_raw.jsonl") == tmp_path / "a" / "b"
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "labels_raw.jsonl").write_text("")
    with pytest.raises(ValueError):
        ft.discover_root(tmp_path, "labels_raw.jsonl")
    deep = tmp_path / "x" / "1" / "2" / "3" / "4"
    deep.mkdir(parents=True)
    (deep / "index.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):       # 4 levels is beyond the search depth
        ft.discover_root(tmp_path / "x", "index.jsonl")


def test_split_guard_refuses_without_file(tmp_path):
    _, labs, rows = make_world(tmp_path)
    with pytest.raises(FileNotFoundError, match="allow-provisional"):
        ft.resolve_splits(rows, labs / "splits_v1.json", False)


def test_provisional_splits_warn_and_use_events(tmp_path, capsys):
    _, labs, rows = make_world(tmp_path)
    splits, kind = ft.resolve_splits(rows, labs / "splits_v1.json", True)
    assert kind == "provisional" and "WARNING" in capsys.readouterr().out
    assert splits["crisismmd:0"] == "train" and splits["crisismmd:2"] == "val" and splits["crisismmd:3"] == "test"


def test_splits_file_formats(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]}))
    assert ft.load_splits(p) == {"a": "train", "b": "val", "c": "test"}
    p.write_text(json.dumps({"splits": {"a": "train"}}))
    assert ft.load_splits(p) == {"a": "train"}
    p.write_text(json.dumps({"a": "dev"}))
    with pytest.raises(ValueError):
        ft.load_splits(p)


def test_manifest_train_only_never_reads_val_test(tmp_path, monkeypatch):
    imgs, labs, rows = make_world(tmp_path)
    splits, _ = ft.resolve_splits(rows, labs / "x.json", True)
    opened = []
    real = ft.image_size
    monkeypatch.setattr(ft, "image_size", lambda p: opened.append(Path(p).name) or real(p))
    train = ft.select_train_rows(rows, splits, ["crisismmd"])
    man = ft.build_manifest(train, imgs, {"min_width": 512, "min_height": 256, "small_image_policy": "skip"},
                            tmp_path / "w")
    assert "2.jpg" not in opened and "3.jpg" not in opened          # nonexistent val/test files never touched
    assert man["counts"] == {"train": 2, "val": 1, "test": 0} and man["skipped_too_small"] == ["crisismmd:small"]
    ft.assert_train_only(man, splits)
    assert {r["uid"] for r in man["images"]} <= {"crisismmd:0", "crisismmd:1"}
    bad = dict(splits, **{"crisismmd:0": "val"})
    with pytest.raises(AssertionError):
        ft.assert_train_only(man, bad)


def test_manifest_matches_trainer_identity_format(tmp_path):
    import hashlib
    imgs, labs, rows = make_world(tmp_path)
    splits, _ = ft.resolve_splits(rows, labs / "x.json", True)
    man = ft.build_manifest(ft.select_train_rows(rows, splits, ["crisismmd"]), imgs,
                            {"min_width": 512, "min_height": 256, "small_image_policy": "upscale"}, tmp_path / "w")
    canon = json.dumps(man["images"], sort_keys=True, separators=(",", ":")).encode()   # dataset.py:110
    assert hashlib.sha256(canon).hexdigest() == man["identity"]
    assert any(r["source"] == "upscaled" for r in man["images"]) and "upscaled" in man["roots"]
    assert all(r["width"] >= 512 and r["height"] >= 256 for r in man["images"])


def test_class_map_file_valid_and_has_water():
    cfg = sm.load_class_map()
    index = sm.build_index(cfg)
    names = [c["name"] for c in cfg["classes"]]
    assert len(index) == 150 and len(names) == 20 and names[19] == "null"
    ade = {r["ade"]: r["to"] for r in cfg["ade_to_project"]}
    assert ade["water"] == "water" and ade["sea"] == "water" and ade["river"] == "water"
    assert "# DECISION:" in SEG.read_text()


def test_class_map_validation_rejects_bad():
    cfg = yaml.safe_load(SEG.read_text())
    cfg["ade_to_project"][0]["to"] = "nonexistent"
    with pytest.raises(ValueError):
        sm.validate(cfg)
    cfg = yaml.safe_load(SEG.read_text())
    cfg["classes"][6]["name"] = "x"
    with pytest.raises(ValueError):
        sm.validate(cfg)


def test_project_logits_synthetic():
    torch = pytest.importorskip("torch")
    cfg = sm.load_class_map()
    index = sm.build_index(cfg)
    ade_id = {r["ade"]: r["id"] for r in cfg["ade_to_project"]}
    logits = torch.full((1, 150, 2, 2), -10.0)
    logits[0, ade_id["sea"], 0, 0] = 10.0
    logits[0, ade_id["road"], 0, 1] = 10.0
    logits[0, ade_id["bed"], 1, 0] = 10.0
    logits[0, ade_id["traffic light"], 1, 1] = 10.0
    out = sm.project_logits(logits, index)
    names = [c["name"] for c in cfg["classes"]]
    assert out.shape == (1, 20, 2, 2)
    ids = out.argmax(1)[0]
    assert names[int(ids[0, 0])] == "water" and names[int(ids[0, 1])] == "road"
    assert int(ids[1, 0]) == 19 and int(ids[1, 1]) == 6
    with pytest.raises(ValueError):
        sm.project_logits(torch.zeros(1, 19, 1, 1), index)


def test_check_teacher_labels():
    cfg = sm.load_class_map()
    good = {r["id"]: r["ade"] for r in cfg["ade_to_project"]}
    sm.check_teacher_labels(cfg, good)
    good[21] = "lava"
    with pytest.raises(ValueError):
        sm.check_teacher_labels(cfg, good)


def fake_trainer(options, run_dir, ctx):
    """Stand-in trainer: writes a tiny torch-zip-like last.pt containing the state-dict key strings."""
    (run_dir / "checkpoints").mkdir(parents=True)
    keys = b"".join(b"core." + c.encode() + b".w" for c in COMPONENTS) + b"teacher.x"
    with zipfile.ZipFile(run_dir / "checkpoints" / "last.pt", "w") as z:
        z.writestr("archive/data.pkl", keys)
    (run_dir / "training_summary.json").write_text("{}")
    assert options["algorithm"] == "sqgan" and Path(options["manifest"]).is_file()
    return {"last_pt": run_dir / "checkpoints" / "last.pt", "teacher_revision": "a" * 40}


def make_ckpts(tmp_path):
    d = tmp_path / "assets"
    d.mkdir()
    for c in COMPONENTS:
        (d / (c + ".ckpt")).write_bytes(c.encode())
    return d


def test_full_pipeline_with_fake_trainer_and_bundle_check(tmp_path, capsys):
    imgs, labs, rows = make_world(tmp_path)
    cfg = ft.load_config(CFG, ["output.out_dir=%s" % (tmp_path / "out").as_posix(),
                               "output.bundle_dir=%s" % (tmp_path / "out" / "bundle").as_posix()])
    args = types.SimpleNamespace(images_root=str(imgs), labels_root=str(labs), splits=None,
                                 checkpoint_dir=str(make_ckpts(tmp_path)), allow_provisional_splits=True)
    bundle = ft.run_pipeline(cfg, args, trainer=fake_trainer, n_gpus=2)
    res = bc.verify_bundle(bundle, deep=True)
    assert res["ok"], res["errors"]
    m = json.loads((bundle / "bundle.json").read_text())
    assert m["algorithm"] == "sqgan" and m["pipeline_completed"] is True
    assert {"last.pt", "portable_config.json", "assets/checkpoint/encoder.ckpt"} <= {f["path"] for f in m["files"]}
    pc = json.loads((bundle / "portable_config.json").read_text())
    assert pc["teacher_revision"] == "a" * 40 and pc["data_parallel"] is True
    values = bc.codec_yaml_values(bundle)
    assert values["last_pt_sha256"] == bc.sha256(bundle / "last.pt") and values["image_fraction"] == 0.25
    capsys.readouterr()
    assert bc.main([str(bundle), "--print-codec-yaml"]) == 0
    assert "last_pt_sha256" in capsys.readouterr().out
    (bundle / "last.pt").write_bytes(b"corrupt")            # corruption is detected
    assert not bc.verify_bundle(bundle)["ok"]


def test_bundle_check_missing_component(tmp_path):
    imgs, labs, rows = make_world(tmp_path)
    cfg = ft.load_config(CFG)
    run = tmp_path / "r"
    fake_trainer({"algorithm": "sqgan", "manifest": str(labs / "labels_raw.jsonl")}, run, {})
    b = ft.write_bundle(run, tmp_path / "b", make_ckpts(tmp_path), {"teacher_model": "m", "teacher_revision": "main"},
                        cfg, {"last_pt": run / "checkpoints" / "last.pt"}, SEG)
    manifest = json.loads((b / "bundle.json").read_text())
    manifest["files"] = [f for f in manifest["files"] if f["path"] != "assets/checkpoint/quantize.ckpt"]
    (b / "bundle.json").write_text(json.dumps(manifest))
    assert any("quantize.ckpt" in e for e in bc.verify_bundle(b)["errors"])


def test_trainer_options_dataparallel_rules():
    cfg = ft.load_config(CFG)
    assert ft.trainer_options(cfg, "m.json", "c", 2)["data_parallel"] is True
    assert ft.trainer_options(cfg, "m.json", "c", 1)["data_parallel"] is False
    cfg["train"]["batch_size"] = 1
    with pytest.raises(ValueError):
        ft.trainer_options(cfg, "m.json", "c", 2)


def test_obtain_source_pin_and_fallback(tmp_path):
    cfg = ft.load_config(CFG, ["source.local_dir=%s" % (tmp_path / "sl").as_posix(),
                               "source.fallback_search_root=%s" % (tmp_path / "in").as_posix()])
    pin = cfg["source"]["pinned_commit"]
    calls = []
    path, banner = ft.obtain_source(cfg, git=lambda a, cwd=None: calls.append(a) or (pin if a[0] == "rev-parse" else ""))
    assert pin in banner and ["checkout", "-q", pin] in calls

    def boom(a, cwd=None):
        raise RuntimeError("private repo")
    with pytest.raises(RuntimeError, match="attach a dataset"):
        ft.obtain_source(cfg, git=boom)
    src = tmp_path / "in" / "d" / "Semantic_Lossless" / "src"
    src.mkdir(parents=True)
    (src / "sqgan_train.py").write_text("")
    path, banner = ft.obtain_source(cfg, git=boom)
    assert "NOT verified" in banner and path.name == "Semantic_Lossless"


def test_notebook_spec():
    nb, meta = mk.build()
    assert meta["dataset_sources"] == list(mk.DATASET_SOURCES) and len(meta["dataset_sources"]) == 5
    assert "satyapaladuguplus/scheduler-dataset-v1" not in meta["dataset_sources"] and meta["kernel_sources"] == []
    assert meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "Codec.edge.sqgan_finetune" in src and "--stage source" in src
