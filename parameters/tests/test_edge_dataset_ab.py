"""Stage A/B dataset builders with fake network (no downloads)."""
import copy
import io
import json
import tarfile
import urllib.error

import pytest
from PIL import Image

from Evaluation.edge.edge_config import load_config
from parameters.edge import build_label_dataset as bld
from parameters.edge import dataset_a as da
from parameters.edge import labels


def img(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, "JPEG")
    return buf.getvalue()


def make_tar(path, names):
    with tarfile.open(path, "w") as tar:
        for name in names:
            data = img((256, 256))
            info = tarfile.TarInfo("val_256/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


DEAD = ("/data_image/ev/1.jpg",)


def opener(url, headers):
    if url.endswith(DEAD[0]):
        raise urllib.error.HTTPError(url, 404, "x", {}, None)
    return img()


def _write_img(path, data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data or img())


def kaggle_inputs(tmp_path):
    """Fake /kaggle/input: Places365 (3 classes x 4 files, 'ruin' blacklisted) and COCO (4 big-person images,
    big1.jpg is corrupt, plus a tiny-person image)."""
    inp = tmp_path / "kaggle_input"
    for cls in ("airfield", "kitchen", "ruin"):
        for i in range(4):
            _write_img(inp / "places365" / "train" / cls / ("%08d.jpg" % (i + 1)))
    coco = inp / "coco-2017-dataset" / "coco2017"
    names = ["big%d.jpg" % i for i in range(4)] + ["tiny.jpg"]
    for n in names:
        _write_img(coco / "train2017" / n, b"garbage" if n == "big1.jpg" else None)
    imgs = [{"id": i, "file_name": n, "width": 100, "height": 100} for i, n in enumerate(names)]
    anns = [{"image_id": i, "category_id": 1, "bbox": [0, 0, 50, 50] if i < 4 else [0, 0, 5, 5], "iscrowd": 0}
            for i in range(5)]
    ann = coco / "annotations" / "person_keypoints_train2017.json"
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text(json.dumps({"images": imgs, "annotations": anns}))
    return inp


def world(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config("data"))
    cfg["normals"].update(places365_per_class=2, people_count=4, places365_blacklist=["ruin"])
    src = cfg["normals"]["sources"]
    names = ["Places365_val_%08d.jpg" % i for i in range(1, 13)]
    val_txt = "\n".join("%s %d" % (n, i % 3) for i, n in enumerate(names))
    filelist = tmp_path / "fl.tar"
    with tarfile.open(filelist, "w") as tar:
        info = tarfile.TarInfo("places365_val.txt")
        info.size = len(val_txt)
        tar.addfile(info, io.BytesIO(val_txt.encode()))
    web = {src["places365_categories"]: b"/a/airfield 0\n/k/kitchen 1\n/r/ruin 2\n",
           src["places365_filelist"]: filelist.read_bytes()}
    recs = {}
    for i in range(3):
        p = "data_image/ev/%d.jpg" % i
        recs[p] = {"image_path": p, "image_id": "c%d" % i, "tweet_id": str(i), "event": "ev",
                   "source": "crisismmd", "labels": {"damage": {"severe_damage"}}, "native_splits": {}}
    monkeypatch.setattr(labels, "load_crisismmd", lambda *a, **k: recs)
    meta = tmp_path / "meta"
    meta.mkdir()
    for t in ("informative", "humanitarian", "damage"):
        (meta / (t + "_train.json")).write_text("[]")
    (meta / "other.json").write_text("[]")
    tar = make_tar(tmp_path / "val_256.tar", names)
    return cfg, meta, (lambda url: web[url]), tar


def run_a(tmp_path, monkeypatch, out_name="a", kaggle=True, **kw):
    cfg, meta, fetch, tar = world(tmp_path, monkeypatch)
    kw.setdefault("opener", opener)
    kw.setdefault("input_root", kaggle_inputs(tmp_path) if kaggle else tmp_path / "no_kaggle")
    manifest = da.build_a(meta, tmp_path / out_name, fetch, cfg=cfg, tar_source=tar, workers=2,
                          progress_dir=tmp_path, **kw)
    return cfg, meta, tmp_path / out_name, manifest, fetch, tar


def test_stage_a_index_and_no_labels(tmp_path, monkeypatch):
    cfg, meta, out, manifest, *_ = run_a(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (out / "index.jsonl").read_text().splitlines()]
    uids = [r["uid"] for r in rows]
    assert len(set(uids)) == len(uids)
    for r in rows:
        assert not {"tier", "tier_probs", "damage", "humanitarian", "informative", "labels"} & set(r)
        assert (out / r["file"]).stat().st_size == r["bytes"]
        assert r["sha256"] and r["width"] and r["height"]
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    assert set(by) == {"crisismmd", "places365", "coco_people"}
    assert len(by["crisismmd"]) == 2 and len(by["places365"]) == 4 and len(by["coco_people"]) == 3
    assert {r["category"] for r in by["places365"]} == {"airfield", "kitchen"}     # ruin blacklisted
    assert all(r["file"].startswith("images/places365/%s/" % r["category"]) for r in by["places365"])
    assert all(r["file"].startswith("images/coco_people/") for r in by["coco_people"])
    assert sorted(r["rank"] for r in by["places365"]) == list(range(4))
    keys = {f["key"] for f in json.loads((out / "failures.json").read_text())}
    assert {"data_image/ev/1.jpg", "big1.jpg"} == keys        # corrupt COCO copy is recorded, not indexed
    assert manifest["report"]["places365_mode"].startswith("kaggle:")
    assert "tiny.jpg" not in {r["image_id"] for r in by["coco_people"]}
    meta_json = json.loads((out / "dataset-metadata.json").read_text())
    assert meta_json["id"] == "satyapaladuguplus/scheduler-dataset-v1-images"
    files = json.loads((out / "manifest.json").read_text())["files"]
    assert all(r["file"] in files for r in rows) and "index.jsonl" in files
    assert not list(out.rglob("*.part"))


def test_stage_a_resumes_without_refetching(tmp_path, monkeypatch):
    cfg, meta, out, _, fetch, tar = run_a(tmp_path, monkeypatch)
    calls = []

    def counting(url, headers):
        calls.append(url)
        return opener(url, headers)
    da.build_a(meta, out, fetch, cfg=cfg, opener=counting, tar_source=tar, workers=2, progress_dir=tmp_path,
               input_root=tmp_path / "kaggle_input")
    assert calls and all(u.endswith(DEAD[0]) for u in calls)


def test_stage_a_disk_guard(tmp_path, monkeypatch):
    class Low:
        free = 1
    *_, manifest, _, _ = run_a(tmp_path, monkeypatch, disk_usage=lambda p: Low, min_free_gb=5)
    assert manifest["report"]["stopped"] and manifest["report"]["index_rows"] == 0


def test_stage_a_max_images(tmp_path, monkeypatch):
    *_, manifest, _, _ = run_a(tmp_path, monkeypatch, max_images=1)
    by = manifest["report"]["by_source"]
    # CrisisMMD pilot takes max(8*N, 160) stratified rows (all 3 fixture rows, one is a dead URL); others keep N
    assert by["crisismmd"] == 2 and by["places365"] <= 1 and by["coco_people"] <= 1


def test_crisis_pilot_size():
    assert da.crisis_pilot_size(20) == 160 and da.crisis_pilot_size(5) == 160 and da.crisis_pilot_size(30) == 240


def _synthetic_rows():
    events = ["e%d" % i for i in range(7)]
    rows = []
    for ei, ev in enumerate(events):
        for ti, tier in enumerate(labels.TIERS):
            # skewed: Low dominates and sorts first, rare tiers are few
            n = 200 if tier == "Low" else 3 + ei
            rows += [{"image_path": "%s/%s_%03d.jpg" % (ev, tier, k), "event": ev, "tier": tier} for k in range(n)]
    return events, sorted(rows, key=lambda r: r["image_path"])


def test_stratified_sample_covers_every_tier_and_event_deterministically():
    events, rows = _synthetic_rows()
    pick = da.stratified_sample(rows, 160)
    assert len(pick) == len(set(pick)) == 160
    by = {r["image_path"]: r for r in rows}
    assert {by[p]["tier"] for p in pick} == set(labels.TIERS)
    assert {by[p]["event"] for p in pick} == set(events)
    for ev in events:
        assert {by[p]["tier"] for p in pick if by[p]["event"] == ev} == set(labels.TIERS)
    assert pick == da.stratified_sample(list(reversed(rows)), 160)
    assert len(da.stratified_sample(rows[:5], 160)) == 5


def test_stage_a_fallback_tar_and_coco_unavailable(tmp_path, monkeypatch):
    cfg, meta, out, manifest, *_ = run_a(tmp_path, monkeypatch, kaggle=False)
    rep = manifest["report"]
    assert rep["places365_mode"] == "fallback_val_256_tar" and rep["people_source"] == "unavailable"
    assert rep["unavailable"] and rep["by_source"]["coco_people"] == 0 and rep["by_source"]["places365"] == 4


def test_stage_b_labels_from_a(tmp_path, monkeypatch):
    cfg, meta, a_out, *_ = run_a(tmp_path, monkeypatch)
    out = tmp_path / "b"
    counts = bld.build_b(meta, out, a_out, cfg=cfg)
    rows = [json.loads(x) for x in (out / "labels_raw.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
        assert (a_out / r["file"]).is_file() and r["uid"]
    assert len(by["crisismmd"]) == 2 and len(by["places365"]) == 4 and len(by["coco_people"]) == 3
    assert all(r["tier"] == "Low" for r in by["places365"] + by["coco_people"])
    index = [json.loads(x) for x in (a_out / "index.jsonl").read_text().splitlines()]
    top = sorted((r for r in index if r["source"] == "places365"), key=lambda r: r["rank"])
    assert [r["uid"] for r in by["places365"]] == [r["uid"] for r in top]
    assert not (out / "images").exists()
    assert (out / "normals_coco_people.jsonl").is_file() and (out / "normals_places365.jsonl").is_file()
    assert counts["normals"]["places365_final"] == 4 and counts["normals"]["coco_people_final"] == 3
    assert counts["totals_by_source_tier"]["coco_people"] == {"Low": 3}
    assert sorted(p.name for p in (out / "crisismmd_metadata").iterdir()) == [
        "damage_train.json", "humanitarian_train.json", "informative_train.json"]
    fails = json.loads((out / "failures.json").read_text())
    assert any(f["key"] == "data_image/ev/1.jpg" for f in fails)
    assert "gates" in counts
    assert json.loads((out / "dataset-metadata.json").read_text())["id"].endswith("scheduler-dataset-v1-labels")


def test_stage_b_drops_blacklisted_places_row(tmp_path, monkeypatch):
    cfg, meta, a_out, *_ = run_a(tmp_path, monkeypatch)
    cfg["normals"]["places365_blacklist"] = ["kitchen"]     # stricter list than A used
    bld.build_b(meta, tmp_path / "b", a_out, cfg=cfg)
    rows = [json.loads(x) for x in (tmp_path / "b" / "normals_places365.jsonl").read_text().splitlines()]
    assert rows and {r["category"] for r in rows} == {"airfield"}


def test_discover_images_root(tmp_path):
    (tmp_path / "ds" / "sub").mkdir(parents=True)
    (tmp_path / "ds" / "sub" / "index.jsonl").write_text("")
    assert bld.discover_images_root(tmp_path) == tmp_path / "ds" / "sub"
    with pytest.raises(FileNotFoundError):
        bld.discover_images_root(tmp_path / "zz")


def test_cli_flags_parse():
    for mod in (da, bld):
        with pytest.raises(SystemExit):
            mod.main(["--help"])


def _fake_root(tmp_path, present):
    root = tmp_path / "kaggle" / "CrisisMMD_v2.0"
    for i in present:
        p = root / ("data_image/ev/%d.jpg" % i)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(img())
    return root


def _crisis_opener(calls):
    def op(url, headers):
        calls.append(url)
        return img()
    return op


def test_crisismmd_from_root_no_fallback_under_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "MAX_COPY_FAIL_FRACTION", 0.5)
    root = _fake_root(tmp_path, [0, 1])          # row 2 missing but threshold raised to 50%... 1/3 < 0.5
    calls = []
    *_, out, manifest, _, _ = run_a(tmp_path, monkeypatch, crisismmd_root=root, opener=_crisis_opener(calls))
    assert manifest["report"]["crisismmd_source"] == "kaggle_input:%s" % root
    assert not [u for u in calls if "data_image" in u]
    assert (out / "images/crisismmd/data_image/ev/0.jpg").is_file()


def test_crisismmd_fallback_downloads_only_missing(tmp_path, monkeypatch):
    root = _fake_root(tmp_path, [0])             # rows 0,1,2 expected; 1 and 2 missing -> over 2%
    calls = []
    *_, out, manifest, _, _ = run_a(tmp_path, monkeypatch, crisismmd_root=root, opener=_crisis_opener(calls))
    crisis_calls = [u for u in calls if "data_image" in u]
    assert sorted(u[-len("data_image/ev/1.jpg"):] for u in crisis_calls) == ["data_image/ev/1.jpg", "data_image/ev/2.jpg"]
    assert manifest["report"]["crisismmd_source"].endswith("+huggingface_fallback")
    assert (out / "images/crisismmd/data_image/ev/1.jpg").is_file()


def test_crisismmd_default_is_huggingface(tmp_path, monkeypatch):
    *_, manifest, _, _ = run_a(tmp_path, monkeypatch)
    assert manifest["report"]["crisismmd_source"] == "huggingface"


def test_resolve_root_none_forces_download(tmp_path, monkeypatch):
    monkeypatch.setattr(da.fetch_images, "find_crisismmd_root", lambda: "/found")
    assert da.resolve_crisismmd_root("none") is None
    assert da.resolve_crisismmd_root(None) == "/found"
    assert da.resolve_crisismmd_root(str(tmp_path)) == str(tmp_path)
