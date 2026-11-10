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


DEAD = ("/data_image/ev/1.jpg", "http://orig/oi0", "http://orig/oi1", "http://thumb/oi1")


def opener(url, headers):
    if url.endswith(DEAD[0]) or url in DEAD[1:]:
        raise urllib.error.HTTPError(url, 404, "x", {}, None)
    return img()


def world(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config("data"))
    cfg["normals"].update(places365_count=3, people_count=3, oversample=2.0, places365_blacklist=["ruin"])
    src = cfg["normals"]["sources"]
    names = ["Places365_val_%08d.jpg" % i for i in range(1, 13)]
    val_txt = "\n".join("%s %d" % (n, i % 3) for i, n in enumerate(names))
    filelist = tmp_path / "fl.tar"
    with tarfile.open(filelist, "w") as tar:
        info = tarfile.TarInfo("places365_val.txt")
        info.size = len(val_txt)
        tar.addfile(info, io.BytesIO(val_txt.encode()))
    ids = ["oi%d" % i for i in range(8)]
    lab = "ImageID,LabelName,Confidence\n" + "\n".join("%s,/m/0sgh53y,1" % i for i in ids) + "\n"
    im = "ImageID,OriginalURL,License,Author,Thumbnail300KURL\n" + "\n".join(
        "%s,http://orig/%s,lic,au,http://thumb/%s" % (i, i, i) for i in ids) + "\n"
    web = {src["places365_categories"]: b"/a/airfield 0\n/k/kitchen 1\n/r/ruin 2\n",
           src["places365_filelist"]: filelist.read_bytes(),
           src["openimages_labels"]: lab.encode(), src["openimages_images"]: im.encode()}
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


def run_a(tmp_path, monkeypatch, out_name="a", **kw):
    cfg, meta, fetch, tar = world(tmp_path, monkeypatch)
    kw.setdefault("opener", opener)
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
    assert len(by["crisismmd"]) == 2 and len(by["places365"]) == 6 and len(by["openimages_people"]) == 5
    assert sorted(r["rank"] for r in by["places365"]) == list(range(6))
    keys = {f["key"] for f in json.loads((out / "failures.json").read_text())}
    assert {"data_image/ev/1.jpg", "oi1"} == keys  # oi0 recovered via thumbnail
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
    da.build_a(meta, out, fetch, cfg=cfg, opener=counting, tar_source=tar, workers=2, progress_dir=tmp_path)
    assert calls and all(u.endswith(DEAD[0]) or u in DEAD[1:] for u in calls)


def test_stage_a_disk_guard(tmp_path, monkeypatch):
    class Low:
        free = 1
    *_, manifest, _, _ = run_a(tmp_path, monkeypatch, disk_usage=lambda p: Low, min_free_gb=5)
    assert manifest["report"]["stopped"] and manifest["report"]["index_rows"] == 0


def test_stage_a_max_images(tmp_path, monkeypatch):
    *_, manifest, _, _ = run_a(tmp_path, monkeypatch, max_images=1)
    by = manifest["report"]["by_source"]
    assert by["crisismmd"] == 1 and by["places365"] <= 2 and by["openimages_people"] <= 2


def test_stage_b_labels_from_a(tmp_path, monkeypatch):
    cfg, meta, a_out, *_ = run_a(tmp_path, monkeypatch)
    cfg["normals"].update(places365_count=2, people_count=2)
    out = tmp_path / "b"
    counts = bld.build_b(meta, out, a_out, cfg=cfg)
    rows = [json.loads(x) for x in (out / "labels_raw.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
        assert (a_out / r["file"]).is_file() and r["uid"]
    assert len(by["crisismmd"]) == 2 and len(by["places365"]) == 2 and len(by["openimages_people"]) == 2
    index = [json.loads(x) for x in (a_out / "index.jsonl").read_text().splitlines()]
    top = sorted((r for r in index if r["source"] == "places365"), key=lambda r: r["rank"])[:2]
    assert [r["uid"] for r in by["places365"]] == [r["uid"] for r in top]
    assert not (out / "images").exists()
    assert sorted(p.name for p in (out / "crisismmd_metadata").iterdir()) == [
        "damage_train.json", "humanitarian_train.json", "informative_train.json"]
    fails = json.loads((out / "failures.json").read_text())
    assert any(f["key"] == "data_image/ev/1.jpg" for f in fails)
    assert "gates" in counts
    assert json.loads((out / "dataset-metadata.json").read_text())["id"].endswith("scheduler-dataset-v1-labels")


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
