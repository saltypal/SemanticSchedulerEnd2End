"""Everyday-image lists: parsing, blacklist safety, deterministic stratified sampling."""
import io
import json
import logging
import tarfile

from PIL import Image

import pytest
from pathlib import Path

from parameters.edge import normals

CATS = "/a/airfield 0\n/k/kitchen 1\n/r/ruin 2\n/a/arena/hockey 3\n"
MIDS = {"selfie": "/m/0sgh53y", "person": "/m/01g317"}


def test_categories_parse_nested_names():
    assert normals.parse_places_categories(CATS) == ["airfield", "kitchen", "ruin", "arena/hockey"]


def test_blacklist_typo_raises():
    cats = normals.parse_places_categories(CATS)
    normals.check_blacklist(["ruin"], cats)
    with pytest.raises(ValueError):
        normals.check_blacklist(["ruins"], cats)


def _val(n_per=5):
    lines = ["Places365_val_%08d.jpg %d" % (i * 4 + c + 1, c) for i in range(n_per) for c in range(4)]
    return "\n".join(lines)


def test_sampling_drops_blacklist_and_is_deterministic_and_stratified():
    cats = normals.parse_places_categories(CATS)
    items = normals.parse_places_val(_val(), cats)
    a = normals.sample_places(items, {"ruin"}, 9, seed=1)
    b = normals.sample_places(items, {"ruin"}, 9, seed=1)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert all(x["category"] != "ruin" for x in a)
    kept = {"airfield", "kitchen", "arena/hockey"}
    assert {x["category"] for x in a} == kept
    counts = [sum(1 for x in a if x["category"] == c) for c in kept]
    assert max(counts) - min(counts) <= 1


def test_tar_val_extraction():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"Places365_val_00000001.jpg 0\n"
        info = tarfile.TarInfo("places365_val.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    assert normals.places_val_from_tar(buf.getvalue()).startswith("Places365_val_00000001")


def _jpg(path, size=(8, 6)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 200, 10)).save(path, "JPEG")


def _places(tmp_path, layout="flat", classes=("airfield", "kitchen", "ruin", "arena/hockey"), n=5, txt=False, split="train"):
    base = tmp_path / "in" / ("places365" if layout == "flat" else "places365/places365")
    lines = []
    for c in classes:
        for i in range(n):
            name = "%08d.jpg" % (i + 1)
            _jpg(base / split / c / name)
            lines.append("/%s/%s %d" % (c, name, 0))
    if txt:
        (base / "train.txt").write_text("\n".join(lines))
    return tmp_path / "in", base


def test_places_val_split_preferred_and_report_field(tmp_path):
    inp, base = _places(tmp_path, split="val")
    for c in ("airfield", "kitchen"):
        _jpg(base / "train" / c / "99999999.jpg")
    assert normals.discover_places_root(inp) == base
    items, info = normals.sample_places_kaggle(base, 2, [], seed=1)
    assert info["places365_split_used"] == "val" and not info["places365_used_train_txt"]
    assert all(it["rel"].startswith("val/") and Path(it["src"]).is_file() for it in items)
    items_t, info_t = normals.sample_places_kaggle(base, 2, [], seed=1, split="train")
    assert info_t["places365_split_used"] == "train" and all(it["rel"].startswith("train/") for it in items_t)


def test_places_split_falls_back_to_train(tmp_path):
    inp, base = _places(tmp_path, split="train", txt=True)
    items, info = normals.sample_places_kaggle(base, 2, [], seed=1, split="val")
    assert info["places365_split_used"] == "train" and info["places365_used_train_txt"]
    assert items and all(it["rel"].startswith("train/") for it in items)


def test_places_root_discovery_two_layouts(tmp_path):
    for layout in ("flat", "nested"):
        sub = tmp_path / layout
        sub.mkdir()
        inp, base = _places(sub, layout)
        assert normals.discover_places_root(inp) == base
    assert normals.discover_places_root(tmp_path / "missing") is None


def test_places_per_class_covers_all_classes_and_blacklist(tmp_path):
    inp, base = _places(tmp_path)
    items, info = normals.sample_places_kaggle(base, 3, ["ruin"], seed=1)
    by = {}
    for it in items:
        by.setdefault(it["category"], []).append(it)
    assert set(by) == {"airfield", "kitchen", "arena/hockey"} and all(len(v) == 3 for v in by.values())
    assert info["places365_blacklisted_classes"] == 1 and info["places365_blacklist_unmatched"] == []
    assert all(Path(it["src"]).is_file() for it in items)


def test_places_per_class_caps_small_classes(tmp_path):
    inp, base = _places(tmp_path, n=2)
    items, _ = normals.sample_places_kaggle(base, 17, [], seed=1)
    assert len(items) == 8


def test_blacklist_nested_forms_and_unmatched_warning(tmp_path, caplog):
    inp, base = _places(tmp_path)
    with caplog.at_level(logging.WARNING):
        items, info = normals.sample_places_kaggle(base, 2, ["arena_hockey", "nothing_here"], seed=1)
    assert "arena/hockey" not in {i["category"] for i in items}
    assert info["places365_blacklist_unmatched"] == ["nothing_here"] and "nothing_here" in caplog.text
    items2, _ = normals.sample_places_kaggle(base, 2, ["arena/hockey"], seed=1)
    assert items == items2


def test_train_txt_and_scandir_agree_and_deterministic(tmp_path):
    a, base_a = _places(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, txt=False)
    (tmp_path / "b").mkdir()
    b, base_b = _places(tmp_path / "b", txt=True)
    ia, _ = normals.sample_places_kaggle(base_a, 3, ["ruin"], seed=7)
    ib, info = normals.sample_places_kaggle(base_b, 3, ["ruin"], seed=7)
    assert info["places365_used_train_txt"]
    assert [i["id"] for i in ia] == [i["id"] for i in ib]
    ic, _ = normals.sample_places_kaggle(base_a, 3, ["ruin"], seed=8)
    assert [i["id"] for i in ia] != [i["id"] for i in ic]


def _coco(tmp_path, with_ann=True, with_train=True):
    root = tmp_path / "in" / "coco-2017-dataset" / "coco2017"
    folder = "train2017" if with_train else "val2017"
    names = ["big.jpg", "tiny.jpg", "none.jpg", "big2.jpg"]
    for n in names:
        _jpg(root / folder / n, (100, 100))
    imgs = [{"id": i, "file_name": n, "width": 100, "height": 100} for i, n in enumerate(names)]
    anns = [{"image_id": 0, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
            {"image_id": 3, "category_id": 1, "bbox": [0, 0, 30, 30], "iscrowd": 0}]
    if with_ann:
        ann = root.parent / "coco2017" / "annotations" / ("person_keypoints_%s.json" % folder)
        ann.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text(json.dumps({"images": imgs, "annotations": anns}))
    return tmp_path / "in", root


def test_coco_discovery_and_person_area_filter(tmp_path):
    inp, root = _coco(tmp_path)
    coco = normals.find_coco(inp)
    assert coco["train_dir"] == root / "train2017" and coco["keypoints_train"].name == "person_keypoints_train2017.json"
    items, part = normals.select_people(coco, 10, 0.05, seed=1)
    assert sorted(i["id"] for i in items) == ["big.jpg", "big2.jpg"]      # tiny person and no person excluded
    assert part["people_source"].startswith("train2017")
    assert len(normals.select_people(coco, 1, 0.05, seed=1)[0]) == 1      # people_count cap


def test_coco_missing_annotation_falls_back_to_val(tmp_path, caplog):
    inp, root = _coco(tmp_path, with_ann=False, with_train=False)
    with caplog.at_level(logging.WARNING):
        items, part = normals.select_people(normals.find_coco(inp), 10, 0.05, seed=1)
    assert len(items) == 4 and "val2017" in part["people_source"] and "falling back" in caplog.text


def test_coco_not_mounted_is_skipped_not_fatal(tmp_path):
    assert normals.find_coco(tmp_path / "nothing") is None
    items, part = normals.select_people(None, 5, 0.05, seed=1)
    assert items == [] and part["people_source"] == "unavailable" and part["unavailable"]


def test_build_normals_fallback_mode_when_places_not_mounted(tmp_path):
    cfg = {"places365_per_class": 2, "places365_blacklist": ["ruin"], "seed": 1, "people_count": 3,
           "coco_min_person_area_frac": 0.05,
           "sources": {"places365_categories": "cat", "places365_filelist": "tar"}}
    val = "\n".join("Places365_val_%08d.jpg %d" % (i + 1, i % 3) for i in range(12))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("places365_val.txt")
        info.size = len(val)
        tar.addfile(info, io.BytesIO(val.encode()))
    web = {"cat": b"/a/airfield 0\n/k/kitchen 1\n/r/ruin 2\n", "tar": buf.getvalue()}
    places, people, rep = normals.build_normals(lambda u: web[u], cfg, input_root=tmp_path / "none")
    assert rep["places365_mode"] == "fallback_val_256_tar" and len(places) == 4
    assert people == [] and rep["people_source"] == "unavailable"


def test_build_normals_kaggle_mode(tmp_path):
    inp, _ = _places(tmp_path)
    _coco(tmp_path)
    cfg = {"places365_per_class": 2, "places365_blacklist": ["ruin"], "seed": 1, "people_count": 5,
           "coco_min_person_area_frac": 0.05, "sources": {}}
    places, people, rep = normals.build_normals(None, cfg, input_root=inp)
    assert rep["places365_mode"].startswith("kaggle:") and len(places) == 6 and len(people) == 2
    assert len(normals.build_normals(None, cfg, input_root=inp, max_images=2)[0]) == 2
