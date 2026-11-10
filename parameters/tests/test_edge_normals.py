"""Everyday-image lists: parsing, blacklist safety, deterministic stratified sampling."""
import csv
import io
import tarfile

import pytest

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


def _csv(rows, fields):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def test_openimages_selfie_first_requires_confidence_one():
    labels = _csv([
        {"ImageID": "a", "Source": "v", "LabelName": MIDS["selfie"], "Confidence": "1"},
        {"ImageID": "b", "Source": "v", "LabelName": MIDS["person"], "Confidence": "1"},
        {"ImageID": "c", "Source": "v", "LabelName": MIDS["person"], "Confidence": "0"},
        {"ImageID": "a", "Source": "v", "LabelName": MIDS["person"], "Confidence": "1"}],
        ["ImageID", "Source", "LabelName", "Confidence"])
    images = _csv([{"ImageID": i, "OriginalURL": "u" + i, "License": "L", "Author": "A"} for i in "abc"],
                  ["ImageID", "OriginalURL", "License", "Author"])
    items = normals.parse_openimages_people(labels, images, MIDS, 5, 1)
    assert [i["id"] for i in items] == ["a", "b"]    # a is a selfie, c has low confidence, no repeats
    assert items[0]["category"] == "selfie" and items[1]["category"] == "person"
    assert len(normals.parse_openimages_people(labels, images, MIDS, 1, 1)) == 1
