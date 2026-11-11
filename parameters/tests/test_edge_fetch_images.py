"""Image fetchers and the image-aware dataset builder; fake opener, no network."""
import io
import json
import tarfile
import urllib.error

import pytest
from PIL import Image

from parameters.edge import fetch_images as fi
from parameters.edge import labels


def img(fmt="JPEG", size=(8, 6), color=(200, 10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def no_sleep(_):
    pass


def test_crisismmd_skips_existing_and_writes_new(tmp_path):
    existing = tmp_path / "images" / "crisismmd" / "d/a.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already")
    calls = []

    def opener(url, headers):
        calls.append(url)
        return img()
    rows = [{"image_path": "d/a.jpg"}, {"image_path": "d/b.jpg"}]
    assert fi.fetch_crisismmd(rows, tmp_path, opener, workers=2) == []
    assert len(calls) == 1 and calls[0].endswith("/d/b.jpg") and "10a5626" in calls[0]
    assert existing.read_bytes() == b"already"
    assert (tmp_path / "images" / "crisismmd" / "d/b.jpg").stat().st_size > 0


def test_retry_on_503_then_success(tmp_path):
    state = {"n": 0}

    def opener(url, headers):
        state["n"] += 1
        if state["n"] < 3:
            raise http_error(503)
        return img()
    fails = fi.fetch_crisismmd([{"image_path": "a.jpg"}], tmp_path, opener, workers=1,
                               base_delay=0, sleep=no_sleep)
    assert fails == [] and state["n"] == 3


def test_non_retryable_and_exhausted_retries_are_recorded(tmp_path):
    def opener(url, headers):
        raise http_error(404 if "gone" in url else 503)
    fails = fi.fetch_crisismmd([{"image_path": "gone.jpg"}, {"image_path": "busy.jpg"}], tmp_path,
                               opener, workers=1, retries=1, base_delay=0, sleep=no_sleep)
    assert {f["key"] for f in fails} == {"gone.jpg", "busy.jpg"}


def test_html_error_page_rejected(tmp_path):
    fails = fi.fetch_crisismmd([{"image_path": "a.jpg"}], tmp_path,
                               lambda u, h: b"<html>429 slow down</html>", workers=1)
    assert len(fails) == 1 and "HTML" in fails[0]["reason"]
    assert not (tmp_path / "images" / "crisismmd" / "a.jpg").exists()


def test_progress_json_flushed(tmp_path):
    prog = tmp_path / "p.json"
    fi.fetch_crisismmd([{"image_path": "a.jpg"}], tmp_path / "o", lambda u, h: img(), progress_path=prog)
    assert json.loads(prog.read_text())["done"] == 1


def make_tar(path, names):
    with tarfile.open(path, "w") as tar:
        for name in names:
            data = img("JPEG", (256, 256))
            info = tarfile.TarInfo("val_256/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_places365_extracts_only_selected(tmp_path):
    names = ["Places365_val_%08d.jpg" % i for i in range(1, 6)]
    tar = make_tar(tmp_path / "val.tar", names)
    items = [{"id": names[1]}, {"id": names[3]}, {"id": "Places365_val_00000099.jpg"}]
    fails = fi.fetch_places365(items, tmp_path / "out", tar)
    got = sorted(p.name for p in (tmp_path / "out" / "images" / "places365").iterdir())
    assert got == sorted([names[1], names[3]])
    assert [f["key"] for f in fails] == ["Places365_val_00000099.jpg"]
    assert tar.exists()


def test_places365_downloaded_tar_deleted_unless_keep(tmp_path):
    names = ["Places365_val_00000001.jpg"]
    made = {}

    def source(dest):
        made["dest"] = make_tar(dest, names)
    fi.fetch_places365([{"id": names[0]}], tmp_path / "out", source)
    assert not made["dest"].exists()
    (tmp_path / "out2").mkdir()
    fi.fetch_places365([{"id": names[0]}], tmp_path / "out2", source, keep=True)
    assert made["dest"].exists()


def test_copy_verified_copies_skips_and_reports(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.jpg").write_bytes(img())
    (src / "bad.jpg").write_bytes(b"not an image")
    pairs = [(src / n, tmp_path / "dst" / n, n) for n in ("ok.jpg", "bad.jpg", "gone.jpg")]
    fails = fi.copy_verified(pairs, workers=2, source="x")
    assert {f["key"]: f["reason"] for f in fails} == {"bad.jpg": "corrupt", "gone.jpg": "missing_in_source"}
    assert (tmp_path / "dst" / "ok.jpg").is_file() and not list((tmp_path / "dst").glob("*.part"))
    assert fi.copy_verified(pairs[:1], workers=1) == []


def test_fill_to_count_reaches_n():
    cands = list(range(10))
    succ, failed = fi.fill_to_count(cands, 4, lambda c: c if c % 3 else None, workers=2)
    assert [c for c, _ in succ] == [1, 2, 4, 5] and 0 in failed and 3 in failed


def test_fill_to_count_runs_out_of_candidates():
    succ, failed = fi.fill_to_count([1, 2], 5, lambda c: c)
    assert len(succ) == 2 and failed == []


def test_verify_images(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images/ok.jpg").write_bytes(img())
    (tmp_path / "images/bad.jpg").write_bytes(b"garbage")
    rows = [{"file": "images/ok.jpg"}, {"file": "images/bad.jpg"}, {"file": "images/none.jpg"}]
    assert fi.verify_images(rows, tmp_path) == (["images/none.jpg"], ["images/bad.jpg"])


def _mk_root(base, names=("data_image/ev/0.jpg", "data_image/ev/1.jpg")):
    for n in names:
        p = base / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(img())
    return base


def test_find_crisismmd_root_both_layouts_and_none(tmp_path):
    assert fi.find_crisismmd_root(tmp_path / "nope") is None
    assert fi.find_crisismmd_root(tmp_path) is None
    a = _mk_root(tmp_path / "a" / "slug" / "CrisisMMD_v2.0")
    assert fi.find_crisismmd_root(tmp_path / "a") == a
    b = _mk_root(tmp_path / "b" / "datasets" / "owner" / "slug" / "CrisisMMD_v2.0")
    assert fi.find_crisismmd_root(tmp_path / "b") == b
    (tmp_path / "c" / "x" / "data_image").mkdir(parents=True)   # no event subfolder
    assert fi.find_crisismmd_root(tmp_path / "c") is None
    assert fi.find_crisismmd_root(tmp_path / "b", max_depth=2) is None


def test_find_crisismmd_root_ambiguous(tmp_path):
    _mk_root(tmp_path / "x" / "CrisisMMD_v2.0")
    _mk_root(tmp_path / "y" / "CrisisMMD_v2.0")
    with pytest.raises(RuntimeError, match="several"):
        fi.find_crisismmd_root(tmp_path)


def test_copy_crisismmd_skips_missing_and_corrupt(tmp_path):
    root = _mk_root(tmp_path / "src")
    (root / "data_image/ev/2.jpg").write_bytes(b"not an image")
    rows = [{"image_path": "data_image/ev/%d.jpg" % i} for i in range(4)]
    out = tmp_path / "out"
    fails, copied = fi.copy_crisismmd_from_root(rows, root, out, workers=2)
    reasons = {f["key"]: f["reason"] for f in fails}
    assert reasons == {"data_image/ev/2.jpg": "corrupt", "data_image/ev/3.jpg": "missing_in_source"}
    assert copied == 2 * len(img())
    assert not list(out.rglob("*.part")) and not (out / "images/crisismmd/data_image/ev/2.jpg").exists()
    fails2, copied2 = fi.copy_crisismmd_from_root(rows[:2], root, out)
    assert fails2 == [] and copied2 == 0      # same-size files skipped
