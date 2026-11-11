import json
import re

from scripts.edge import make_run_all_notebook as mk

SQGAN_SOURCES = {"satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102",
                 "satyapaladuguplus/semantic-lossless-private-sqgan-source",
                 "satyapaladuguplus/semantic-lossless-private-sqgan-assets"}


AB_PUBLIC = {"satyapaladuguplus/scheduler-dataset-v1-images", "satyapaladuguplus/scheduler-dataset-v1-labels"}


def _src(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _slug_ok(meta):
    return re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") == meta["id"].split("/")[1]


def test_metadata_notebook1_has_no_sqgan_sources():
    nb, meta = mk.build()
    assert meta["id"] == "satyapaladuguplus/dataset-semschedule" and meta["title"] == "Dataset SemSchedule"
    assert _slug_ok(meta) and meta["code_file"] == "dataset-semschedule.ipynb"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert not meta["kernel_sources"]
    assert set(meta["dataset_sources"]) == {
        "seaninggg/crisismmd-multimodal-crisis-dataset", "benjaminkz/places365", "awsaf49/coco-2017-dataset"}
    assert "'--which', 'ab-embed'" in _src(nb)


def test_metadata_notebook2_sqgan():
    nb, meta = mk.build(which="sqgan")
    assert meta["id"] == "satyapaladuguplus/dataset-semschedule-sqgan" and meta["title"] == "Dataset SemSchedule Sqgan"
    assert _slug_ok(meta) and meta["code_file"] == "dataset-semschedule-sqgan.ipynb"
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert set(meta["dataset_sources"]) == SQGAN_SOURCES | AB_PUBLIC and len(meta["dataset_sources"]) == 5
    assert meta["kernel_sources"] == []
    assert mk.build(which="sqgan", account="win")[1]["id"] == "satyapaladugu/dataset-semschedule-sqgan"
    text = _src(nb)
    assert "'--which', 'sqgan'" in text and "dataset-semschedule" in text


def test_mode_default_and_full(tmp_path):
    assert mk.MODE == "pilot"
    assert "'--mode', 'pilot'" in _src(mk.build()[0])
    assert "'--mode', 'full'" in _src(mk.build("full")[0])
    mk.main(["--mode", "full", "--out-dir", str(tmp_path)])
    nb = json.loads((tmp_path / "dataset-semschedule.ipynb").read_text())
    assert "'--mode', 'full'" in _src(nb)
    assert (tmp_path / "dataset-semschedule.kernel-metadata.json").exists()
    mk.main(["--which", "sqgan", "--mode", "full", "--out-dir", str(tmp_path)])
    nb2 = json.loads((tmp_path / "dataset-semschedule-sqgan.ipynb").read_text())
    assert "'--mode', 'full'" in _src(nb2) and "'--which', 'sqgan'" in _src(nb2)
    meta2 = json.loads((tmp_path / "dataset-semschedule-sqgan.kernel-metadata.json").read_text())
    assert meta2["kernel_sources"] == []


def test_cells_are_quiet_and_collect_outputs():
    for which in ("ab-embed", "sqgan"):
        text = _src(mk.build(which=which)[0])
        for needle in ("-q", "--ignore-requires-python", "TQDM_DISABLE", "--no-progress", "parameters.edge.run_all",
                       "PILOT_REPORT.md", "RUN_SUMMARY.md", "logs"):
            assert needle in text, (which, needle)
        assert "tqdm" not in text.replace("TQDM_DISABLE", "").lower()


def test_sqgan_full_mode_attaches_v1_bundle():
    meta = mk.build(mode="full", which="sqgan", account="win")[1]
    assert "satyapaladugu/scheduler-sqgan-v1" in meta["dataset_sources"]
    assert "satyapaladuguplus/scheduler-dataset-v1-images" in meta["dataset_sources"]
    assert "satyapaladugu/scheduler-sqgan-v1" not in mk.build(mode="pilot", which="sqgan")[1]["dataset_sources"]
