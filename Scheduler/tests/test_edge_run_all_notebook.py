import json
import re

from scripts.edge import make_run_all_notebook as mk


def _src(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_metadata():
    nb, meta = mk.build()
    assert meta["id"] == "satyapaladuguplus/dataset-semschedule" and meta["title"] == "Dataset SemSchedule"
    assert re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") == meta["id"].split("/")[1]
    assert meta["code_file"] == "dataset-semschedule.ipynb"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert not meta["kernel_sources"]
    assert set(meta["dataset_sources"]) == {
        "seaninggg/crisismmd-multimodal-crisis-dataset", "benjaminkz/places365", "awsaf49/coco-2017-dataset",
        "satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102",
        "satyapaladuguplus/semantic-lossless-private-sqgan-source",
        "satyapaladuguplus/semantic-lossless-private-sqgan-assets"}
    assert len(meta["dataset_sources"]) == 6


def test_mode_default_and_full(tmp_path):
    assert mk.MODE == "pilot"
    assert "'--mode', 'pilot'" in _src(mk.build()[0])
    assert "'--mode', 'full'" in _src(mk.build("full")[0])
    mk.main(["--mode", "full", "--out-dir", str(tmp_path)])
    nb = json.loads((tmp_path / "dataset-semschedule.ipynb").read_text())
    assert "'--mode', 'full'" in _src(nb)
    assert (tmp_path / "dataset-semschedule.kernel-metadata.json").exists()


def test_cells_are_quiet_and_collect_outputs():
    text = _src(mk.build()[0])
    for needle in ("-q", "--ignore-requires-python", "TQDM_DISABLE", "--no-progress", "parameters.edge.run_all",
                   "PILOT_REPORT.md", "RUN_SUMMARY.md", "logs"):
        assert needle in text, needle
    assert "tqdm" not in text.replace("TQDM_DISABLE", "").lower()
