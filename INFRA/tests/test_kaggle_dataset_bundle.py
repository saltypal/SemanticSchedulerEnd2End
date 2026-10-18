from pathlib import Path
from zipfile import ZipFile

import pytest

from INFRA.kaggle_dataset_bundle import resolve_ladi_manifest


def test_resolves_loose_manifest_without_extracting(tmp_path):
    manifest = tmp_path / "ladi60_semantic_scheduler_v1.csv"
    manifest.write_text("image_id\n", encoding="utf-8")
    assert resolve_ladi_manifest(manifest, tmp_path / "unused") == manifest
    assert not (tmp_path / "unused").exists()


def test_extracts_manual_upload_zip_with_flat_files(tmp_path):
    bundle = tmp_path / "kaggle-input"
    bundle.mkdir()
    with ZipFile(bundle / "dataset.zip", "w") as archive:
        archive.writestr("ladi60_semantic_scheduler_v1.csv", "image_id\n")
        archive.writestr("payload_costs.csv", "image_id,channel_uses\n")
    output = tmp_path / "extracted"
    resolved = resolve_ladi_manifest(
        bundle / "ladi60_semantic_scheduler_v1.csv", output)
    assert resolved == output / "ladi60_semantic_scheduler_v1.csv"
    assert (output / "payload_costs.csv").is_file()


def test_rejects_zip_paths_outside_flat_bundle(tmp_path):
    bundle = tmp_path / "kaggle-input"
    bundle.mkdir()
    with ZipFile(bundle / "dataset.zip", "w") as archive:
        archive.writestr("../ladi60_semantic_scheduler_v1.csv", "image_id\n")
    with pytest.raises(ValueError, match="unique flat filenames"):
        resolve_ladi_manifest(bundle / "ladi60_semantic_scheduler_v1.csv",
                              tmp_path / "extracted")
