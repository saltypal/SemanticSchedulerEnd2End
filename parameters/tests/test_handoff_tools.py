import hashlib
import json
from pathlib import Path
import pytest
from parameters.handoff_tools import restore_fixture


def test_portable_fixture_metadata_matches_its_manifest():
    source=Path(__file__).resolve().parents[1]/'handoff/evidence/h3'
    manifest=json.loads((source/'FIXTURE_MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected
    assert len(json.loads((source/'data/records.json').read_text()))==12


def test_restore_requires_an_explicit_image_source_before_creating_output(tmp_path):
    with pytest.raises(ValueError): restore_fixture(tmp_path/'missing')
    with pytest.raises(ValueError): restore_fixture(tmp_path/'both', download_images=True, image_source='anything')
    assert not list(tmp_path.iterdir())


def test_restore_from_existing_images_preserves_exact_pinned_metadata(tmp_path):
    root=Path(__file__).resolve().parents[1]
    original=root/'results/ops/20261025T2256Z/semantic_validation_h3/monitor/retrieved/results/semantic_validation/20261026h3'
    if not (original/'data/images').is_dir(): pytest.skip('original image bytes not restored in this checkout')
    destination=tmp_path/'restored'
    receipt=restore_fixture(destination,image_source=original)
    assert receipt['status']=='complete' and receipt['images']==12 and receipt['image_bytes']==5544858
    assert not receipt['model_inference']
    source=root/'handoff/evidence/h3'
    assert (source/'checksums.json').read_bytes()==(destination/'checksums.json').read_bytes()
    assert (source/'data/records.json').read_bytes()==(destination/'data/records.json').read_bytes()
    with pytest.raises(FileExistsError): restore_fixture(destination,image_source=original)
