from dataclasses import replace
import hashlib
import json
import pytest
from parameters.cloud_study import macro_score,select_campaign,project_episode
from parameters.cloud_data import load_portable_records
from parameters.contracts import MethodSpecification
from .test_study import episode


def test_fresh_final_requires_explicit_exposure_ledger(tmp_path):
    with pytest.raises(ValueError,match='exposure ledger'):
        select_campaign([],{},tmp_path/'must_not_be_created')
    assert not (tmp_path/'must_not_be_created').exists()


def test_pooled_comparison_requires_known_tasks_and_complete_policy_support():
    building=episode()
    assert macro_score([building],'fifo')[0] is None
    road=replace(building,mission='road_damage')
    assert macro_score([building,road],'fifo')[0] is not None
    projected=project_episode(road,MethodSpecification(enabled_groups=('novelty',)))
    assert macro_score([building,projected],'relevance')[0] is None
    assert macro_score([building,projected],'fifo')[0] is not None


def test_portable_images_are_content_verified(tmp_path):
    data=tmp_path/'data';data.mkdir()
    image=tmp_path/'image.bin';image.write_bytes(b'original image bytes')
    manifest=data/'records.json'
    manifest.write_text(json.dumps([{'image_path':'image.bin',
                                    'image_id':hashlib.sha256(image.read_bytes()).hexdigest()}]))
    assert load_portable_records(manifest)[0]['image_path']==str(image)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_portable_records(manifest)
