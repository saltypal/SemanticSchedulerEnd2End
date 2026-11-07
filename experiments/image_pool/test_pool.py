"""Contract tests use fixtures only, never mislabeled as detector/SPI validation."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from experiments.image_pool.sources import Budget, group_pool, managed_rows
from experiments.image_pool.student import freeze_student
from experiments.image_pool.runtime import cache_student, join_runtime
from experiments.scene_array_cache.cache import cache_all, file_hash
from parameters.scene_arrays import REGISTRY_SHA256


def row(i,sha,parent,role='development'):
    return {'image_id':i,'image_sha256':sha,'parent_id':parent,'role':role,'parameter_training_exposed':False}


def test_group_clip_exact_copies_and_transfer_isolation():
    rows=[row('a','same','clip1'),row('b','other','clip1'),row('c','same','clip2','scheduler_unseen_task')]
    result=group_pool(rows)
    assert len({r['group'] for r in result})==1
    assert {r['split'] for r in result}=={'transfer'}


def test_network_acquisition_is_blocked_locally(tmp_path):
    with pytest.raises(RuntimeError,match='cloud-only'):
        Budget(tmp_path,100,False).fetch('https://example.com/image',tmp_path/'image',50)


def test_temporal_frames_remain_grouped_and_ordered(tmp_path):
    records=[]
    for clip in ('one','two'):
        for frame in range(20):
            image=tmp_path/(clip+str(frame)+'.png');Image.new('RGB',(8,8)).save(image)
            records.append({'source_id':image.name,'image_path':image.name,'source_revision':'fixture',
                'source_split':'train','source_page':'fixture','clip_id':clip,'frame_index':frame})
    manifest=tmp_path/'frames.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    selected=managed_rows(manifest,'bdd100k',{'role':'temporal','cap':6,'clips':2,'frames_per_clip':3},29)
    assert len(selected)==6
    for clip in ('one','two'):
        assert [r['frame_index'] for r in selected if r['clip_id']==clip]==[0,9,19]
    records[0].pop('clip_id');manifest.write_text(json.dumps(records[0])+'\n')
    with pytest.raises(ValueError,match='clip/frame'):managed_rows(manifest,'bdd100k',{'role':'temporal'},29)


def test_validation_selection_does_not_choose_by_test(tmp_path):
    comparison=tmp_path/'comparison';comparison.mkdir()
    (comparison/'study_status.json').write_text(json.dumps([{'status':'complete','balance':b} for b in
        ('original','class_weighted','random_oversampling','moderate_undersampling')]))
    pairs=[]
    combos=[(e,h) for e in ('clip_b32','siglip2_b16','dino','convnextv2_tiny') for h in
       ('logistic','ridge','linear_svm','rbf_svm','knn','random_forest','extra_trees','hist_boost','mlp',
        'mean_mlp','gated_abmil','single_query_cross_attention')]
    for i,(encoder,head) in enumerate(combos):
        pairs.append({'encoder':encoder,'head':head,'balance':'original',
            'kind':'token' if head in ('mean_mlp','gated_abmil','single_query_cross_attention') else 'vector',
            'development':{'macro_f1':i/100},'test':{'macro_f1':1-i/100}})
    (comparison/'frozen_pair_selections.json').write_text(json.dumps(pairs))
    folder=comparison/'original/convnextv2_tiny';folder.mkdir(parents=True)
    (folder/'single_query_cross_attention.pt').write_bytes(b'fixture-not-a-model')
    identity=comparison/'features/convnextv2_tiny';identity.mkdir(parents=True)
    (identity/'identity.json').write_text('{}')
    receipt=tmp_path/'training.json';receipt.write_text(json.dumps({'cohort_size':4000,'status':'complete'}))
    cohort=[{'image_id':str(i)} for i in range(4000)]
    labels=[{'item_id':str(i),'image_sha256':str(i),'available':True,'level':'Low'} for i in range(4000)]
    fitting=[{'image_id':str(i),'image_sha256':str(i)} for i in range(4000)]
    (tmp_path/'selected_manifest.json').write_text(json.dumps(cohort))
    (tmp_path/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
    (comparison/'records.json').write_text(json.dumps(fitting))
    result=freeze_student(comparison,tmp_path/'bundle',receipt)
    assert json.loads(result.read_text())['head']=='single_query_cross_attention'
    (comparison/'records.json').write_text(json.dumps(fitting[:1000]))
    with pytest.raises(ValueError,match='fitting rows'):freeze_student(comparison,tmp_path/'stale',receipt)
    receipt.write_text(json.dumps({'cohort_size':1000,'status':'complete'}))
    with pytest.raises(ValueError,match='no silent'):freeze_student(comparison,tmp_path/'other',receipt)


class FixtureScene:
    signature={'fixture':True,'registry':REGISTRY_SHA256}
    producer_sha256='1'*64
    def sync(self):pass
    def predict(self,image):return []


class FixtureStudent:
    identity_hash='2'*64
    def predict(self,path):return {'level':'Low','available':True,'reference_type':'fixture_not_a_model'}


def test_join_identity_missing_context_and_truth_separation(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(32,32)).save(image)
    rows=[{'image_id':'fixture','image_path':str(image),'image_sha256':file_hash(image),'split':'train','group':'fixture'}]
    cache_all(rows,FixtureScene(),tmp_path/'scene')
    spi=cache_student(rows,FixtureStudent(),tmp_path/'spi')
    result=join_runtime(rows,tmp_path/'scene/scene_records.jsonl',spi,tmp_path/'joined')
    assert result['all_complete']
    record=json.loads((tmp_path/'joined/image_parameters.jsonl').read_text())
    assert record['novelty_available'] is False and record['mission_relevance_available'] is False
    assert 'native_annotations' not in record
    with pytest.raises(ValueError,match='evaluator'):
        join_runtime([{**rows[0],'native_annotations':{'truth':1}}],tmp_path/'scene/scene_records.jsonl',spi,tmp_path/'bad')
