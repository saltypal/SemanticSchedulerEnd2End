"""Synthetic notebook9 integration fixtures; no real teacher/model accuracy."""
import hashlib
from pathlib import Path
import json
import numpy as np
import pytest
from unittest.mock import patch
from PIL import Image
from parameters.artifacts import write_json
from parameters.contracts import MissionSpec,digest
from parameters.dataset_pipeline import validate_records,label_dataset,cache_parameters,export_spi_training
from parameters.extractor import VisualEvidence,embedding_signature
from parameters.scene_measurements import measure_detections
from parameters.spi_direct_priority import PROMPT


class DummyProvider:
    device='cpu'
    def extract_batch(self,images,missions):
        return [VisualEvidence(embedding=np.eye(512)[int(np.asarray(image)[0,0,0])%4],
            provenance={'backbone':'synthetic_fixture'},width=image.width,height=image.height) for image in images]
    def context_embedding(self,mission):return np.eye(512)[0]


class DummyScene:
    def measure(self,image):
        return measure_detections([{'label':'object','score':.9,'box':[0,0,8,8]}],16,16)


def make_dummy_inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);records=[]
    for i in range(8):
        path=root/f'image_{i}.png';Image.new('RGB',(16,16),(i,20,30)).save(path)
        records.append({'image_id':str(i),'image_path':str(path.resolve()),
            'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'group':str(i),
            'split':'train' if i<4 else 'validation','source':'synthetic_fixture'})
    signature=embedding_signature(VisualEvidence(provenance={'backbone':'synthetic_fixture'}))
    w=np.zeros((4,512));w[:,:4]=np.eye(4)*8
    artifact={'schema':'better_spi_linear_v1','levels':['Low','Medium','High','Critical'],
        'rubric_sha256':'synthetic_fixture_only',
        'input_dimension':512,'feature_signature':signature,'w2':w.tolist(),'b2':[0.]*4}
    artifact['sha256']=digest(artifact);write_json(root/'model.json',artifact)
    # Synthetic receipt is for software tests only, never a real readiness claim.
    receipt={'fixture_only':True,'runtime_promotion_allowed':True,
        'release_checks':{k:True for k in ('export_parity','all_class_support','per_class_recall')},
        'spi':{'file_sha256':hashlib.sha256((root/'model.json').read_bytes()).hexdigest(),'model_sha256':artifact['sha256']},
        'diagnostic_review':{'accepted':True,'unresolved_decisive_contradictions':0},
        'class_support':{s:{c:{'rows':n,'groups':g} for c in artifact['levels']}
                         for s,n,g in [('train',12,6),('validation',5,3)]}}
    write_json(root/'readiness.json',receipt)
    return records


def run_dummy(root):
    root=Path(root);records=make_dummy_inputs(root/'inputs')
    missions=[(ue,MissionSpec(ue,'Inspect visible objects.',(('object','object'),))) for ue in ('ue1','ue2')]
    output=cache_parameters(records,missions,DummyProvider(),DummyScene(),
        root/'inputs/model.json',root/'inputs/readiness.json',root/'cache',synthetic=True)
    return output


def test_dummy_pipeline_shared_production_path(tmp_path):
    output=run_dummy(tmp_path)
    assert json.loads((output/'status.json').read_text())['completed_images']==8
    streams=json.loads((output/'streams.json').read_text())
    assert len(streams)==16 and all(r['header_bytes']==35 for r in streams)
    assert all(r['parameters']['capture_time_ms'] is None for r in streams)
    first=[r for r in streams if r['sequence']==0]
    assert len(first)==4 and all(r['parameters']['novelty_cold_start'] for r in first)
    f=np.load(output/'features.npz',allow_pickle=False)
    assert f['embeddings'].shape==(8,512)
    for name,h in json.loads((output/'checksums.json').read_text()).items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==h


def test_hash_and_split_leakage_rejected(tmp_path):
    records=make_dummy_inputs(tmp_path)
    records[4]['group']=records[0]['group']
    with pytest.raises(ValueError,match='crosses splits'):validate_records(records)
    records[4]['group']='4';records[0]['image_sha256']='bad'
    with pytest.raises(ValueError,match='hash mismatch'):validate_records(records)


def test_not_ready_rejects_before_creating_cache(tmp_path):
    records=make_dummy_inputs(tmp_path/'inputs')
    receipt=json.loads((tmp_path/'inputs/readiness.json').read_text());receipt['runtime_promotion_allowed']=False
    write_json(tmp_path/'inputs/readiness.json',receipt)
    with pytest.raises(ValueError,match='readiness'):
        cache_parameters(records,[],None,None,tmp_path/'inputs/model.json',tmp_path/'inputs/readiness.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_labels_export_whole_image_schema_no_native_truth_input(tmp_path):
    records=make_dummy_inputs(tmp_path/'inputs')
    records[0]['native_labels']={'urgency':'Critical'}
    class Teacher:
        provenance={'model_id':'fixture','revision':'fixture'}
        def load(self):pass
        def review_structured(self,paths,payload,prompt,parser,*args,**kwargs):
            assert set(payload)=={'instruction'}
            parsed,_=parser('LOW');return {'status':'complete','parsed':parsed}
    receipt={'accepted':True,'model_id':'fixture','revision':'fixture',
             'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest()}
    labels=label_dataset(records,Teacher(),receipt,tmp_path/'labels')
    assert len(labels)==8 and all(r['label_unit']=='whole_image' and r['rubric_sha256'] for r in labels)
    assert all('native_labels' not in r for r in labels)


def test_training_bundle_compatible_with_actual_spi_loader(tmp_path):
    output=run_dummy(tmp_path)
    records=json.loads((output/'manifest.json').read_text())['records']
    labels=[{**r,'level':['Low','Medium','High','Critical'][i%4],
             'available':True,'reason':None,'label_unit':'whole_image',
             'label_source':'vlm_pseudo','rubric_sha256':'synthetic_fixture_only'} for i,r in enumerate(records)]
    write_json(tmp_path/'labels.json',labels)
    bundle=export_spi_training(output/'features.npz',tmp_path/'labels.json',tmp_path/'training')
    manifest=json.loads((bundle/'training_manifest.json').read_text())
    assert manifest['roles']['train']==4 and manifest['roles']['validation']==4
    assert manifest['training_performed'] is False


def test_fixture_receipt_cannot_enable_real_cache(tmp_path):
    records=make_dummy_inputs(tmp_path/'inputs')
    with pytest.raises(ValueError,match='fixture'):
        cache_parameters(records,[],None,None,tmp_path/'inputs/model.json',tmp_path/'inputs/readiness.json',tmp_path/'real')
    assert not (tmp_path/'real').exists()


def test_bad_context_does_not_destroy_other_parameters(tmp_path):
    records=make_dummy_inputs(tmp_path/'inputs')
    class BadContext(DummyProvider):
        def context_embedding(self,mission):raise ValueError('unsupported mission paragraph')
    mission=MissionSpec('mission','Inspect visible objects.',(('object','object'),))
    output=cache_parameters(records,[('ue',mission)],BadContext(),DummyScene(),
        tmp_path/'inputs/model.json',tmp_path/'inputs/readiness.json',tmp_path/'cache',synthetic=True)
    streams=json.loads((output/'streams.json').read_text())
    assert all(r['parameters']['relevance'] is None and r['parameters']['spi_available'] for r in streams)


def test_selected_revision_pinned_before_loading():
    from parameters.dataset_pipeline import load_selected_teacher
    receipt={'accepted':True,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
             'model_id':'google/gemma-4-E4B-it','revision':'a'*40}
    with patch('parameters.gemma_spi_reviewer.load_gemma',return_value='fixture') as loader:
        assert load_selected_teacher(receipt)=='fixture'
        loader.assert_called_once_with({'revision':'a'*40})


def test_shard_failure_does_not_count_current_image_completed(tmp_path):
    import parameters.dataset_pipeline as pipeline
    records=make_dummy_inputs(tmp_path/'inputs')
    original=pipeline.write_json
    def fail_shard(path,data):
        if 'shards' in Path(path).parts:raise OSError('fixture interrupted write')
        original(path,data)
    mission=MissionSpec('mission','Inspect visible objects.',(('object','object'),))
    with patch.object(pipeline,'write_json',side_effect=fail_shard),pytest.raises(OSError):
        pipeline.cache_parameters(records,[('ue',mission)],DummyProvider(),DummyScene(),
            tmp_path/'inputs/model.json',tmp_path/'inputs/readiness.json',tmp_path/'cache',synthetic=True)
    status=json.loads((tmp_path/'cache/status.json').read_text())
    assert status['status']=='incomplete' and status['completed_images']==0
    assert status['extracted_images']==1
