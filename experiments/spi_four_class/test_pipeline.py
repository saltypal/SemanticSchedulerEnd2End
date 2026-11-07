"""Engineering/reference fixtures only; no VLM accuracy is simulated as evidence."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from parameters.spi_protocol import CONTROLS, parse_evidence, request, control_metrics
from parameters.spi_model_selection import stopping_split
from parameters.spi_pipeline import execute
from parameters.spi import SPIClassifier
from parameters.contracts import digest
from parameters.spi_labels import LEVELS
from parameters.spi_release import candidate_release


@pytest.mark.parametrize('claims,expected', [((False,False,False),'Low'),
    ((True,False,False),'Medium'),((True,True,False),'High'),((True,True,True),'Critical')])
def test_hand_mapped_four_levels(claims, expected):
    row = dict(zip(('action_needed','substantial_consequence','immediate_threat'), claims))
    parsed,_ = parse_evidence(json.dumps({**row,'evidence':'fixture visible condition'}))
    assert parsed['level'] == expected and parsed['available']


def test_unknown_and_contradiction_are_not_low():
    row={'action_needed':True,'substantial_consequence':None,'immediate_threat':False,'evidence':'unclear extent'}
    assert parse_evidence(json.dumps(row))[0]['level'] is None
    row.update(action_needed=False,substantial_consequence=True)
    with pytest.raises(ValueError,match='contradictory'):
        parse_evidence(json.dumps(row))
    row.update(action_needed=1,substantial_consequence=False)
    with pytest.raises(ValueError,match='boolean'):
        parse_evidence(json.dumps(row))


def test_exact_markdown_wrapper_is_recorded_without_semantic_repair():
    raw=json.dumps({'action_needed':True,'substantial_consequence':False,
                    'immediate_threat':False,'evidence':'minor defect'})
    plain,flag=parse_evidence(raw)
    fenced,recovered=parse_evidence('```json\n'+raw+'\n```')
    assert fenced==plain and not flag and recovered and fenced['level']=='Medium'
    with pytest.raises(json.JSONDecodeError):
        parse_evidence('Answer: '+raw)
    with pytest.raises(ValueError,match='contradictory'):
        parse_evidence('```json\n'+raw.replace('"substantial_consequence": false','"substantial_consequence": true').replace('"action_needed": true','"action_needed": false')+'\n```')


def test_invalid_controls_stay_in_denominator():
    records=[{'id':c['id'],'level':c['expected'],'valid':True} for c in CONTROLS]
    assert control_metrics(records)['passed']
    for row in records[12:24]:row.update(level=None,valid=False)
    metric=control_metrics(records)
    assert metric['per_class']['Medium']['recall']==0 and not metric['passed']
    assert metric['valid_fraction']==.75


def test_early_stopping_stays_inside_supplied_fit_groups():
    y=np.tile(np.arange(4),12)
    groups=np.array([f'parent{i}' for i in range(len(y))])
    fit,stop=stopping_split(y,groups,17)
    assert not set(groups[fit])&set(groups[stop])
    assert set(y[fit])==set(range(4)) and set(y[stop])==set(range(4))
    assert stopping_split(np.arange(4),np.array(['a','b','c','d']),17) is None


class ReferenceReviewer:
    """Uses known fixture answers; explicitly not a real teacher/model."""
    provenance={'model_id':'REFERENCE_FIXTURE_NO_VLM','revision':'fixture'}
    def __init__(self, fail=False):self.fail=fail;self.closed=False;self.requests=[]
    def review_structured(self,paths,payload,prompt,parser,case_id,raw_dir,**kwargs):
        self.requests.append((paths,payload,prompt))
        if not paths:
            level=next(c['expected'] for c in CONTROLS if c['description']==payload['hypothetical_scene'])
        else:level=('Low','Medium','High','Critical')[int(Path(paths[0]).stem)%4]
        if self.fail:level='Low'
        if 'action_needed' in prompt:
            index=('Low','Medium','High','Critical').index(level)
            raw=json.dumps({'action_needed':index>=1,'substantial_consequence':index>=2,
                            'immediate_threat':index>=3,'evidence':'fixture only'})
        else:raw=json.dumps({'level':level,'evidence':'fixture only','reason':None})
        parsed,_=parser(raw)
        return {'status':'complete','parsed':parsed,'image_inputs':[{'sha256':'fixture'}] if paths else []}
    def close(self):self.closed=True


def cohort(tmp_path):
    folder=tmp_path/'input';images=folder/'images';images.mkdir(parents=True)
    rows=[]
    for i in range(12):
        path=images/f'{i}.bin';path.write_bytes(bytes([i]))
        rows.append({'image_id':str(i),'image_path':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'group':str(i),'split':'train' if i<10 else 'test'})
    return folder,rows


def test_failed_teacher_stops_labels_but_preserves_recovery(tmp_path):
    folder,rows=cohort(tmp_path)
    reviewer=ReferenceReviewer(fail=True)
    with patch('parameters.spi_pipeline.verify_inputs',return_value=(rows,{'layout':'expanded'})):
        output=execute(folder,tmp_path/'run',seconds=30,teacher_loader=lambda spec:reviewer,
                       teacher_specs=({'name':'fixture'},),allow_student=False,visual_panel_size=10)
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='teacher_gate_failed' and status['labelled_images']==0
    assert reviewer.closed and (output/'recovery.zip').is_file()
    assert not (output/'student').exists()


def test_test_images_not_sent_to_teacher_and_support_failure_explicit(tmp_path):
    folder,rows=cohort(tmp_path);reviewer=ReferenceReviewer()
    with patch('parameters.spi_pipeline.verify_inputs',return_value=(rows,{'layout':'expanded'})):
        output=execute(folder,tmp_path/'run',seconds=30,teacher_loader=lambda spec:reviewer,
                       teacher_specs=({'name':'fixture'},),allow_student=False,visual_panel_size=10)
    labels=json.loads((output/'data/spi_labels.json').read_text())
    assert {r['image_id'] for r in labels}=={str(i) for i in range(10)}
    assert {r['level'] for r in labels}=={'Low','Medium','High','Critical'}
    assert json.loads((output/'status.json').read_text())['status']=='four_class_support_failed'
    assert all(not paths or int(Path(paths[0]).stem)<10 for paths,_,_ in reviewer.requests)


def test_teacher_input_rejects_evaluator_truth(tmp_path):
    with pytest.raises(ValueError,match='undeclared'):
        request(ReferenceReviewer(),['image.png'],{'instruction':'x','expected':'Critical'},'id',tmp_path)


def linear_artifact():
    model={'schema':'better_spi_linear_v1','levels':list(LEVELS),'input_dimension':4,
        'feature_signature':'fixture','rubric_sha256':'fixture','w2':(np.eye(4)*8).tolist(),'b2':[0.]*4}
    model['sha256']=digest(model)
    return model


def test_student_order_maps_to_wire_by_class_name():
    classifier=SPIClassifier(linear_artifact())
    for index,name in enumerate(LEVELS):
        result=classifier.predict(np.eye(4)[index],'fixture')
        assert result['level']==name and result['scheduling_class']==3-index
        assert set(result['scores'])==set(LEVELS)


def test_release_guard_preserves_history_and_rejects_weak_model(tmp_path):
    model=tmp_path/'model.json';model.write_text(json.dumps(linear_artifact()))
    current=tmp_path/'current.json';current.write_text(json.dumps({'spi':{'model_path':'historical.json'}}))
    before=current.read_bytes()
    validation=tmp_path/'validation.json'
    per_class={k:{'support':10,'recall':1.} for k in LEVELS}
    per_class['Medium']['recall']=0.
    validation.write_text(json.dumps({'student':{'per_class':per_class},'export_max_absolute_difference':0.}))
    release=candidate_release(model,validation,current,tmp_path/'release.json',root=tmp_path)
    assert not release['runtime_promotion_allowed']
    assert release['spi']['supported_levels']==list(LEVELS)
    assert current.read_bytes()==before
    with pytest.raises(FileExistsError):
        candidate_release(model,validation,current,tmp_path/'release.json',root=tmp_path)


def test_cli_four_class_gate_precedes_visual_loading(tmp_path, monkeypatch):
    import parameters.parameter_cli as cli
    model=linear_artifact()
    model['levels']=['Low','High','Critical']
    model['w2']=[model['w2'][i] for i in (0,2,3)]
    model['b2']=[0.]*3
    model.pop('sha256');model['sha256']=digest(model)
    path=tmp_path/'historical.json';path.write_text(json.dumps(model))
    def forbidden(*args,**kwargs):
        raise AssertionError('visual producer must not load before class gate')
    monkeypatch.setattr(cli,'MinimalVisionProvider',forbidden)
    monkeypatch.setattr(sys,'argv',['parameters','--images','unreadable.png','--mission','inspect',
        '--model',str(path),'--clip-checkpoint','not_loaded','--output',str(tmp_path/'out.json'),
        '--require-four-classes'])
    with pytest.raises(ValueError,match='four actual SPI classes'):
        cli.main()
