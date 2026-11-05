from copy import deepcopy
import json
import math
import pytest
from parameters.annotation_review import (native_reference,make_rubric,control_cases,
    parse_review,reviewer_input,score_controls,direct_fact_metrics)
from parameters.ue_profiles import load_ue_profiles
from parameters.pilot_parameters import (PilotMemory,PilotParameterExtractor,
    canonical_label,union_area,matches_target)
from parameters.extractor import VisualEvidence


def record(index=0,damaged=True):
    return {'image_id':str(index),'image_path':'fixture.png','source':'ladi','group':'g'+str(index),
            'native_labels':{'buildings_any':True,'buildings_affected_or_greater':damaged,'roads_damage':True}}


def evidence(label='tv'):
    return VisualEvidence(embedding=(1.,0.),similarities={'building_damage':.5},
      detections=({'label':label,'box':[0,0,50,50],'score':.9},),width=100,height=100,
      statuses={'detector':'ok'},provenance={})


def test_reference_unknowns_and_damage_not_blockage():
    ref=native_reference(record())
    assert ref['facts']['building.affected_or_greater'] is True and 'road.blocked' not in ref['facts']
    assert 'household.target_present' not in ref['facts']
    ref=native_reference({'source':'medic','native_labels':{'informative':0},'image_id':'m','group':'g'})
    assert ref['facts']=={'incident.informative':True}
    assert native_reference({'source':'medic','native_labels':{},'image_id':'m','group':'g'})['facts']=={}


def test_control_truth_and_answer_key_isolation():
    missions,_=load_ue_profiles();cases=control_cases([record(0,False),record(1,True)],missions,parents=2)
    assert len(cases)==6
    assert {c['control_arm'] for c in cases}=={'complete','incomplete','contradicted'}
    payload=reviewer_input(cases[0]['candidate'],cases[0]['reference'],missions['building_damage'],cases[0]['rubric'],[])
    assert 'expected' not in payload and 'control_arm' not in payload
    assert 'published_annotation_reference' in payload # Intentional annotation-assisted arm.


def test_review_parser_rejects_string_bools_and_inconsistent_verdicts():
    valid={'parameters_consistent':True,'representation_sufficient':False,'reason':'Missing condition',
           'missing_or_conflicting_fields':['building.damaged']}
    assert parse_review(json.dumps(valid))[0]==valid
    assert parse_review('```json\n'+json.dumps(valid)+'\n```')[1] is True
    wrong=dict(valid,parameters_consistent='yes')
    with pytest.raises(ValueError):parse_review(json.dumps(wrong))
    with pytest.raises(ValueError):parse_review(json.dumps(dict(valid,parameters_consistent=False,representation_sufficient=True)))
    with pytest.raises(ValueError):parse_review(json.dumps(dict(valid,representation_sufficient=True)))
    with pytest.raises(ValueError):parse_review('Here is the answer: '+json.dumps(valid))
    with pytest.raises(ValueError):parse_review(json.dumps(valid).replace('"parameters_consistent": true','"parameters_consistent": false, "parameters_consistent": true'))


def test_v2_controls_use_new_groups_without_repairing_omissions():
    missions,_=load_ue_profiles()
    rows=[record(0,False),record(1,True),record(2,False),record(3,True)]
    old=control_cases(rows,missions,parents=2)
    excluded={c['reference']['group'] for c in old}
    fresh=control_cases(rows,missions,parents=2,excluded_groups=excluded)
    assert not {c['reference']['group'] for c in fresh}&excluded
    for case in fresh:
        if case['control_arm']!='incomplete':continue
        before=deepcopy(case['candidate'])
        payload=reviewer_input(case['candidate'],case['reference'],missions['building_damage'],case['rubric'],[])
        assert 'building.affected_or_greater' not in payload['candidate_parameter_representation']['claims']
        assert case['candidate']==before and 'expected' not in payload
        assert payload['review_protocol']=='candidate_provenance_v2'
    with pytest.raises(ValueError,match='insufficient'):
        control_cases(rows,missions,parents=2,excluded_groups={r['group'] for r in rows})


def test_degenerate_reviewer_fails_both_control_endpoints():
    missions,_=load_ue_profiles();cases=control_cases([record(0,False),record(1,True)],missions,parents=2)
    for boolean in (False,True):
        output=[{'case_id':c['case_id'],'status':'complete','parsed':{
            'parameters_consistent':boolean,'representation_sufficient':boolean}} for c in cases]
        assert score_controls(cases,output)['engineering_gate_passed'] is False
    perfect=[{'case_id':c['case_id'],'status':'complete','parsed':dict(c['expected'])} for c in cases]
    assert score_controls(cases,perfect)['engineering_gate_passed'] is True
    assert score_controls(cases,[])['valid_fraction']==0


def test_kb_observation_sidecar_and_concept_rarity_are_causal():
    mission=load_ue_profiles()[0]['building_damage'];memory=PilotMemory('u','building_damage')
    extractor=PilotParameterExtractor();ev=evidence('building')
    first=extractor.extract_evidence(ev,mission,{'image_token':'a','source_id':'u'},memory)
    memory.commit(ev,0,{'observation_id':'a','image_id':'same','image_path':'a.png','mission_version':mission.version})
    second=extractor.extract_evidence(ev,mission,{'image_token':'b','source_id':'u'},memory)
    assert first.parameters['novelty'].value['cold_start']==1
    assert second.parameters['novelty'].value['distance']==0
    assert second.parameters['novelty'].value['new_concept_fraction']==0
    assert second.parameters['rarity'].supporting['probabilities']['building']==pytest.approx(2/3)
    assert second.parameters['rarity'].value<first.parameters['rarity'].value
    assert memory.history()[0]['observation']['image_id']=='same'
    with pytest.raises(ValueError):memory.commit(ev,1,{'reference_labels':{'damage':True}})
    assert memory.version==1
    ev.provenance={'bad':float('nan')}
    with pytest.raises(ValueError):memory.commit(ev,1,{'observation_id':'b'})
    assert memory.version==1 and len(memory.entries)==1


def test_aliases_vehicle_groups_and_union_area_have_numerical_reference():
    assert canonical_label('TV')=='television' and matches_target('truck',['vehicle'])
    assert union_area([[0,0,2,2],[1,0,3,2]])==6
    mission=load_ue_profiles()[0]['household_inventory'];extractor=PilotParameterExtractor()
    row=extractor.extract_evidence(evidence(),mission,{'image_token':'a','source_id':'u'},PilotMemory('u',mission.mission_id))
    assert row.parameters['spatial'].value['largest_area']==.25
    assert row.parameters['spatial'].value['union_area']==.25
    assert row.parameters['spatial'].supporting['raw_target_count']==1


def test_missing_detection_not_confused_with_available_zero():
    mission=load_ue_profiles()[0]['building_damage'];extractor=PilotParameterExtractor();ev=evidence()
    ev.detections=()
    zero=extractor.extract_evidence(ev,mission,{'image_token':'a','source_id':'u'},PilotMemory('u',mission.mission_id))
    assert zero.parameters['spatial'].available and zero.parameters['spatial'].value['largest_area']==0
    ev.statuses['detector']='failed'
    missing=extractor.extract_evidence(ev,mission,{'image_token':'a','source_id':'u'},PilotMemory('u',mission.mission_id))
    assert not missing.parameters['spatial'].available and not missing.parameters['rarity'].available


def test_reference_poisoning_cannot_change_runtime_extraction():
    mission=load_ue_profiles()[0]['building_damage'];extractor=PilotParameterExtractor();ev=evidence('building')
    ref=record();a=extractor.extract_evidence(ev,mission,{'image_token':'a','source_id':'u'},PilotMemory('u',mission.mission_id))
    ref['native_labels']['buildings_affected_or_greater']=False
    b=extractor.extract_evidence(ev,mission,{'image_token':'a','source_id':'u'},PilotMemory('u',mission.mission_id))
    assert a==b
    assert direct_fact_metrics({'building.affected_or_greater':True},native_reference(ref))['correct']==0


def test_control_output_identity_and_invalid_fact_values_do_not_disappear():
    missions,_=load_ue_profiles();cases=control_cases([record(0,False),record(1,True)],missions,parents=2)
    row={'case_id':cases[0]['case_id'],'status':'model_failed'}
    with pytest.raises(ValueError):score_controls(cases,[row,row])
    with pytest.raises(ValueError):score_controls(cases,[{'case_id':'foreign'}])
    metrics=direct_fact_metrics({'building.present':None},native_reference(record()))
    assert metrics['known_comparable_facts']==0 and metrics['invalid_reference_fact_values']==['building.present']


@pytest.mark.parametrize('technical_failure',[False,True])
def test_orchestrator_never_loads_producer_when_reviewer_fails(tmp_path,monkeypatch,technical_failure):
    import hashlib
    import torch
    from PIL import Image
    import parameters.qwen_reviewer as qwen_module
    import parameters.pilot_producers as producer_module
    from parameters.annotation_pilot import run_annotation_pilot
    root=tmp_path/'prepared';(root/'data').mkdir(parents=True);(root/'images').mkdir()
    rows=[]
    for i in range(24):
        path=root/'images'/(str(i)+'.png');Image.new('RGB',(8,8),(i*8,0,0)).save(path)
        row=record(i,i>=12);row.update(image_id=hashlib.sha256(path.read_bytes()).hexdigest(),image_path='images/'+path.name)
        rows.append(row)
    manifest=root/'data/records.json';manifest.write_text(json.dumps(rows))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda index:'fixture-only-not-a-real-GPU')
    monkeypatch.setattr(producer_module,'snapshot_model',lambda name:('fixture-model','fixture-revision'))
    class Reviewer:
        provenance={'fixture_only':True}
        def __init__(self,*args,**kwargs):pass
        def load(self):pass
        def close(self):pass
        def review(self,image_paths,*args):
            return {'case_id':args[-2],'status':'model_failed' if technical_failure else 'complete',
                'parsed':{'parameters_consistent':False,'representation_sufficient':False},
                'wall_seconds':0.,'image_inputs':[{'sha256':hashlib.sha256(__import__('pathlib').Path(image_paths[0]).read_bytes()).hexdigest()}]}
    monkeypatch.setattr(qwen_module,'QwenAnnotationReviewer',Reviewer)
    def forbidden(*args,**kwargs):raise AssertionError('producer must not load after reviewer failure')
    monkeypatch.setattr(producer_module,'PilotVisionProducer',forbidden)
    output=tmp_path/'run'
    if technical_failure:
        with pytest.raises(RuntimeError,match='technical'):run_annotation_pilot(manifest,output,steps_per_ue=2)
        assert json.loads((output/'status.json').read_text())['status']=='failed'
    else:
        run_annotation_pilot(manifest,output,steps_per_ue=2)
        assert json.loads((output/'status.json').read_text())['status']=='reviewer_gate_failed'
        assert len((output/'reviews/control_reviews.jsonl').read_text().splitlines())==72
    assert not (output/'models/producer.json').exists()
