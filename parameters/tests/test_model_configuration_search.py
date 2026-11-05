from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from parameters.model_settings import DetectorSettings,ReviewerSettings
from parameters.pilot_parameters import detections,PilotParameterExtractor,PilotMemory
from parameters.extractor import VisualEvidence,SenderMemory,SemanticParameterExtractor
from parameters.ue_profiles import load_ue_profiles
from parameters.approach_grid import search_plan,evaluate_cache,decide,project_representation,REPRESENTATIONS
from parameters.annotation_review import reviewer_input,make_rubric,review_system_prompt
from parameters.qwen_reviewer import build_review_messages


def test_detector_threshold_is_carried_to_downstream_and_kb_signature():
    mission=load_ue_profiles()[0]['building_damage']
    ev=VisualEvidence(embedding=(1.,0.),width=10,height=10,statuses={'detector':'ok'},
        detections=({'label':'building','score':.1,'box':[0,0,5,5]},),
        provenance={'detector_provenance':{'conf_threshold':.05,'settings_sha256':'lower'}})
    assert len(detections(ev))==1
    memory=PilotMemory('u',mission.mission_id);memory.commit(ev,0)
    ev.provenance={'detector_provenance':{'conf_threshold':.25,'settings_sha256':'higher'}}
    assert detections(ev)==[]
    assert not PilotParameterExtractor().extract_scarcity(ev,mission,memory).available
    with pytest.raises(ValueError):DetectorSettings(confidence=float('nan'))
    with pytest.raises(ValueError):DetectorSettings(image_size=641)
    args=DetectorSettings(confidence=.05,image_size=1024).predict_kwargs()
    assert args['conf']==.05 and args['imgsz']==1024 and not args['half']


def test_appearance_kb_does_not_mix_native_crop_and_pad():
    mission=load_ue_profiles()[0]['building_damage'];memory=SenderMemory('u',mission.mission_id)
    e=VisualEvidence(embedding=(1.,0.),provenance={'preprocessing':'checkpoint_native'})
    memory.commit(e,0);e.provenance={'preprocessing':'aspect_preserving_pad'}
    result=SemanticParameterExtractor().extract_novelty(e,mission,memory)
    assert not result.available and 'comparable' in result.reason


def test_reviewer_settings_cannot_claim_high_resolution_after_small_resize():
    settings=ReviewerSettings(max_pixels=768*768,image_max_side=None)
    assert settings.image_max_side is None
    assert replace(settings,max_new_tokens=192).sha256!=settings.sha256
    with pytest.raises(ValueError):ReviewerSettings(max_new_tokens=0)
    with pytest.raises(ValueError):ReviewerSettings(prompt_variant='unknown')
    for variant in ('candidate_provenance_v2','candidate_provenance_concise_v3'):
        message=build_review_messages([{'type':'text','text':'data'}],variant)
        assert message[0]['content'][0]['text']==review_system_prompt(variant)


def test_reviewer_uses_effective_size_and_rejects_pixel_budget_overrun(monkeypatch):
    import torch
    from transformers import AutoProcessor
    from parameters.qwen_reviewer import load_review_processor,pixel_input_receipt
    captured={}
    def load(path,**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(image_processor=SimpleNamespace(size=kwargs['size'],patch_size=16))
    monkeypatch.setattr(AutoProcessor,'from_pretrained',load)
    settings=ReviewerSettings(max_pixels=512*512,image_max_side=None)
    processor=load_review_processor('cached-fixture',settings)
    assert captured['size']=={'shortest_edge':65536,'longest_edge':262144}
    assert 'max_pixels' not in captured
    inputs={'pixel_values':torch.ones((10,12)),'image_grid_thw':torch.tensor([[1,32,32]])}
    receipt=pixel_input_receipt(inputs,processor,262144)
    assert receipt['processed_image_pixels']==[262144]
    inputs['image_grid_thw']=torch.tensor([[1,64,64]])
    with pytest.raises(RuntimeError,match='exceeded'):pixel_input_receipt(inputs,processor,262144)


def test_review_scope_is_explicit_without_deleting_inconvenient_claims():
    mission=load_ue_profiles()[0]['building_damage']
    reference={'facts':{'building.present':True,'building.affected_or_greater':False}}
    rubric=make_rubric(mission,reference)
    candidate={'claims':{'building.present':True,'road.damaged':False}}
    p=reviewer_input(candidate,reference,mission,rubric,[],claim_scope='mission')
    assert 'road.damaged' in p['candidate_parameter_representation']['claims']
    assert p['assessment_scope']['assessed_claim_names']==rubric['required_fact_names']
    assert 'expected' not in p


def test_search_plan_is_staged_and_never_selects_from_exposed_test(tmp_path):
    plan=search_plan();assert plan['stages']['native_score_replay']['count']==12
    assert plan['stages']['representations']['native_cross_product_rows']==48
    assert plan['stages']['reviewer_controls']['count']==8
    assert decide(.5,.5,0.) is False and decide(.5,.5,.1) is None
    assert decide(.4,.5,.1) is False and decide(.6,.5,.1) is True # Open abstention band, explicit endpoint tolerance.
    facts={'building.present':'buildings_any','road.present':'roads_any',
        'building.affected_or_greater':'buildings_affected_or_greater','road.damaged':'roads_damage'}
    records=[{'image_sha256':str(i),'native_labels':{native:bool(i) for native in facts.values()}} for i in range(2)]
    cache={str(i):{fact:{'score':.9 if i else .1} for fact in facts} for i in range(2)}
    rows=evaluate_cache({'checkpoint_native':cache},records,tmp_path/'out')
    assert len(rows)==6
    assert json.loads((tmp_path/'out/status.json').read_text())['selected_configuration'] is None
    with pytest.raises(ValueError,match='diagnostic only'):
        evaluate_cache({'checkpoint_native':cache},records,tmp_path/'illegal',role='final')


def test_parameter_combinations_preserve_missingness_and_existing_claim_values():
    candidate={'claims':{'building.present':True,'road.present':False},'parameters':{
        'relevance':{'value':.8},'novelty':{'value':.1},'scene':{'value':{}}},
        'claim_availability':{'building.present':{'available':True},
            'building.affected_or_greater':{'available':False}},'temporal':{'available':False}}
    for profile in REPRESENTATIONS:
        projected=project_representation(candidate,'building_damage',profile)
        assert projected['claims']['building.present'] is True
        assert 'building.affected_or_greater' not in projected['claims']
    assert 'road.present' in candidate['claims'] and 'scene' in candidate['parameters']


def test_reviewer_grid_reuses_model_and_scores_all_control_failures(tmp_path):
    from parameters.annotation_review import control_cases
    from parameters.reviewer_grid import run
    missions,_=load_ue_profiles();image=tmp_path/'fixture.png';image.write_bytes(b'not-loaded-by-mocked-reviewer')
    records=[{'source':'ladi','image_id':str(i),'image_path':str(image),'group':'g'+str(i),
        'native_labels':{'buildings_any':True,'buildings_affected_or_greater':bool(i%2)}} for i in range(8)]
    cases=control_cases(records,missions,parents=8)
    class BrokenReviewer:
        provenance={'fixture_only':True}
        def configure(self,settings):self.settings=settings
        def review(self,*args):return {'case_id':args[-2],'status':'model_failed'}
    with pytest.raises(ValueError,match='exclusion evidence'):
        run(BrokenReviewer(),cases,missions,tmp_path/'no-exclusions')
    results=run(BrokenReviewer(),cases,missions,tmp_path/'review-grid',excluded_groups=['historical-group'])
    assert len(results)==8 and all(not r['metrics']['engineering_gate_passed'] for r in results)
    assert all(r['metrics']['valid_fraction']==0 for r in results)
    assert len(list((tmp_path/'review-grid').glob('*_model.json')))==8
