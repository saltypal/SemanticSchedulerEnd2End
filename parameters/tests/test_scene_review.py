from copy import deepcopy
import json
from pathlib import Path
import pytest
from parameters.scene_review import (audit_truth, policy_urgency, reviewer_payload,
    parse_result, score_cases, POLICY, PROTOCOL)
from parameters.scene_review_run import prepare


def claim(name,value,status='available'):
    return dict(name=name,value=value,status=status)


def test_explicit_negative_unknown_and_masks_do_not_borrow_reference_values():
    document={'scene':{'claims':[claim('thing.present',False),claim('thing.damaged',True,'unavailable')]}}
    result=audit_truth(document,{'thing.present':False,'thing.damaged':True},['thing.present','thing.damaged'])
    assert result==dict(claims={'thing.present':'match','thing.damaged':'unrepresented'},
        parameters_consistent=True,representation_sufficient=False)
    document['scene']['claims'][0]['value']=True
    assert not audit_truth(document,{'thing.present':False},['thing.present'])['parameters_consistent']


def test_operator_policy_time_boundaries_and_unknown_are_not_low_urgency():
    facts={'building.present':True,'building.affected_or_greater':True}
    assert not policy_urgency(facts,{'timestamp_trusted':False,'evidence_age_ms':0})['determinable']
    assert policy_urgency(facts,{'timestamp_trusted':True,'evidence_age_ms':300000})['response_window_ms']==1800000
    assert not policy_urgency(facts,{'timestamp_trusted':True,'evidence_age_ms':300001})['determinable']
    assert not policy_urgency(facts,{'timestamp_trusted':True,'evidence_age_ms':True})['determinable']
    assert policy_urgency({'building.present':False},{'timestamp_trusted':True,'evidence_age_ms':0})==dict(
        determinable=True,action_required=False,response_window_ms=None)


def prepared(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=root/'results/ops/20261025T2256Z/semantic_validation_h3/monitor/retrieved/results/semantic_validation/20261026h3'
    if not source.is_dir():pytest.skip('retained h3 fixture unavailable')
    spec=json.loads((root/'config/scene_review_v1.json').read_text())
    return prepare(source,tmp_path/'run',spec)


def test_44_cases_keep_actual_predictions_and_route_information_isolation(tmp_path):
    cases=prepared(tmp_path)
    assert len(cases)==44 and len({c['id'] for c in cases})==44
    assert sum(c['route']=='audit' for c in cases)==30
    assert sum(c['route']=='json_only' for c in cases)==8
    assert sum(c['route']=='image_only' for c in cases)==6
    for case in cases:
        payload,_=reviewer_payload(case)
        assert not set(payload)&{'expected','arm','id','image_id','json_encoded_answers'}
        if case['route']!='audit': assert 'known_reference_facts' not in payload
        if case['route']=='image_only': assert 'candidate' not in payload
        elif case['route']=='json_only':
            poisoned=deepcopy(case);poisoned['expected']={'secret':'must not affect payload'}
            assert reviewer_payload(poisoned)==reviewer_payload(case)
    fresh=next(c for c in cases if c['id']=='policy-fresh')
    stale=next(c for c in cases if c['id']=='policy-stale')
    assert fresh['document']==stale['document']
    assert fresh['expected']['urgency']['determinable']
    assert not stale['expected']['urgency']['determinable']
    assert fresh['context']['scope']=='synthetic engineering context'


def test_controls_are_independent_constructions_not_model_accuracy_claims(tmp_path):
    cases=prepared(tmp_path)
    for case in cases:
        if case['arm']=='complete_control':
            assert case['expected']['parameters_consistent'] and case['expected']['representation_sufficient']
        if case['arm'] in ('omitted','masked'):
            assert case['expected']['parameters_consistent'] and not case['expected']['representation_sufficient']
        if case['arm']=='flipped':
            assert not case['expected']['parameters_consistent'] and not case['expected']['representation_sufficient']


def test_parser_keeps_strict_unknown_and_duplicate_key_semantics():
    expected=dict(answers={'thing.present':False},urgency=dict(determinable=False,action_required=None,response_window_ms=None),reason='No trusted time.')
    assert parse_result(json.dumps(expected),'json_only',['thing.present'])[0]==expected
    bad=deepcopy(expected);bad['answers']['thing.present']='false'
    with pytest.raises(ValueError):parse_result(json.dumps(bad),'json_only',['thing.present'])
    bad=deepcopy(expected);bad['urgency']['action_required']=False
    with pytest.raises(ValueError):parse_result(json.dumps(bad),'json_only',['thing.present'])
    with pytest.raises(ValueError):parse_result('{"reason":"a","reason":"b"}','json_only',[])


def test_scores_do_not_mix_routes_or_exclude_failed_responses(tmp_path):
    cases=prepared(tmp_path)
    rows=[dict(case_id=c['id'],status='complete',parsed={**deepcopy(c['expected']),'reason':'fixture'}) for c in cases]
    report=score_cases(cases,rows)
    assert all(g['joint_correct']==g['planned'] for g in report['groups'].values())
    assert report['controls']['engineering_gate_passed']
    assert report['groups']['image_only:actual']['ungraded_reference_answers']==6
    rows[0]['status']='model_failed'
    changed=score_cases(cases,rows)
    assert changed['groups']['audit:actual']['planned']==6
    assert changed['groups']['audit:actual']['valid']==5
    assert not report['real_world_urgency_verified'] and not report['whole_scene_depth_verified']


def test_notebook_uses_new_protocol_not_the_legacy_run_path(tmp_path):
    import nbformat
    from parameters.build_scene_review_notebook import build
    folder=build(tmp_path/'package','fixture-v2')
    nb=nbformat.read(folder/'notebook.ipynb',as_version=4)
    cells=[c.source for c in nb.cells if c.cell_type=='code']
    for source in cells:compile(source,'scene-review-cell','exec')
    assert 'scene_review_run import run' in cells[2] and 'jsonschema' in cells[1]
    manifest=json.loads((folder/'pilot_manifest.json').read_text())
    assert manifest['planned_reviews']==44 and manifest['server_seconds']==1800
    assert manifest['protocol']==PROTOCOL and manifest['enable_thinking'] is False
    assert manifest['internal_seconds']==1650 and manifest['maximum_allocated_gpu_hours']==1
    assert json.loads((folder/'review_manifest.json').read_text())==manifest


def test_policy_following_is_not_penalized_as_reference_correctness_when_json_is_wrong():
    reference={'building.present':True,'building.affected_or_greater':True}
    encoded={'building.present':False,'building.affected_or_greater':False}
    context={'timestamp_trusted':True,'evidence_age_ms':1000}
    case=dict(id='wrong-extraction',route='json_only',arm='diagnostic',requested=list(reference),
        context=context,json_encoded_answers=encoded,
        expected={'answers':reference,'urgency':policy_urgency(reference,context)})
    row=dict(case_id=case['id'],status='complete',parsed=dict(answers=encoded,
        urgency=policy_urgency(encoded,context),reason='Follows the supplied encoded facts.'))
    group=score_cases([case],[row])['groups']['json_only:diagnostic']
    assert group['policy_following_correct']==1 and group['urgency_correct']==0
    assert group['encoded_answers_correct']==2 and group['facts_correct']==0
