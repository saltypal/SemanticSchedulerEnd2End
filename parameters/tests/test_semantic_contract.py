from copy import deepcopy
import pytest
from parameters.semantic_contract import audit_candidate,decode_claims,BUILDING_CONTRACT
from parameters.semantic_panel import panel_cases,score_panel
from parameters.qwen_reviewer import thinking_template_receipt
from parameters.model_settings import ReviewerSettings

P='building.present';D='building.affected_or_greater'


def test_absence_recovers_condition_but_presence_does_not():
    negative=decode_claims({'claims':{P:False}})
    assert not negative['literal_complete'] and negative['semantic_complete']
    assert negative['decoded_claims']=={P:False,D:False}
    assert negative['origins'][D]['kind']=='derived'
    positive=decode_claims({'claims':{P:True}})
    assert not positive['semantic_complete'] and positive['semantic_missing']==[D]
    affected=decode_claims({'claims':{D:True}})
    assert affected['decoded_claims']=={P:True,D:True}


def test_derived_predictions_are_not_ground_truth_and_conflicts_never_overwrite():
    candidate={'claims':{P:False}}
    result=audit_candidate(candidate,{'facts':{P:True,D:True}})
    assert result['semantic_complete'] and result['factual']['all_required_verified'] is False
    assert set(result['factual']['wrong_fields'])=={P,D}
    conflict=decode_claims({'claims':{P:False,D:True}})
    assert conflict['decoded_claims']=={P:False,D:True} and not conflict['semantic_complete']
    assert len(conflict['conflicts'])==2
    with pytest.raises(ValueError,match='hierarchy'):
        audit_candidate(candidate,{'facts':{P:False,D:True}})


def test_unknown_missingness_masks_and_non_detection_stay_unavailable():
    for candidate in ({'claims':{}},{'claims':{P:None}},
        {'claims':{P:False},'claim_availability':{P:{'available':False}}},
        {'claims':{},'parameters':{'detector_observation':{'building':'not_detected'}}}):
        assert not decode_claims(candidate)['semantic_complete']
    partial=audit_candidate({'claims':{P:False}},{'facts':{P:False}})
    assert partial['semantic_complete'] and partial['factual']['all_required_verified'] is None
    assert partial['factual']['reference_unknown']==[D]
    assert audit_candidate({'claims':{P:True,D:False}},{'facts':{}})['factual']['known_subset_consistent'] is None
    masked=audit_candidate({'claims':{P:False,D:True},'claim_availability':{D:{'available':False}}},
                           {'facts':{P:False,D:False}})
    assert masked['semantic_complete'] and not masked['conflicts'] and masked['factual']['all_required_verified']
    with pytest.raises(ValueError):decode_claims({'claims':{P:'false'}})


def test_exhaustive_three_state_decoder_matches_independent_allowed_worlds():
    # Reference worlds use native presence/affected ontology. Completeness means
    # all compatible worlds agree; a contradictory candidate has no such world.
    worlds=[{P:False,D:False},{P:True,D:False},{P:True,D:True}]
    for p in (None,False,True):
        for d in (None,False,True):
            supplied={P:p,D:d}
            compatible=[w for w in worlds if all(v is None or w[k] is v for k,v in supplied.items())]
            decoded=decode_claims({'claims':supplied})
            assert decoded['semantic_complete']==(len(compatible)==1)
            if compatible:
                for key,value in decoded['decoded_claims'].items():
                    assert all(w[key] is value for w in compatible)


def test_panel_balances_states_and_genuine_omissions_separately():
    records=[{'source':'ladi','source_row':i,'image_id':str(i),'image_path':'fixture.jpg','group':'g'+str(i),
        'native_labels':{'buildings_any':p,'buildings_affected_or_greater':d}}
        for i,(p,d) in enumerate(((False,False),(True,False),(True,True)))]
    cases=panel_cases(records);assert len(cases)==12
    by_id={c['case_id']:c for c in cases}
    assert by_id['control-0-condition_omitted']['expected']['representation_sufficient'] is True
    assert by_id['control-1-condition_omitted']['expected']['representation_sufficient'] is False
    reviews=[{'case_id':c['case_id'],'status':'complete','parsed':c['expected']} for c in cases]
    metrics=score_panel(cases,reviews)
    assert metrics['joint_correct']==12 and metrics['genuinely_missing_condition_rejection']==1
    assert metrics['declared_recoverable_absence_acceptance']==1
    unknown=deepcopy(cases[0]);unknown['expected']={k:None for k in unknown['expected']}
    ungraded=score_panel([unknown],[])
    assert ungraded['fully_gradable']==0


def test_balanced_controls_exercise_masks_with_stale_values():
    states=((False,False),(True,False),(True,True))*2
    records=[{'source':'ladi','source_row':i,'image_id':str(i),'image_path':'fixture.jpg','group':'g'+str(i),
        'native_labels':{'buildings_any':p,'buildings_affected_or_greater':d}} for i,(p,d) in enumerate(states)]
    cases=panel_cases(records);masked=[c for c in cases if c['arm']=='condition_masked']
    assert len(masked)==3
    for case in masked:
        assert case['expected']['parameters_consistent'] is True
        assert case['expected']['representation_sufficient']==(case['stratum']=='absent')


def test_thinking_disable_is_enforced_and_checked_in_rendered_template():
    with pytest.raises(ValueError,match='thinking'):ReviewerSettings(enable_thinking=True)
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['enable_thinking'] is False
            return '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    assert thinking_template_receipt(Processor(),[],'qwen3_5')['thinking_disabled_verified']
    class Wrong(Processor):
        def apply_chat_template(self,*args,**kwargs):return '<|im_start|>assistant\n<think>\n'
    with pytest.raises(RuntimeError):thinking_template_receipt(Wrong(),[],'qwen3_5')
