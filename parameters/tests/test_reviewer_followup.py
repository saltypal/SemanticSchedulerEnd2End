from copy import deepcopy
import json
import pytest
from parameters.annotation_review import control_cases,review_system_prompt
from parameters.model_settings import ReviewerSettings
from parameters.qwen_reviewer import QwenAnnotationReviewer
from parameters.reviewer_followup import prepare_cases,evaluate_cell
from parameters.ue_profiles import load_ue_profiles


def records():
    return [{'source':'ladi','image_id':f'{i:03}','image_path':'fixture.png','group':f'g{i:03}',
             'native_labels':{'buildings_any':True,'buildings_affected_or_greater':bool(i%2)}} for i in range(64)]


def test_followup_keeps_fresh_groups_outside_all_prior_controls(monkeypatch):
    missions,_=load_ue_profiles();source=records()
    spec={'fresh_control_parents':8,'max_review_calls':72,
          'excluded_groups':[f'g{i:03}' for i in range(56)],
          'diagnostic_cases':control_cases(source,missions,parents=2)}
    monkeypatch.setattr('parameters.reviewer_followup.resolve_image',lambda row,manifest:row)
    diagnostic,fresh=prepare_cases(source,missions,spec,'unused')
    assert len(diagnostic)==6 and len(fresh)==24
    assert not {c['reference']['group'] for c in fresh}&set(spec['excluded_groups'])
    poisoned=deepcopy(spec)
    poisoned['diagnostic_cases'][0]['reference']['facts']['building.present']=False
    with pytest.raises(ValueError,match='no longer matches'):
        prepare_cases(source,missions,poisoned,'unused')


def test_followup_records_text_modality_and_scores_unmodified_callback(tmp_path):
    missions,_=load_ue_profiles();cases=control_cases(records(),missions,parents=2)
    class Reviewer:
        def configure(self,settings):self.provenance={'scope':'fixture'}
        def review(self,images,candidate,reference,mission,rubric,history,case_id,raw,**kwargs):
            assert images==[] and kwargs['text_only_diagnostic'] is True
            assert 'expected' not in candidate and 'control_arm' not in candidate
            return {'case_id':case_id,'status':'complete',
                    'parsed':next(c['expected'] for c in cases if c['case_id']==case_id)}
    result=evaluate_cell(Reviewer(),cases,missions,ReviewerSettings(),tmp_path/'cell',lambda:None,True)
    assert result['joint_correct']==6 and result['metrics']['engineering_gate_passed']
    assert result['text_only_diagnostic'] and len((tmp_path/'cell/reviews.jsonl').read_text().splitlines())==6


def test_new_prompt_is_versioned_and_empty_images_require_explicit_diagnostic():
    v2=review_system_prompt('candidate_provenance_v2')
    v4=review_system_prompt('candidate_provenance_explicit_v4')
    assert v4.startswith(v2) and 'All required candidate claims match the known reference.' in v4
    ReviewerSettings(prompt_variant='candidate_provenance_explicit_v4')
    reviewer=QwenAnnotationReviewer('unused','fixture')
    with pytest.raises(ValueError,match='supply images'):
        reviewer.review([],{}, {},None,{},[],'fixture','unused')
    with pytest.raises(ValueError,match='supply images'):
        reviewer.review(['fixture.png'],{}, {},None,{},[],'fixture','unused',text_only_diagnostic=True)


def test_followup_notebook_embeds_exact_cells_and_resource_caps(tmp_path):
    import nbformat
    from parameters.build_reviewer_followup_notebook import build
    package=build(tmp_path/'package','fixture')
    notebook=nbformat.read(package/'notebook.ipynb',as_version=4)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    for source in cells:compile(source,'fixture','exec')
    assert 'reviewer_followup import run' in cells[2] and 'seconds=900' in cells[2]
    assert 'ultralytics' not in cells[1] and 'pip' in cells[1]
    metadata=json.loads((package/'kernel-metadata.json').read_text())
    manifest=json.loads((package/'pilot_manifest.json').read_text())
    assert metadata['is_private'] and manifest['server_seconds']==1200 and manifest['max_review_calls']==72
