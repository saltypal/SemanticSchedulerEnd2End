import hashlib
import json
from pathlib import Path
import pytest
from parameters.review_only import copy_review_inputs
from parameters.semantic_validation_run import review_case_order


def test_pinned_cache_rebased_without_changing_candidate_or_reference(tmp_path):
    source=Path(__file__).resolve().parents[1]/'results/ops/20261025T2256Z/semantic_validation_h3/monitor/retrieved/results/semantic_validation/20261026h3'
    if not source.is_dir():pytest.skip('retained h3 artifact fixture unavailable')
    expected={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('manifest.json','checksums.json')}
    target=tmp_path/'reused'
    copy_review_inputs(source,target,expected)
    old=json.loads((source/'data/actual_evaluator_only.json').read_text())
    new=json.loads((target/'data/actual_evaluator_only.json').read_text())
    assert len(new)==6
    for before,after in zip(old,new):
        assert before['candidate']==after['candidate'] and before['reference']==after['reference']
        assert Path(after['image_path']).is_file()
    with pytest.raises(FileExistsError):copy_review_inputs(source,target,expected)
    with pytest.raises(ValueError,match='identity mismatch'):
        copy_review_inputs(source,tmp_path/'wrong',{'manifest.json':'not-the-pinned-manifest'})


def test_largest_actual_preflight_once_then_all_remaining_cases():
    cases=[{'arm':arm,'stratum':kind,'candidate':{'claims':{}}} for kind,arm in (
        ('absent','complete'),('present_unaffected','condition_omitted'),('affected','wrong_condition'))]
    cases += [{'arm':'actual_extractor','stratum':'affected','candidate':{'extra':'x'*size}} for size in (5,500,10)]
    order,count=review_case_order(cases,'largest_actual_first')
    assert count==4 and order[:4]==[4,0,1,2]
    assert len(order)==len(set(order))==len(cases) and set(order)==set(range(len(cases)))
    legacy,n=review_case_order(cases)
    assert n==3 and legacy[:3]==[0,1,2]
    with pytest.raises(ValueError):review_case_order(cases,'arbitrary')


def test_review_only_notebook_contains_one_reviewer_and_bounded_budget(tmp_path):
    import nbformat
    from parameters.build_review_only_notebook import build
    package=build(tmp_path/'package','fixture-review')
    notebook=nbformat.read(package/'notebook.ipynb',as_version=4)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    for cell in cells:compile(cell,'review-only-cell','exec')
    assert 'review_only import run' in cells[2]
    assert 'semantic_validation_driver import run' not in cells[2]
    metadata=json.loads((package/'kernel-metadata.json').read_text())
    manifest=json.loads((package/'review_manifest.json').read_text())
    pilot=json.loads((package/'pilot_manifest.json').read_text())
    assert metadata['kernel_sources']==['satyapaladugu/betterscheduler-semantic-validation-20261026h3']
    assert manifest['server_seconds']==1800 and manifest['planned_reviews']==30
    assert manifest['enable_thinking'] is False
    assert pilot['server_seconds']==manifest['server_seconds'] and pilot['internal_seconds']==1650
    assert pilot['maximum_allocated_gpu_hours']==1 and pilot['upstream']==metadata['kernel_sources'][0]
