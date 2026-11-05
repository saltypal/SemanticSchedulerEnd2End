import hashlib
import io
import json
from pathlib import Path
import sys
import pytest
from parameters.semantic_validation_run import materialize_panel
from parameters.semantic_validation_driver import stream_process
from parameters.artifacts import Run
from parameters.official_panel import near_match,stratum
from parameters.semantic_validation_run import select_review_cohort
from parameters.semantic_panel import panel_cases,make_case


def test_refreshed_asset_metadata_is_checked_before_pixels_reach_extraction(monkeypatch,tmp_path):
    states=((False,False),(True,False),(True,True));records=[];metadata=[]
    for i,(p,d) in enumerate(states):
        image=('fixture'+str(i)).encode();identity=hashlib.sha256(image).hexdigest()
        labels={'buildings_any':p,'buildings_affected_or_greater':d}
        records.append({'source':'ladi','source_row':i,'image_id':identity,'image_path':'unused','group':'g'+str(i),
            'native_labels':labels,'dataset_revision':'revision'})
        metadata.append({'row_idx':i,'row':{**labels,'image':{
            'src':'https://datasets-server.huggingface.co/--/revision/--/'+str(i)+'?temporary=not-persisted'}}})
    monkeypatch.setattr('parameters.semantic_validation_run.fetch_page',lambda *args:{'rows':metadata})
    monkeypatch.setattr('parameters.semantic_validation_run.urlopen',
        lambda req,**kwargs:io.BytesIO(('fixture'+req.full_url.split('/--/')[-1].split('?')[0]).encode()))
    run=Run(tmp_path/'run',{'seconds':10})
    rows=materialize_panel({'panel_records':records},run.path/'data')
    assert len(rows)==3 and all(Path(r['image_path']).is_file() for r in rows)
    assert 'temporary=' not in (run.path/'data/records.json').read_text()
    with pytest.raises(FileExistsError,match='not an empty directory'):
        materialize_panel({'panel_records':records},run.path/'data')
    metadata[1]['row']['buildings_any']=False
    with pytest.raises(ValueError,match='annotations changed'):
        materialize_panel({'panel_records':records},tmp_path/'poisoned')


def test_worker_logs_and_nonzero_exit_are_retained(tmp_path):
    log=tmp_path/'child.log'
    result=stream_process([sys.executable,'-u','-c','print("before failure",flush=True); raise SystemExit(7)'],log,10)
    assert result['returncode']==7 and 'before failure' in log.read_text()


def test_official_states_and_near_duplicate_reference():
    assert stratum({'buildings_any':False,'buildings_affected_or_greater':True})=='inconsistent'
    assert stratum({'buildings_any':True})=='unknown'
    assert near_match({'dhash':'3','std':20},{'dhash':'7','std':20})
    assert not near_match({'dhash':'3','std':0},{'dhash':'7','std':20})


def test_budget_review_subset_retains_three_states_and_full_extraction_cases(tmp_path):
    states=((False,False),(True,False),(True,True))*4
    records=[{'source':'ladi','source_row':i,'image_id':str(i),'image_path':'fixture.jpg','group':'g'+str(i),
        'stratum':stratum({'buildings_any':p,'buildings_affected_or_greater':d}),
        'native_labels':{'buildings_any':p,'buildings_affected_or_greater':d}} for i,(p,d) in enumerate(states)]
    data=tmp_path/'data';data.mkdir()
    controls=panel_cases(records)
    actual=[make_case(r,{'claims':{'building.present':r['native_labels']['buildings_any'],
        'building.affected_or_greater':r['native_labels']['buildings_affected_or_greater']}},'actual-'+str(i))
        for i,r in enumerate(records)]
    for name,value in (('records',records),('controls_evaluator_only',controls),('actual_evaluator_only',actual)):
        (data/(name+'.json')).write_text(json.dumps(value))
    result=select_review_cohort(tmp_path,2)
    selected=json.loads((data/'controls_evaluator_only.json').read_text())
    assert result['parent_images']==6 and len(selected)==24
    assert sum(c['arm']=='condition_masked' for c in selected)==3
    assert len(json.loads((data/'actual_evaluator_only.json').read_text()))==6
    assert len(json.loads((data/'full_controls_evaluator_only.json').read_text()))==48


def test_semantic_notebook_records_actual_workers_and_isolated_runtime(tmp_path):
    import nbformat
    from parameters.build_semantic_validation_notebook import build
    package=build(tmp_path/'package','fixture',server_seconds=1200)
    notebook=nbformat.read(package/'notebook.ipynb',as_version=4)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    for source in cells:compile(source,'semantic-cell','exec')
    assert 'semantic_validation_driver import run' in cells[2]
    assert 'enable_thinking' in cells[1]
    manifest=json.loads((package/'pilot_manifest.json').read_text())
    metadata=json.loads((package/'kernel-metadata.json').read_text())
    assert manifest['server_seconds']==1200 and manifest['max_review_calls']==60
    assert 'server_seconds=1200' in cells[2]
    assert metadata['is_private'] and metadata['kernel_sources']==[]
