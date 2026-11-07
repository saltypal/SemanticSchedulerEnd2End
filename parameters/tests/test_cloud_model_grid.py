import json
from pathlib import Path
from copy import deepcopy
import pytest
from parameters.cloud_model_grid import parity_matches
from parameters.annotation_review import control_cases
from parameters.ue_profiles import load_ue_profiles


def test_detection_cache_parity_detects_changed_boxes_or_missing_predictions():
    rows=[{'label':'building','score':.4,'box':[0.,0.,10.,10.]},
          {'label':'road','score':.1,'box':[1.,1.,8.,8.]}]
    assert parity_matches(rows,[rows[0]])
    assert not parity_matches(rows,[])
    changed=deepcopy(rows[0]);changed['box'][0]=5.
    assert not parity_matches(rows,[changed])


def test_campaign_control_groups_exclude_all_prior_reviews():
    records=[{'source':'ladi','image_id':str(i),'image_path':'fixture.png','group':'g'+str(i),
        'native_labels':{'buildings_any':True,'buildings_affected_or_greater':bool(i%2)}} for i in range(12)]
    cases=control_cases(records,load_ue_profiles()[0],parents=8,excluded_groups=['g0','g1'])
    assert len(cases)==24 and len({c['reference']['group'] for c in cases})==8
    assert not {c['reference']['group'] for c in cases}&{'g0','g1'}


def test_grid_notebook_records_executed_grid_cells_and_bounded_resources(tmp_path):
    import nbformat as nbf
    from parameters.build_grid_notebook import build
    output=build(tmp_path/'package','fixture')
    notebook=nbf.read(output/'notebook.ipynb',as_version=4)
    code=[c.source for c in notebook.cells if c.cell_type=='code']
    for source in code:compile(source,'grid-cell','exec')
    assert 'cloud_model_grid import run' in code[2]
    assert 'grid_control_exclusions.json' in code[2]
    assert 'annotation_pilot import run_annotation_pilot' not in code[0]
    meta=json.loads((output/'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['kernel_sources']==['satyapaladugu/betterscheduler-prepare-20261025b']
    assert meta['id']=='satyapaladugu/betterscheduler-model-grid-fixture'
    run=json.loads((output/'grid_manifest.json').read_text())
    assert run['server_seconds']==3600 and run['maximum_allocated_gpu_hours']==2
