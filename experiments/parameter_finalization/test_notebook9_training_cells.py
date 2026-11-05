"""Local synthetic training/receipt tests; no notebook9 or GPU execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import nbformat
import pytest
from parameters.artifacts import write_json
from parameters.contracts import digest
from parameters.dataset_pipeline import train_spi_for_cache
from parameters.readiness import require_spi_ready
from parameters.spi_direct_priority import PROMPT
from parameters.grounding_scene import suppress_duplicates
from parameters.spatial_provider import make_spatial_provider


def test_notebook_cells_train_before_cache_and_remain_disabled():
    path=Path(__file__).parent/'notebooks/Full_dataset_Labeling_and_parameter_caching.ipynb'
    notebook=nbformat.read(path,as_version=4)
    code=[c.source for c in notebook.cells if c.cell_type=='code']
    assert any('RUN_REAL=False' in c for c in code)
    train=next(i for i,c in enumerate(code) if 'TRAINING=train_spi_for_cache' in c)
    cache=next(i for i,c in enumerate(code) if ' try:cache_parameters' in c)
    assert train<cache and "CONFIG['model_path']=TRAINING['model_path']" in code[train]
    for c in code:compile(c,'notebook9','exec')


def test_spatial_selection_is_single_and_nms_is_class_aware():
    assert type(make_spatial_provider('missing.pt')).__name__=='YOLOWorldSceneProvider'
    assert type(make_spatial_provider('missing',detector='grounding_dino')).__name__=='GroundingDINOSceneProvider'
    with pytest.raises(ValueError):make_spatial_provider('x',detector='both')
    rows=[{'label':'person','box':[0,0,10,10],'score':.9},
          {'label':'person','box':[0,0,10,10],'score':.8},
          {'label':'car','box':[0,0,10,10],'score':.7}]
    assert [r['label'] for r in suppress_duplicates(rows)]==['person','car']


def test_synthetic_four_class_training_creates_run_local_cache_receipt(tmp_path):
    # Real CPU fitting on synthetic separable features; never a real SPI result.
    names=('Low','Medium','High','Critical');rows=[];features=[]
    for split,n in [('train',12),('validation',5)]:
        for k,name in enumerate(names):
            for i in range(n):
                identity=f'{split}_{k}_{i}'
                rows.append({'image_id':identity,'image_sha256':hashlib.sha256(identity.encode()).hexdigest(),
                    'group':identity,'split':split,'available':True,'level':name,'reason':None,
                    'label_source':'vlm_pseudo','label_unit':'whole_image','rubric_sha256':'synthetic_fixture',
                    'teacher_model_id':'fixture','teacher_revision':'f'*40})
                features.append(np.eye(4,dtype=np.float32)[k])
    np.savez(tmp_path/'features.npz',embeddings=np.array(features),image_ids=np.array([r['image_id'] for r in rows]),
        image_sha256=np.array([r['image_sha256'] for r in rows]),feature_signature=np.array('synthetic_fixture'))
    write_json(tmp_path/'labels.json',rows)
    write_json(tmp_path/'current.json',{'spi':{},'wire':{'bytes':35}})
    selection={'accepted':True,'acceptance_basis':'explicit_user_choice','model_id':'fixture',
        'revision':'f'*40,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest()}
    before=(tmp_path/'current.json').read_bytes()
    result=train_spi_for_cache(tmp_path/'features.npz',tmp_path/'labels.json',selection,tmp_path/'student',
                               tmp_path/'current.json',root=tmp_path/'student',seconds=120)
    assert result['cache_allowed'] is True
    model=require_spi_ready(result['model_path'],result['readiness_path'])
    assert model.levels==names
    assert (tmp_path/'current.json').read_bytes()==before
    receipt=json.loads(Path(result['readiness_path']).read_text())
    assert receipt['independent_urgency_validated'] is False
    assert receipt['spi']['rubric_sha256'] == model.artifact['rubric_sha256']
    assert receipt['spi']['feature_signature'] == model.artifact['feature_signature']
    assert Path(receipt['spi']['model_path']).as_posix() == 'models/spi.json'
    assert receipt['diagnostic_review']['accepted'] is False
