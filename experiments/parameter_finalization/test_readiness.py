import hashlib
import json
import pytest
from parameters.readiness import require_spi_ready
from parameters.minimal_parameters import compact_parameter_record
from parameters.scene_measurements import SceneMeasurements
from parameters.spi_labels import LEVELS
from experiments.spi_four_class.test_pipeline import linear_artifact


def test_readiness_binds_review_support_and_actual_model(tmp_path):
    artifact=linear_artifact();model=tmp_path/'model.json'
    model.write_text(json.dumps(artifact))
    receipt={'runtime_promotion_allowed':True,'release_checks':{'export_parity':True,'all_class_support':True,'per_class_recall':True},
        'spi':{'file_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
               'model_sha256':artifact['sha256']},
        'diagnostic_review':{'accepted':True,'unresolved_decisive_contradictions':0},
        'class_support':{role:{name:{'rows':n,'groups':g} for name in LEVELS}
                         for role,n,g in [('train',12,6),('validation',5,3)]}}
    path=tmp_path/'ready.json';path.write_text(json.dumps(receipt))
    assert require_spi_ready(model,path).levels==LEVELS
    receipt['class_support']['train']['Medium']['rows']=4
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='support insufficient'):require_spi_ready(model,path)
    receipt['class_support']['train']['Medium']['rows']=12
    receipt['diagnostic_review']['unresolved_decisive_contradictions']=1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='review not accepted'):require_spi_ready(model,path)


@pytest.mark.parametrize('level,available',[(None,False),('Low',True)])
def test_normal_display_never_changes_raw_unknown(level,available):
    bundle={'spi':{'level':level,'available':available,'reason':'fixture' if level is None else None},
        'relevance':{'cosine':None},'novelty':{'value':None,'cold_start':False},
        'capture':{'capture_time_ms':None,'allowed_delay_ms':None},
        'scene':SceneMeasurements.unavailable('fixture')}
    record=compact_parameter_record(bundle)
    assert record['spi_display']=='Normal'
    assert record['spi']==level and record['spi_available']==available
