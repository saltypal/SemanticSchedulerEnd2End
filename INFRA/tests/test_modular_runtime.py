import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from INFRA.configuration import ROOT, resolve_config

def test_configuration_precedence_and_paths(tmp_path):
    source={'module':'Scheduler','action':'demo','profile':'fast','profiles':{'fast':{'seed':3,'scheduler':{'slot_ms':5}}},
            'seed':7,'paths':{'results':'outputs'}}
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(source))
    result=resolve_config(path,{'seed':11})
    assert result['seed']==11
    assert result['scheduler']['slot_ms']==5
    assert result['scheduler']['horizon_ms']==200
    assert result['paths']['results']==str((tmp_path/'outputs').resolve())

@pytest.mark.parametrize('override',[{'seed':-1},{'device':'tpu'},{'action':'bogus'},{'implementation':'rl'},{'scheduler':{'slot_ms':0}}])
def test_bad_configuration_fails(override):
    with pytest.raises((ValueError,KeyError)):
        resolve_config(overrides=override)

def test_scheduler_import_has_no_model_dependencies():
    code="import sys; import Scheduler; from INFRA.configuration import resolve_config; assert not any(x in sys.modules for x in ('torch','transformers','EncDec','SemParam'))"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

def test_legacy_classes_are_canonical():
    from Evaluation.kodak_benchmark import EncodedSample as old
    from EncDec.Evaluation.kodak_benchmark import EncodedSample as new
    from TransformerPipeline.catalog import SemanticPipelineSettings as old_settings
    from SemParam.catalog import SemanticPipelineSettings as new_settings
    assert old is new
    assert old_settings is new_settings

def test_cli_from_unrelated_directory_writes_traceable_run(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'E2EMain.py'),'--module','INFRA','--action','validate','--output',str(tmp_path)],cwd=tmp_path,check=True,capture_output=True)
    run=next((tmp_path/'INFRA').iterdir())
    assert json.loads((run/'status.json').read_text())['status']=='completed'
    assert json.loads((run/'provenance.json').read_text())['git_commit']
    assert (run/'config.resolved.yaml').is_file()
    assert len(json.loads((run/'jobs.json').read_text()))==6

@pytest.mark.parametrize('filename,action',[('stage1a.yaml','validate'),('stage1b.yaml','semantic-validate')])
def test_legacy_cli_still_runs(filename,action):
    result = subprocess.run([sys.executable,str(ROOT/'E2EMain.py'),'--config',str(ROOT/'config'/filename),
                             '--action',action],cwd=ROOT,check=True,capture_output=True,text=True)
    assert 'validated' in json.loads(result.stdout)['status']
