"""Validate the actual process protocol with a versioned, test-only codec factory."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('factory_provenance', [False, True])
def test_isolated_worker_produces_benchmark_and_environment(tmp_path, factory_provenance):
    from PIL import Image
    from INFRA.configuration import resolve_config
    from INFRA.runs import Run
    from INFRA.workers import run_benchmark_worker

    source = tmp_path/'external'
    source.mkdir()
    produced = '([JPEGCodec(20)], {"calibration": "fixture"})' if factory_provenance else '[JPEGCodec(20)]'
    (source/'factory.py').write_text(f'from EncDec.Evaluation.kodak_benchmark import JPEGCodec\ndef build(config):\n    return {produced}\n')
    (source/'.gitignore').write_text('__pycache__/\n')
    def git(*args):
        return subprocess.check_output(['git','-c',f'safe.directory={source.as_posix()}',
            '-c','user.name=Test Fixture','-c','user.email=fixture@example.invalid','-C',str(source),*args], text=True).strip()
    git('init')
    git('add','factory.py','.gitignore')
    git('commit','-m','Test-only worker factory')
    dataset = tmp_path/'kodak'
    dataset.mkdir()
    for index in range(1,25):
        Image.new('RGB',(16,16),(index,80,140)).save(dataset/f'kodim{index:02d}.png')
    config = resolve_config(overrides={'module':'EncDec','action':'smoke','implementation':'external_worker',
        'paths':{'results':str(tmp_path/'results'),'kodak':str(dataset)},
        'dependencies':{'external_worker':{'python':sys.executable,'source_root':str(source),
            'revision':git('rev-parse','HEAD'),'factory':'factory:build','timeout_seconds':90}}})
    run = Run(config)
    result = run_benchmark_worker(config,run)
    run.finish(result)
    assert result['failed'] == 0
    assert len(result['outcomes']) == 1
    assert (run.path/'benchmark.csv').is_file()
    assert json.loads((run.path/'worker_provenance.json').read_text())['executable'] == sys.executable
    assert (run.path/'factory_provenance.json').exists() == factory_provenance
