"""Promotion contracts without vision inference or remote execution."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import base64
import io
import re

ROOT=Path(__file__).resolve().parents[2]

def test_legacy_imports_share_canonical_state():
    for old,new in [('experiments.stream_scheduler.semantic_suite_data','Scheduler.training.stream_suite.semantic_suite_data'),
                    ('experiments.master_study.paired_runtime','Scheduler.training.study.paired_runtime')]:
        assert importlib.import_module(old) is importlib.import_module(new)

def test_canonical_runtime_modules_do_not_import_experimental_implementations():
    for owner in ['stream_suite','study']:
        for path in (ROOT/'Scheduler/training'/owner).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    assert not (node.module or '').startswith('experiments.'),str(path)

def test_old_and_new_rate_cli_work_outside_checkout(tmp_path):
    for relative in ['experiments/stream_scheduler/ue_rate_study.py','Scheduler/training/stream_suite/ue_rate_study.py']:
        result=subprocess.run([sys.executable,str(ROOT/relative),'--help'],cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert '--ues' in result.stdout

def test_paired_notebook_capsules_include_live_implementation():
    for profile,slug in [('wsl','satyapaladuguplus/semantic-decoder-scheduler-20261030'),('windows','satyapaladugu/semantic-decoder-scheduler-rl')]:
        folder=ROOT/'Notebooks/SchedulerSuite/paired'/profile
        assert json.loads((folder/'kernel-metadata.json').read_text())['id']==slug
        book=json.loads(next(folder.glob('*.ipynb')).read_text())
        code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
        blob=base64.b64decode(re.search(r"base64.b64decode\('([^']+)'\)",code)[1])
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for owner in ['stream_suite','study']:
                for source in (ROOT/'Scheduler/training'/owner).glob('*.py'):
                    name=source.relative_to(ROOT).as_posix()
                    assert z.read(name)==source.read_bytes(),name
