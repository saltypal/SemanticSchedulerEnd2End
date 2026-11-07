"""Execute every notebook cell locally in software-fixture mode, no downloads."""
import json
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'experiments/scene_array_cache'
run_id = 'fixture_validation_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
run = HERE/'runs'/run_id
run.mkdir(parents=True, exist_ok=False)
kernel_dir = run/'kernels'/'scene_array_fixture'
kernel_dir.mkdir(parents=True)
(kernel_dir/'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Scene array fixture', 'language': 'python'}), encoding='utf8')
os.environ['SCENE_ARRAY_MODE'] = 'fixture'
os.environ['SCENE_ARRAY_RUN_ID'] = run_id
notebook = nbformat.read(HERE/'Scene_Array_Caching.ipynb', 4)
kernel = KernelManager(kernel_name='scene_array_fixture',
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
try:
    NotebookClient(notebook, km=kernel, timeout=180,
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    print('EXECUTED ALL CELLS', run, flush=True)
finally:
    nbformat.write(notebook, run/'executed.ipynb')
    print('SAVED', run/'executed.ipynb', flush=True)
