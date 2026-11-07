"""Execute notebook offline on twelve software fixtures, never download media."""
import json
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experiments/image_pool'
run=HERE/'runs'/('fixture_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
run.mkdir(parents=True,exist_ok=False)
kernel=run/'kernels/pool_fixture';kernel.mkdir(parents=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'pool fixture','language':'python'}),encoding='utf8')
os.environ['IMAGE_POOL_MODE']='fixture';os.environ['IMAGE_POOL_RESUME_DIR']=str(run)
notebook=nbformat.read(HERE/'Image_Pool_Caching.ipynb',4)
manager=KernelManager(kernel_name='pool_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
try:
    NotebookClient(notebook,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    print('ALL CELLS EXECUTED',run,flush=True)
finally:
    nbformat.write(notebook,run/'executed.ipynb')
