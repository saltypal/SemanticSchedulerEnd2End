"""Prepare a new version of the established SPI kernel, never a new identity."""
from pathlib import Path
import base64
import hashlib
import json
import nbformat
from .build_annotation_notebook import build as source_package
from .artifacts import write_json

KERNEL = 'satyapaladugu/betterscheduler-spi-20261027finalc'
INPUT = 'satyapaladugu/betterscheduler-spiv2-inputs-20261027b'


def build(output, run_id, server_seconds=7200):
    if not 1800 <= server_seconds <= 7200:
        raise ValueError('one submission is bounded to 0.5..2 hours (<=4 aggregate GPUh)')
    output = source_package(output, run_id, scope='four-class SPI v3 teacher controls and grouped student; no spatial/policy training')
    archive = (output / 'source.zip').read_bytes()
    payload = base64.b64encode(archive).decode()
    internal = server_seconds - 480
    setup = f'''import os,sys,json,base64,zipfile,io,hashlib,time
from pathlib import Path
print('START SPI v3 immutable source setup',flush=True)
ROOT=Path('/kaggle/working/spi_source_{run_id}')
ROOT.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
    if any(not (ROOT/name).resolve().is_relative_to(ROOT) for name in archive.namelist()):raise ValueError('unsafe source archive')
    archive.extractall(ROOT)
SOURCE=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name,expected in SOURCE['files'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('source checksum mismatch')
sys.path.insert(0,str(ROOT))
os.environ['BETTER_SOURCE_MANIFEST']=str(ROOT/'SOURCE_MANIFEST.json')
os.environ['HF_HOME']='/kaggle/temp/spi_models'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from parameters.cloud_notebook import CellRecorder
recorder=CellRecorder(CELL_SOURCES,Path('/kaggle/working/experiments/spi_four_class/runs/{run_id}/cell_evidence'))
with recorder.cell(0):
    print('Source',SOURCE['commit'],'files',len(SOURCE['files']),flush=True)
    print('Interpreter',sys.executable,sys.version,flush=True)
    print('Thinking OFF; one GPU1 quantized teacher at a time; cached CLIP; no YOLO or scheduler training',flush=True)
'''
    dependencies = '''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata,importlib.util
    import torch
    print('START dependency/hardware preflight; retain managed Torch/CUDA',flush=True)
    if torch.cuda.device_count()!=2:raise RuntimeError('existing notebook contract requires two GPUs')
    print('GPUs',[torch.cuda.get_device_name(i) for i in (0,1)],flush=True)
    result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
        'transformers==5.3.0','huggingface-hub==1.5.0','accelerate==1.12.0',
        'bitsandbytes==0.49.2','sentencepiece==0.2.2','scikit-learn==1.7.2',
        'nbformat==5.10.4'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=360)
    print(result.stdout,flush=True);result.check_returncode()
    from transformers import Qwen3_5ForConditionalGeneration,Qwen3VLForConditionalGeneration
    if any(importlib.util.find_spec(x) is not None for x in ('fla','causal_conv1d')):
        raise RuntimeError('unverified optional DeltaNet kernels inherited')
    from parameters.artifacts import write_json
    write_json(Path('/kaggle/working/runtime.json'),{'python':sys.version,'interpreter':sys.executable,
        'packages':{p:metadata.version(p) for p in ('torch','numpy','transformers','huggingface-hub','accelerate','bitsandbytes','scikit-learn','nbformat')},
        'gpus':[torch.cuda.get_device_name(i) for i in (0,1)],'cuda':torch.version.cuda,'thinking':False})
    import shutil
    shutil.copy2('/kaggle/working/runtime.json',recorder.output.parent/'runtime.json')
    print('END dependency/hardware preflight',flush=True)
'''
    execute = f'''with recorder.cell(2):
    import signal
    from parameters.cloud_notebook import locate
    from parameters.spi_pipeline import execute
    def cutoff(signum,frame):raise TimeoutError('whole-cell budget exceeded')
    signal.signal(signal.SIGALRM,cutoff);signal.alarm({internal})
    INPUT_ROOT=locate('input_manifest.json').parent
    OUTPUT=execute(INPUT_ROOT,Path('/kaggle/working/experiments/spi_four_class/runs/{run_id}/campaign'),
        seconds={internal-60},maximum_images=300,runtime_receipt=recorder.output.parent/'runtime.json')
    signal.alarm(0)
    print('ACTUAL SPI OUTCOME',json.loads((OUTPUT/'status.json').read_text()),flush=True)
'''
    export = f'''with recorder.cell(3):
    import zipfile
    run=Path('/kaggle/working/experiments/spi_four_class/runs/{run_id}')
    files=[p for p in run.rglob('*') if p.is_file() and '/data/images/' not in str(p) and p.name!='run_recovery.zip']
    from parameters.artifacts import write_json
    write_json(run/'retrieval_checksums.json',{{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        if 'cell_evidence' not in str(p)}})
    with zipfile.ZipFile('/kaggle/working/run_recovery.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,str(p.relative_to(run)))
        archive.write(run/'retrieval_checksums.json','retrieval_checksums.json')
    print('END recovery export; inspect campaign/status.json and verify checksums on retrieval',flush=True)
'''
    sources = [setup, dependencies, execute, export]
    for i, source in enumerate(sources):
        compile(source, f'spi-v3-cell-{i}', 'exec')
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(f'''# Four-class SPI evidence campaign — {run_id}

Repository: SemanticSchedulerEnd2End. Immutable source snapshot embedded and
verified; commit alone does not identify the dirty source. Notebook imports the
shared parameters package; no source fork is edited in Kaggle.

Existing identity: {KERNEL}; publish a new version, not a new notebook.
Input: {INPUT}, up to 300 exposed development images and cached CLIP features.
No independent natural-image SPI gold is supplied. All visual labels generated
here are pseudo-labels. Authored text controls measure rubric conformance only.

Managed Kaggle runtime, not the local .venv. Setup pins/reports actual libraries.
Requires two T4s; GPU1 loads one NF4 teacher with thinking disabled; GPU0 stays
unused because features are cached. Both GPUs count toward quota. Maximum
server {server_seconds}s = {server_seconds*2/3600:g} aggregate GPU-hours.

Stages: input/preflight; two prompt protocols on 48 fixed text controls; bounded
challenger only if needed; image labels; all-four-class support gate; CPU grouped
linear/MLP selection with separate early stopping; model export and metrics.

Outputs: /kaggle/working/experiments/spi_four_class/runs/{run_id}/ and
/kaggle/working/run_recovery.zip. Partial labels/raw requests/status are saved
incrementally and campaign recovery runs on failure. Cell logs and an execution
notebook are saved by CellRecorder; verify remote persistence and retrieval.

Teacher controls do not prove visual urgency accuracy. No class is invented,
removed, or relabeled to fill a quota. A failed gate preserves evidence and stops
the corresponding downstream stage. No scheduler policy or codec is trained.
''')] + [nbformat.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if i == 0 else '') + s)
          for i, s in enumerate(sources)], metadata={'kernelspec': {
              'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
    nbformat.write(notebook, output / 'notebook.ipynb')
    write_json(output / 'kernel-metadata.json', {'id': KERNEL,
        'title': 'BetterScheduler SPI four-class evidence pipeline', 'code_file': 'notebook.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': True,
        'dataset_sources': [INPUT], 'competition_sources': [], 'kernel_sources': []})
    write_json(output / 'PACKAGE.json', {'run_id': run_id, 'kernel': KERNEL,
        'source_sha256': hashlib.sha256(archive).hexdigest(), 'input': INPUT,
        'server_seconds': server_seconds, 'aggregate_gpu_hour_cap': server_seconds*2/3600,
        'status': 'prepared_not_submitted'})
    return output
