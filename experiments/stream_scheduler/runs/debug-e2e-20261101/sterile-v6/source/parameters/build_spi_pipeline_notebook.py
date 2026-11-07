"""Prepare a new version of the established SPI kernel, never a new identity."""
from pathlib import Path
import base64
import hashlib
import json
import nbformat
from .build_annotation_notebook import build as source_package
from .artifacts import write_json

KERNEL = 'satyapaladugu/betterscheduler-spi-20261027finalc'
KERNEL_ID = 141089219
KERNEL_TITLE = 'BetterScheduler SPI 20261027finalc'
INPUT = 'satyapaladugu/betterscheduler-spiv2-inputs-20261027b'


def build(output, run_id, server_seconds=7200, diagnostic=False, gemma=False, gemma_all300=False, qwen_all300=False, direct_priority=False, comparison=False):
    if comparison and not (diagnostic and gemma):raise ValueError('comparison requires diagnostic/Gemma runtime')
    if direct_priority and not (gemma_all300 or qwen_all300):
        raise ValueError('direct priority requires an explicitly selected all300 run')
    if qwen_all300 and (gemma or not diagnostic):raise ValueError('Qwen comparison requires diagnostic=True, gemma=False')
    if gemma_all300 and not gemma:raise ValueError('all300 mode requires Gemma')
    if gemma and not diagnostic:
        raise ValueError('Gemma starts with diagnostic only; bulk work requires separate readiness')
    if not 1800 <= server_seconds <= 7200:
        raise ValueError('one submission is bounded to 0.5..2 hours (<=4 aggregate GPUh)')
    extras = ('experiments/spi_four_class/visual_reliability.py',
              'experiments/spi_four_class/grounding_panel.json') if diagnostic else ()
    if gemma_all300 or qwen_all300:extras+=('experiments/spi_four_class/gemma_300.py',)
    if comparison:extras+=('experiments/spi_four_class/compare_vlms.py',)
    output = source_package(output, run_id, scope=(
        'SPI visual grounding diagnosis; no bulk labeling, training or promotion' if diagnostic else
        'four-class SPI v3 teacher controls and grouped student; no spatial/policy training'),
        extra_sources=extras)
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
    if diagnostic:
        execute = execute.replace('from parameters.spi_pipeline import execute',
            'from experiments.spi_four_class.visual_reliability import execute')
    if gemma:
        dependencies = dependencies.replace('transformers==5.3.0','transformers==5.5.0')
        execute = execute.replace('    INPUT_ROOT=locate',
            '    from parameters.gemma_spi_reviewer import load_gemma\n    INPUT_ROOT=locate')
        execute = execute.replace('seconds='+str(internal-60)+',maximum_images=300,',
            'seconds='+str(internal-60)+',maximum_images=300,teacher_loader=load_gemma,teacher_specs=({"name":"gemma4_e4b"},),')
    if gemma_all300:
        execute=execute.replace('from experiments.spi_four_class.visual_reliability import execute',
                               'from experiments.spi_four_class.gemma_300 import execute')
    if qwen_all300:
        execute=execute.replace('from experiments.spi_four_class.visual_reliability import execute',
                               'from experiments.spi_four_class.gemma_300 import execute')
        execute=execute.replace('    INPUT_ROOT=locate',
            '    from parameters.spi_teachers import load_teacher, TEACHERS\n    INPUT_ROOT=locate')
        execute=execute.replace('maximum_images=300,',
            'maximum_images=300,teacher_loader=load_teacher,teacher_spec=TEACHERS[1],')
    if direct_priority:
        execute=execute.replace('maximum_images=300,','maximum_images=300,label_mode="direct",')
    if comparison:
        execute=execute.replace('from experiments.spi_four_class.visual_reliability import execute',
                               'from experiments.spi_four_class.compare_vlms import execute')
        execute=execute.replace('    from parameters.gemma_spi_reviewer import load_gemma\n','')
        execute=execute.replace('teacher_loader=load_gemma,teacher_specs=({"name":"gemma4_e4b"},),','')
    export = f'''with recorder.cell(3):
    print('Prepare recovery export after cell completion; preserve final cell status/logs',flush=True)
# Package after CellRecorder closes so the bundle contains the completed cell,
# rather than a stale running status and a log ending at CELL START.
import zipfile
run=Path('/kaggle/working/experiments/spi_four_class/runs/{run_id}')
files=[p for p in run.rglob('*') if p.is_file() and '/data/images/' not in str(p)
       and p.name not in ('run_recovery.zip','retrieval_checksums.json')]
from parameters.artifacts import write_json
write_json(run/'retrieval_checksums.json',{{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
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
    if diagnostic:
        notebook.cells[0].source = f'''# SPI visual grounding diagnostic — {run_id}

Same established notebook: {KERNEL}, numeric ID {KERNEL_ID}.
Canonical project SemanticSchedulerEnd2End; embedded immutable source hashes.
Managed Kaggle Python; pinned and actual dependencies/hardware recorded below.
Requires two T4 GPUs; teachers loaded sequentially on GPU1, thinking OFF.

Purpose: investigate confirmed visual-grounding errors from version 3. Reuse
its frozen 24-image enriched development subset and baseline responses; compare
Qwen3.5-9B and Qwen3-VL-8B with one identical tightened evidence prompt. Run the
48 authored text controls for each; these are not image accuracy references.
First three images are the two audited counterexamples and one routine case.
No class quotas, bulk labeling, student training or automatic promotion.

Sampling categories and prior responses remain evaluator-only; model receives
only the image and fixed instruction. Prior AI-audited visible-fact findings are
diagnostic findings, not human gold or independent urgency labels. Disagreement
and class-balance changes do not establish accuracy. Review actual claims.

Input {INPUT}; existing images only, no new dataset download.
Server cap {server_seconds}s, at most {server_seconds*2/3600:g} aggregate GPU-hours.
Outputs: /kaggle/working/experiments/spi_four_class/runs/{run_id}/,
campaign recovery ZIP, root run_recovery.zip and cell_evidence logs/notebook.
The setup cell records actual interpreter, source, libraries, devices and CUDA.
Partial outputs survive in campaign recovery; verify retrieval and checksums.
'''
    if gemma:
        notebook.cells[0].source += '\nGemma-only challenger: google/gemma-4-E4B-it; Transformers5.5.0; native Gemma processor/template. Qwen v5 results remain preserved; exact same panel and prompt.\n'
    if gemma_all300:
        notebook.cells[0].source=f'''# Gemma all-300 labeling — {run_id}
Explicit user-authorized extension after the24-image diagnostic. Same frozen
v5 prompt and native Gemma adapter; all300 existing development images. Unknowns
remain unknown. No text controls rerun, training, parameter cache or promotion.
Transformers5.5.0, NF4/FP16 GPU1, thinking OFF; setup records actual source,
interpreter, libraries and devices. Existing notebook identity {KERNEL}.
Server cap {server_seconds}s; both assigned GPUs count toward conservative cap.
Per-image wall/generation timing, generated tokens, raw outputs and receipts
persist incrementally. FLOPs NOT instrumented; never substitute zero.
Outputs: /kaggle/working/experiments/spi_four_class/runs/{run_id}/campaign;
partial campaign recovery on failure and root run_recovery.zip on completion.
'''
    if qwen_all300:
        notebook.cells[0].source=f'''# Qwen3-VL-8B all300 comparison — {run_id}
Same300 development images, frozenv5 prompt, strict parser and abstention policy
as Gemma7. Existing native Qwen3-VL adapter, NF4FP16GPU1, thinkingOFF; immutable
resolved revision and actual runtime recorded. Transformers5.3.0. No training,
promotion or parameter cache. Partial labels/raw requests/timing saved perimage.
Server cap{server_seconds}s. FLOPs not instrumented; no accuracy claim without
independent reference. Same notebook identity{KERNEL}, new version only.
'''
    if direct_priority:
        notebook.cells[0].source += '''
DIRECT FOUR-CLASS PROTOCOL: disaster/emergency-management scene triage.
Return one word CRITICAL/HIGH/MEDIUM/LOW. No nested boolean claims or semantic
abstention. Invalid output/runtime errors remain explicit processing failures.
New prompt version; do not compare label availability as factual accuracy.
No classifier training, promotion or parameter cache is authorized by this run.
'''
    if comparison:
        notebook.cells[0].source=f'''# Version8 under10B VLM comparison —{run_id}
Five models: Gemma4E4Bit, Qwen3.5-9B, Qwen3-VL-8B-Instruct,
Qwen2.5-VL-7B-Instruct, InternVL3-8B-hf (NOT3.5). Same24 frozen
development images and direct4class prompt;48 authored text controls each.
No automatic winner, student training, all300 labeling or parameter caching.
Review factual grounding and class coverage, not agreement or willingness alone.
Transformers5.5.0; native model-specific processors; NF4FP16 GPU1 sequentially.
Thinking not requested; model-specific template checks and output-marker guards.
Nativeplain-instruct checkpoints without thinking toggles are recorded honestly.
Immutable source verifies below; actual interpreter, libraries and devices logged.
Servercap{server_seconds}s; conservativeaggregatecap{server_seconds*2/3600:g}GPUh.
Same existing notebook{KERNEL}; sourcepack and partialraw outputs/checksums saved
under /kaggle/working/experiments/spi_four_class/runs/{run_id}/campaign.
FLOPs not instrumented; response timing recorded. Model revisions pinned/resolved.
Candidate load/inference failure remains a failed row; no silent substitution.
GroundingDINO is a separate conditional localization audit, not a VLM contestant.
'''
    nbformat.write(notebook, output / 'notebook.ipynb')
    write_json(output / 'kernel-metadata.json', {'id': KERNEL, 'id_no': KERNEL_ID,
        'title': KERNEL_TITLE, 'code_file': 'notebook.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': True,
        'dataset_sources': [INPUT], 'competition_sources': [], 'kernel_sources': []})
    write_json(output / 'PACKAGE.json', {'run_id': run_id, 'kernel': KERNEL,
        'source_sha256': hashlib.sha256(archive).hexdigest(), 'input': INPUT,
        'server_seconds': server_seconds, 'aggregate_gpu_hour_cap': server_seconds*2/3600,
        'diagnostic': diagnostic, 'gemma':gemma, 'gemma_all300':gemma_all300,
        'qwen_all300':qwen_all300, 'direct_priority':direct_priority, 'comparison':comparison, 'status': 'prepared_not_submitted'})
    return output
