"""Create one immutable private two-GPU pilot notebook; never launch implicitly."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import subprocess
import zipfile
import nbformat as nbf
from .artifacts import write_json


def build(output,tag,scope='annotation-assisted pilot;GPU0producer/GPU1reviewer;no policy training',extra_sources=()):
    root=Path(__file__).resolve().parents[1];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    files={str(p.relative_to(root)).replace('\\','/'):p.read_bytes() for p in (root/'parameters').glob('*.py')}
    files['project_compat.py']=(root/'project_compat.py').read_bytes()
    for relative in extra_sources:
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('extra source must be an existing canonical project file')
        files[path.relative_to(root).as_posix()]=path.read_bytes()
    for directory,pattern in (('config','*.json'),('schemas','*.json')):
        for path in (root/'parameters'/directory).glob(pattern):files[str(path.relative_to(root)).replace('\\','/')]=path.read_bytes()
    # Only selected canonical dependencies;namespace stubs avoid unrelated eager imports.
    for name in ('Evaluation/decoder_semantic_benchmark.py','Evaluation/decoder_datasets.py',
                 'Scheduler/training/decoder_context_training.py','Scheduler/contextual_decoder_scheduler.py','Scheduler/decoder_scheduler.py'):
        files[name]=(root/name).read_bytes()
    for name in ('Evaluation/__init__.py','Scheduler/training/__init__.py','Scheduler/__init__.py'):
        files[name]=b'"""Isolated experimental namespace;canonical selected modules copied verbatim."""\n'
    manifest={'tag':tag,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'files':{key:hashlib.sha256(value).hexdigest() for key,value in files.items()},
              'scope':scope}
    files['SOURCE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in files.items():archive.writestr(name,value)
    (output/'source.zip').write_bytes(buffer.getvalue());payload=base64.b64encode(buffer.getvalue()).decode()
    setup=f'''import os,sys,json,base64,zipfile,io,hashlib
from pathlib import Path
print("START immutable source setup",flush=True)
ROOT=Path('/kaggle/working/better_source');ROOT.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
    if any(not (ROOT/name).resolve().is_relative_to(ROOT) for name in archive.namelist()):raise ValueError("unsafe archive")
    archive.extractall(ROOT)
SOURCE=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name,expected in SOURCE['files'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError("source checksum mismatch")
sys.path.insert(0,str(ROOT));os.environ['BETTER_SOURCE_MANIFEST']=str(ROOT/'SOURCE_MANIFEST.json')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['YOLO_CONFIG_DIR']='/tmp/better_yolo_config'
from parameters.cloud_notebook import CellRecorder
recorder=CellRecorder(CELL_SOURCES,Path('/kaggle/working/cell_evidence'))
with recorder.cell(0):
    print('Verified source',SOURCE['commit'],SOURCE['tag'])
    print('Actual interpreter',sys.executable,sys.version)
    print('Configuration:GPU0CLIP+YOLOWorld,GPU1Qwen4B;server timeout3600s;internal3000s')'''
    dependencies='''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata,urllib.request
    print("Installing pinned pilot libraries in active Kaggle kernel;keeping managed torch/CUDA",flush=True)
    # OpenAI CLIP is used internally by the public YOLO-World implementation.
    request=urllib.request.Request('https://api.github.com/repos/openai/CLIP/commits/main',headers={'User-Agent':'parameters-research'})
    with urllib.request.urlopen(request,timeout=30) as response:clip_commit=json.load(response)['sha']
    print('OpenAI CLIP source revision',clip_commit)
    commands=[
      [sys.executable,'-m','pip','install','--disable-pip-version-check','transformers==4.57.6','huggingface-hub==0.36.2','accelerate>=1.1,<2','sentencepiece==0.2.2','ftfy','nbformat','pyyaml'],
      [sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','ultralytics==8.3.237','git+https://github.com/openai/CLIP.git@'+clip_commit]]
    for command in commands:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
        print(result.stdout,flush=True);result.check_returncode()
    for package in ('torch','torchvision','numpy','transformers','huggingface-hub','accelerate','ultralytics'):
        print(package,metadata.version(package))
    import torch
    print('Assigned GPUs',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    assert torch.cuda.device_count()==2,'Two actual GPUs required before model download/inference'
    from transformers import Qwen3VLForConditionalGeneration
    from ultralytics import YOLOWorld
    from parameters.preflight_reviewer_processor import run as processor_preflight
    processor_preflight('/kaggle/working/processor_preflight.json')
    from parameters.artifacts import write_json
    write_json(Path('/kaggle/working/dependency_receipt.json'),{'openai_clip_commit':clip_commit,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_count':2})'''
    run=f'''with recorder.cell(2):
    from parameters.cloud_notebook import locate
    from parameters.annotation_pilot import run_annotation_pilot
    MANIFEST=locate('data/records.json')
    print('Verified upstream prepared manifest',MANIFEST,flush=True)
    OUTPUT=run_annotation_pilot(MANIFEST,Path('/kaggle/working/results/annotation_pilot/{tag}'),seconds=3000,control_parents=24,steps_per_ue=6,control_exclusions=ROOT/'parameters/config/reviewer_control_exclusions.json')
    print('Pilot function completed;inspect outcome status,not just notebook success:',OUTPUT)'''
    summary='''with recorder.cell(3):
    from parameters.artifacts import write_json
    status=json.loads((OUTPUT/'status.json').read_text());print('ACTUAL PILOT OUTCOME',status)
    for name in ('control_metrics.json','parameter_summary.json'):
        path=OUTPUT/'metrics'/name
        if path.exists():print(name,json.loads(path.read_text()))
    hashes={str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.rglob('*') if p.is_file()}
    write_json(Path('/kaggle/working/artifact_checksums.json'),hashes)
    print('Saved logs,raw reviews,control truth,parameter/history records and checksum receipt.')
    print('Reviewer gate failure is a preserved scientific outcome,not permission to run extraction anyway.')
    print('No scheduler policies trained;no codec transmitted or decoded;annotations intentionally visible to reviewer.')'''
    sources=[setup,dependencies,run,summary]
    cells=[nbf.v4.new_markdown_cell('''# parameters annotation-assisted parameter pilot

GPU0:CLIP+YOLO-World;GPU1:Qwen3-VL-4B-Instruct. Six simulatedUEs,five fixed tasks,
independent senderKBs. Purpose:assess parameter consistency and mission-fact
sufficiency,not train scheduling policies. Published annotations are intentionally
visible to the offline reviewer and never enter extraction. Reviewer controls
test annotation-assisted auditing,not independent visual recognition.

Managed Kaggle Python/CUDA runtime;setup prints actual packages/devices/source.
Internal cap3000seconds;server timeout3600seconds;2allocated GPUs consume up to
2aggregateGPUhours including setup. Reuse the attached prepared data—no new image
dataset download.72review controls precede GPU0production. Then at most36UE
observations in compact and structured views. Gate failure preserves artifacts.

Outputs:/kaggle/working/results/annotation_pilot/<tag>/ and cell_evidence/ with
executed notebook,cell logs/status,source manifest and checksum receipt. Partial
raw reviews/shards flush incrementally. Saved-version retrieval must be verified.
Kernel-kill/native-process buffers may not be fully captured. This is an
exploratory pilot,not a universal best-parameter or journal-superiority claim.''')]
    for index,source in enumerate(sources):
        compile(source,'pilot_cell_'+str(index),'exec')
        cells.append(nbf.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if index==0 else '')+source))
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(notebook,output/'notebook.ipynb')
    slug='betterscheduler-annotation-pilot-'+tag
    write_json(output/'kernel-metadata.json',{'id':'satyapaladugu/'+slug,'title':'Betterscheduler Annotation Pilot '+tag,
       'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
       'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
       'dataset_sources':[],'competition_sources':[],
       'kernel_sources':['satyapaladugu/betterscheduler-prepare-20261025b']})
    write_json(output/'pilot_manifest.json',{'source_manifest':manifest,'slug':'satyapaladugu/'+slug,
        'server_seconds':3600,'internal_seconds':3000,'maximum_allocated_gpu_hours':2,
        'upstream':'satyapaladugu/betterscheduler-prepare-20261025b'})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args();print(build(args.output,args.tag))
