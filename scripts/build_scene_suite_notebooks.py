"""Two observable Kaggle templates with frozen source/upstream archive."""
from pathlib import Path
import argparse,base64,hashlib,io,json,subprocess,zipfile

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT.parent/'SemanticSchedulerParameterAudit/parameters'


def snapshot():
    stream=io.BytesIO();hashes={};uphashes={}
    allowed=('__init__.py','contracts.py','extractor.py','pilot_parameters.py','pilot_producers.py',
        'vision.py','image_claims.py','model_settings.py','scene_json.py','ue_profiles.py')
    entries=[]
    for folder in ('Evaluation','Scheduler','Archived.owners_before_simplification_20261027.SemParam'):
        entries.extend((str(p.relative_to(ROOT)).replace('\\','/'),p) for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in str(p) and 'tests' not in p.parts)
    entries.extend((str(p.relative_to(ROOT)).replace('\\','/'),p) for p in (ROOT/'Archived/owners_before_simplification_20261027/SemParam/schemas').glob('*.json'))
    for name in allowed:
        entries.append(('upstream/parameters/'+name,UPSTREAM/name))
    for p in (UPSTREAM/'config').glob('*.json'):entries.append(('upstream/parameters/config/'+p.name,p))
    for p in (UPSTREAM/'schemas').glob('*.json'):entries.append(('upstream/parameters/schemas/'+p.name,p))
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in entries:
            raw=path.read_bytes();archive.writestr(name,raw);hashes[name]=hashlib.sha256(raw).hexdigest()
            if name.startswith('upstream/'):uphashes[name[len('upstream/'):]]=hashes[name]
        archive.writestr('upstream/upstream_manifest.json',json.dumps({'files':uphashes},indent=2))
    raw=stream.getvalue()
    return raw,{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'dirty':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'files':hashes,'archive_sha256':hashlib.sha256(raw).hexdigest()}


def exclusions():
    ids=set()
    def visit(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('image_id','image_sha256','source_sha256') and isinstance(v,str) and len(v)==64:ids.add(v)
                visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for path in (UPSTREAM/'config').glob('*.json'):
        visit(json.loads(path.read_text()))
    for path in (UPSTREAM/'results/local').rglob('*.json'):
        if path.stat().st_size<10_000_000:
            try:visit(json.loads(path.read_text()))
            except (ValueError,UnicodeError):pass
    for path in (ROOT/'results/decoder_context').rglob('*.json'):
        if path.stat().st_size<10_000_000:
            try:visit(json.loads(path.read_text()))
            except (ValueError,UnicodeError):pass
    return sorted(ids)


def build(output,tag,stage,*,gpu=False,data_source=None,seconds=5400,target_counts=(80,80,80),source_splits=('test',),dataset_source=None):
    import nbformat as nbf
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    archive,manifest=snapshot();(output/'source.zip').write_bytes(archive)
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    encoded=base64.b64encode(archive).decode();excluded=exclusions()
    bootstrap=f'''print("START setup: scene suite frozen source", flush=True)
import os,sys,time,json,base64,hashlib,zipfile,io,subprocess,platform,traceback
from pathlib import Path
START=time.perf_counter()
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
RAW=base64.b64decode({encoded!r})
MANIFEST={manifest!r}
assert hashlib.sha256(RAW).hexdigest()==MANIFEST['archive_sha256']
CODE=Path('/kaggle/working/scene_source');CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(RAW)) as z:z.extractall(CODE)
for name,expected in MANIFEST['files'].items():assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==expected
sys.path.insert(0,str(CODE))
RESULT=Path('/kaggle/working/results/scene_suite/{tag}');RESULT.mkdir(parents=True,exist_ok=True)
(RESULT/'source_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
# Linux worker deadline also covers dependency/model setup before extraction.
import signal
def stage_deadline(signum,frame):
    receipt={{'status':'partial','reason':'assigned worker wall-clock budget expired','seconds':time.perf_counter()-START}}
    (RESULT/'deadline.json').write_text(json.dumps(receipt,indent=2))
    raise TimeoutError(receipt['reason'])
signal.signal(signal.SIGALRM,stage_deadline);signal.alarm({max(60,seconds-60)})
subprocess.run([sys.executable,'-m','pip','install','--quiet','jsonschema>=4.23,<5','lightgbm==4.6.0','nbformat>=5.10,<6'],check=True)
import torch,numpy,sklearn,lightgbm,jsonschema
ENV={{'python':sys.executable,'version':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,
     'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'cuda':torch.cuda.is_available(),
     'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}
(RESULT/'environment.json').write_text(json.dumps(ENV,indent=2));print(ENV,flush=True)
print('END setup',round(time.perf_counter()-START,2),'seconds',flush=True)
'''
    if stage=='data':
        execute=f'''print('START official cohort preparation; no model inference',flush=True)
from Scheduler.training.scene_prepare import prepare_data
prepare_data(RESULT/'data', {excluded!r},target_counts={tuple(target_counts)!r},source_splits={tuple(source_splits)!r})
print('END cohort preparation',flush=True)
'''
    elif stage=='extract':
        execute=f'''print('START GPU0 extraction using frozen parameters',flush=True)
import shutil
subprocess.run([sys.executable,'-m','pip','install','--quiet','transformers==4.57.6','huggingface-hub>=0.35,<1','ultralytics==8.3.237','timm'],check=True)
matches=list(Path('/kaggle/input').rglob('data_receipt.json'));assert len(matches)==1,matches
DATA=RESULT/'data';shutil.copytree(matches[0].parent,DATA)
from Scheduler.training.scene_prepare import extract_data
extract_data(DATA,CODE/'upstream',deadline_seconds={max(60,seconds-600)})
print('END extraction',flush=True)
'''
    elif stage=='train':
        execute=f'''print('START suite training; actor sees numeric scene parameters only',flush=True)
matches=list(Path('/kaggle/input').rglob('features.jsonl'));matches=[p for p in matches if (p.parent/'records.json').is_file()]
assert len(matches)==1,matches
from Scheduler.training.scene_study import run_study
run_study(matches[0].parent,CODE/'upstream',RESULT/'study',seconds={max(60,seconds-300)})
print('END suite training',flush=True)
'''
    elif stage=='teacher':
        execute=f'''print('START isolated Qwen worker; GPU1, bounded calls and wall time',flush=True)
import venv
matches=[p for p in Path('/kaggle/input').rglob('features.jsonl') if (p.parent/'records.json').is_file()]
assert len(matches)==1,matches
requests=list(Path('/kaggle/input').rglob('teacher_requests_train.jsonl'));assert len(requests)==1,requests
QENV=Path('/kaggle/working/qwen_env');venv.EnvBuilder(system_site_packages=True,with_pip=True).create(QENV)
QPY=QENV/'bin/python'
subprocess.run([str(QPY),'-m','pip','install','--quiet','transformers==5.3.0','huggingface-hub==1.5.0','accelerate==1.12.0','bitsandbytes==0.49.2'],check=True)
env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONUNBUFFERED='1')
print('Teacher subprocess log destination',RESULT/'teacher_process.log',flush=True)
with (RESULT/'teacher_process.log').open('w') as log:
    subprocess.run([str(QPY),'-m','Scheduler.training.scene_teacher_run','--data',str(matches[0].parent),
        '--requests',str(requests[0].parent),'--output',str(RESULT/'teacher'),'--seconds','{max(60,seconds-600)}'],
        env=env,stdout=log,stderr=subprocess.STDOUT,timeout={max(60,seconds-300)},check=True)
print('END teacher worker; inspect task gate before accepting labels',flush=True)
'''
    else:raise ValueError('stage must be data/extract/train/teacher')
    capture='''
from contextlib import redirect_stdout,redirect_stderr
class Tee:
    def __init__(self,stream,file):self.stream=stream;self.file=file
    def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush();return len(text)
    def flush(self):self.stream.flush();self.file.flush()
'''
    wrapped=capture+f'''
status={{'stage':{stage!r},'status':'running'}}
stage_start=time.perf_counter()
with (RESULT/'stage_stdout.log').open('w') as stdout,(RESULT/'stage_stderr.log').open('w') as stderr:
    with redirect_stdout(Tee(sys.stdout,stdout)),redirect_stderr(Tee(sys.stderr,stderr)):
        try:
            exec({execute!r})
            status['status']='complete'
        except Exception as e:
            status.update(status='failed',error=type(e).__name__,reason=str(e));traceback.print_exc()
        finally:
            status['seconds']=time.perf_counter()-stage_start
            (RESULT/'status.json').write_text(json.dumps(status,indent=2))
            checks={{str(p.relative_to(RESULT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULT.rglob('*') if p.is_file() and p.name!='artifact_checksums.json'}}
            (RESULT/'artifact_checksums.json').write_text(json.dumps(checks,indent=2))
print(status,flush=True)
bundle=RESULT.parent/({tag!r}+'-artifacts.zip')
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in RESULT.rglob('*'):
        if path.is_file():archive.write(path,str(path.relative_to(RESULT)))
print('Recovery bundle',bundle,'bytes',bundle.stat().st_size,flush=True)
if status['status']!='complete':raise RuntimeError('Experiment stage failed; inspect preserved artifacts')
'''
    final='''print('START artifact inventory',flush=True)
checks={str(p.relative_to(RESULT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULT.rglob('*') if p.is_file() and p.name!='artifact_checksums.json'}
(RESULT/'artifact_checksums.json').write_text(json.dumps(checks,indent=2))
print('END inventory',len(checks),'files',RESULT,flush=True)
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# Scene scheduler '+stage+'\n\nFrozen parameters v2 source. Managed Kaggle runtime; no global local environment is assumed. Outputs: `/kaggle/working/results/scene_suite/'+tag+'`.\n\nThis is a bounded feasibility run; independent source annotations and teacher predictions remain separate.'),
        nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell(wrapped),nbf.v4.new_code_cell(final)])
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata['language_info']={'name':'python','version':'3.11'}
    nbf.validate(nb)
    nbf.write(nb,output/'notebook.ipynb')
    metadata={'id':'satyapaladugu/scene-suite-'+stage+'-'+tag.lower(),'title':'Scene Suite '+stage+' '+tag,
        'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':gpu,'enable_internet':True,'dataset_sources':dataset_source.split(',') if dataset_source else [],'competition_sources':[],
        'kernel_sources':data_source.split(',') if data_source else []}
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (output/'launch_config.json').write_text(json.dumps({'seconds':seconds,'gpu':gpu,'stage':stage,'tag':tag,'target_counts':target_counts},indent=2))
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook-cell','exec')
    print(json.dumps({'output':str(output),'stage':stage,'archive_sha256':manifest['archive_sha256'],'files':len(manifest['files'])},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--stage',choices=['data','extract','train','teacher'],required=True);p.add_argument('--gpu',action='store_true')
    p.add_argument('--data-source');p.add_argument('--seconds',type=int,default=5400)
    p.add_argument('--target-counts',type=int,nargs=3,default=(80,80,80))
    p.add_argument('--source-splits',nargs='+',choices=['test','train'],default=['test'])
    p.add_argument('--dataset-source')
    a=p.parse_args();build(a.output,a.tag,a.stage,gpu=a.gpu,data_source=a.data_source,seconds=a.seconds,target_counts=tuple(a.target_counts),source_splits=tuple(a.source_splits),dataset_source=a.dataset_source)
