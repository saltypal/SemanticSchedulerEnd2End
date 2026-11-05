from pathlib import Path
import base64, hashlib, json, zipfile, io, subprocess, time, shutil
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'experiments/parameter_finalization/runs/notebook9_20261028'
OUT = RUN / 'submission_v13_features'
if OUT.exists():
    raise FileExistsError(f'Preserve existing package: {OUT}')
OUT.mkdir(parents=True)
SOURCE_NOTEBOOK = ROOT / 'experiments/parameter_finalization/notebooks/Full_dataset_Labeling_and_parameter_caching.ipynb'
nb = nbf.read(SOURCE_NOTEBOOK, as_version=4)
source_codes = [c.source for c in nb.cells if c.cell_type == 'code']
if len(source_codes) != 7:
    raise ValueError(f'Expected the reviewed seven canonical code cells; found {len(source_codes)}')

# Snapshot the run's exact working-tree source bytes, including dirty/untracked modules.
files = {}
for p in sorted((ROOT / 'parameters').rglob('*.py')):
    if '/tests/' in p.as_posix() or '/results/' in p.as_posix() or '__pycache__' in p.parts:
        continue
    files[p.relative_to(ROOT).as_posix()] = p.read_bytes()
for folder in ('config', 'schemas'):
    for p in sorted((ROOT / 'parameters' / folder).glob('*.json')):
        files[p.relative_to(ROOT).as_posix()] = p.read_bytes()
files['project_compat.py'] = (ROOT / 'project_compat.py').read_bytes()
# The pinned HF model processor/config/tokenizer are small and must match cached feature provenance.
clip_dir = ROOT / 'models/priority_classifiers/clip'
for name in ('config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json'):
    p=clip_dir/name;files[p.relative_to(ROOT).as_posix()]=p.read_bytes()
try:
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
except Exception:
    revision = 'unavailable-dirty-snapshot'
source_manifest = {
  'run_id':'notebook9_20261028_v13', 'canonical_source_notebook':str(SOURCE_NOTEBOOK.relative_to(ROOT)),
  'git_head':revision, 'working_tree_snapshot':True,
  'scope':'Notebook9 full 300-image development cohort; teacher labels, grouped SPI fit, gated parameter cache',
  'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
  'asset_policies':{
    'hf_clip_repo':'openai/clip-vit-base-patch32','hf_clip_revision':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
    'hf_clip_weight_sha256':'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f',
    'hf_clip_tokenizer_assets':{name:{'size':(clip_dir/name).stat().st_size,'sha256':hashlib.sha256((clip_dir/name).read_bytes()).hexdigest()} for name in ('tokenizer.json','vocab.json','merges.txt')},
    'yolo_world_sha256':'9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792',
    'openai_clip_sha256':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
  }
}
files['SOURCE_MANIFEST.json'] = json.dumps(source_manifest,indent=2).encode()
buf = io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in files.items(): z.writestr(name,data)
source_zip = buf.getvalue()
(OUT/'source.zip').write_bytes(source_zip)
(OUT/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2),encoding='utf-8')

config = {
  'input_root':'/kaggle/input/betterscheduler-spiv2-inputs-20261027b',
  'clip_checkpoint':'/kaggle/working/project/models/priority_classifiers/clip',
  'detector':'yolo_world',
  'detector_checkpoint':'/kaggle/working/project/models/yolov8s-worldv2.pt',
  'device':'cuda:0'
}
(OUT/'runtime_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

# Expose reviewed source/config as immutable literals in the packaged notebook; config is separately hashed.
zip_b64 = base64.b64encode(source_zip).decode('ascii')
config_bytes = json.dumps(config,sort_keys=True,separators=(',',':')).encode()
config_b64 = base64.b64encode(config_bytes).decode('ascii')
config_sha = hashlib.sha256(config_bytes).hexdigest()
source_sha = hashlib.sha256(source_zip).hexdigest()
bootstrap = f'''import os,sys,json,base64,hashlib,zipfile,io,subprocess,time,shutil,urllib.request
from pathlib import Path
print('START immutable Notebook9 package bootstrap',flush=True)
PKG=Path('/kaggle/working/notebook9_package'); ROOT=Path('/kaggle/working/project')
PKG.mkdir(exist_ok=False); ROOT.mkdir(exist_ok=False)
SOURCE_BYTES=base64.b64decode({zip_b64!r})
if hashlib.sha256(SOURCE_BYTES).hexdigest()!={source_sha!r}: raise RuntimeError('embedded source archive checksum mismatch')
with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as z:
    if any(not (ROOT/name).resolve().is_relative_to(ROOT) for name in z.namelist()): raise ValueError('unsafe source path')
    z.extractall(ROOT)
SOURCE=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name,digest in SOURCE['files'].items():
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise RuntimeError('source hash mismatch: '+name)
CONFIG_BYTES=base64.b64decode({config_b64!r})
if hashlib.sha256(CONFIG_BYTES).hexdigest()!={config_sha!r}: raise RuntimeError('runtime config hash mismatch')
CONFIG_PATH=Path('/kaggle/working/notebook9_config.json'); CONFIG_PATH.write_bytes(CONFIG_BYTES)
os.environ['SEMANTIC_PROJECT_ROOT']=str(ROOT);os.environ['PARAMETER_RUN_REAL']='1'
os.environ['PARAMETER_RUN_ID']='20261028';os.environ['PARAMETER_CONFIG']=str(CONFIG_PATH)
os.environ['PARAMETER_OUTPUT_ROOT']='/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'
CONFIG=json.loads(CONFIG_BYTES)
os.environ['YOLO_CONFIG_DIR']='/kaggle/temp/notebook9_ultralytics'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='300'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
sys.path.insert(0,str(ROOT))
CELL_SOURCES={source_codes!r}
from parameters.cloud_notebook import CellRecorder
INTERNAL_CELLS=['dependency and two-GPU preflight plus pinned model assets',*CELL_SOURCES[:3],
 'input-bundle verification, CLIP signature equality and real YOLO-WORLD warmup',*CELL_SOURCES[3:]]
RECORDER=CellRecorder(INTERNAL_CELLS,Path('/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v13/cell_evidence'))
print('Verified source files',len(SOURCE['files']),'source SHA-256',{source_sha!r},'config SHA-256',{config_sha!r},flush=True)
print('END immutable Notebook9 package bootstrap',flush=True)
'''

preflight = r'''import importlib.metadata as md, torch, gc, hashlib, json, shutil, subprocess, urllib.request, time, zipfile
from pathlib import Path
print('START dependency, hardware, model-asset preflight',flush=True)
commands=[
 [sys.executable,'-m','pip','install','--disable-pip-version-check','setuptools==80.9.0'],
 [sys.executable,'-m','pip','install','--disable-pip-version-check','transformers==5.5.0','huggingface-hub==1.5.0','accelerate==1.12.0','bitsandbytes==0.49.2','sentencepiece==0.2.2','scikit-learn==1.7.2','nbformat==5.10.4','ftfy==6.3.1','opencv-python==4.11.0.86','polars==1.37.1','polars-runtime-32==1.37.1','ultralytics-thop==2.0.18'],
 [sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','ultralytics==8.3.237'],
 [sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--no-build-isolation','git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33']]
for command in commands:
 print('RUN',command,flush=True);result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600);print(result.stdout,flush=True);result.check_returncode()
import torch
if torch.cuda.device_count()!=2: raise RuntimeError('Notebook9 requires exactly two allocated GPUs before downloading or inference')
GPU_NAMES=[torch.cuda.get_device_name(i) for i in range(2)]
if any('T4' not in name for name in GPU_NAMES): raise RuntimeError('Runtime must match reviewed two-T4 resource contract: '+repr(GPU_NAMES))
print('Actual interpreter',sys.executable,sys.version,'torch',torch.__version__,'CUDA',torch.version.cuda,'GPUs',GPU_NAMES,flush=True)

def download(url,destination,expected_sha,expected_bytes,timeout=90):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);part=destination.with_suffix(destination.suffix+'.partial')
    h=hashlib.sha256();count=0;started=time.monotonic()
    request=urllib.request.Request(url,headers={'User-Agent':'Notebook9-parameter-finalization/1'})
    with urllib.request.urlopen(request,timeout=timeout) as response,part.open('wb') as out:
        while True:
            block=response.read(1024*1024)
            if not block: break
            count+=len(block)
            if count>expected_bytes: raise RuntimeError('asset exceeds declared byte cap')
            h.update(block);out.write(block);out.flush()
    if count!=expected_bytes or h.hexdigest()!=expected_sha: raise RuntimeError('downloaded asset size/hash mismatch: '+str(destination))
    part.replace(destination);print('Verified asset',destination,'bytes',count,'sha256',h.hexdigest(),'seconds',round(time.monotonic()-started,2),flush=True)

# Download only the pinned HF model tensor; config/processor/tokenizer are source-archive members.
from huggingface_hub import hf_hub_download
clip_root=Path(CONFIG['clip_checkpoint']);clip_root.mkdir(parents=True,exist_ok=True)
for rel in ('config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json'):
    if not (clip_root/rel).is_file(): raise RuntimeError('source archive missing HF CLIP file: '+rel)
REMOTE_CLIP_FILES={name:(entry['size'],entry['sha256']) for name,entry in SOURCE['asset_policies']['hf_clip_tokenizer_assets'].items()}
for name,(size,expected) in REMOTE_CLIP_FILES.items():
    downloaded=Path(hf_hub_download(repo_id='openai/clip-vit-base-patch32',filename=name,revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'))
    actual_size=downloaded.stat().st_size;actual_sha=hashlib.sha256(downloaded.read_bytes()).hexdigest()
    print('Tokenizer asset',name,'actual bytes/hash',actual_size,actual_sha,'expected',size,expected,flush=True)
    if len(expected)!=64 or actual_size!=size or actual_sha!=expected: raise RuntimeError('pinned HF CLIP tokenizer asset size/hash mismatch: '+name)
    shutil.copy2(downloaded,clip_root/name)
weight=hf_hub_download(repo_id='openai/clip-vit-base-patch32',filename='pytorch_model.bin',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
weight_dst=clip_root/'pytorch_model.bin';shutil.copy2(weight,weight_dst)
if hashlib.sha256(weight_dst.read_bytes()).hexdigest()!='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f':raise RuntimeError('pinned HF CLIP weight hash differs from locally verified feature producer')
download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt',CONFIG['detector_checkpoint'],'9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792',25923032)
openai_cache=Path.home()/'.cache/clip/ViT-B-32.pt'
download('https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',openai_cache,'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',353976522)
DEPENDENCIES={p:md.version(p) for p in ('torch','numpy','transformers','huggingface-hub','accelerate','bitsandbytes','scikit-learn','nbformat','ultralytics','setuptools','ftfy')}
RUNTIME={'python':sys.version,'interpreter':sys.executable,'packages':DEPENDENCIES,'cuda':torch.version.cuda,'gpus':GPU_NAMES,'source_sha256':hashlib.sha256(SOURCE_BYTES).hexdigest(),'runtime_config_sha256':hashlib.sha256(CONFIG_BYTES).hexdigest(),'run_id':'notebook9_20261028_v13'}
Path('/kaggle/working/notebook9_runtime.json').write_text(json.dumps(RUNTIME,indent=2))
print('END dependencies and pinned assets preflight',flush=True)
'''

input_preflight = r'''import json,zipfile,hashlib,gc,time
from PIL import Image
from parameters.spi_campaign import verify_inputs
from parameters.contracts import MissionSpec
from parameters.extractor import embedding_signature
from parameters.minimal_vision import MinimalVisionProvider
from parameters.spatial_provider import make_spatial_provider
INPUT=Path(CONFIG['input_root'])
rows,INPUT_RECEIPT=verify_inputs(INPUT)
if len(rows)!=300: raise RuntimeError('Existing input must contain exactly 300 verified rows')
manifest=json.loads((INPUT/'input_manifest.json').read_text())
expected=manifest['feature_signature']
profile=json.loads((ROOT/'parameters/config/ue_parameter_profiles_v2.json').read_text())
if profile['feature_signature']!=expected: raise RuntimeError('profile/input feature signature mismatch')
row=rows[0]
if INPUT_RECEIPT['layout']=='zip':
    with zipfile.ZipFile(INPUT/'images.zip') as z: raw=z.read(row['image_path'])
    from io import BytesIO
    image=Image.open(BytesIO(raw)).convert('RGB')
else:
    image_path=INPUT/'images'/row['image_path']
    if hashlib.sha256(image_path.read_bytes()).hexdigest()!=row['image_sha256']: raise RuntimeError('first-image SHA mismatch')
    image=Image.open(image_path).convert('RGB')
first_profile=profile['profiles'][0]
mission=MissionSpec(first_profile['mission_id'],first_profile['paragraph'],((first_profile['mission_id'],first_profile['paragraph']),),allowed_delay_ms=first_profile['allowed_delay_ms'])
vision=MinimalVisionProvider(CONFIG['clip_checkpoint'],{mission.mission_id:mission},device='cuda:0')
try:
    vision.provider.load()
    evidence=vision.extract_batch([image],[mission])[0]
    measured=embedding_signature(evidence)
    print('Input layout',INPUT_RECEIPT,'CLIP signature',measured,'expected',expected,flush=True)
    if measured!=expected: print('Old cached feature signature differs; canonical labeling cell will re-extract all300 in this runtime before fitting',flush=True)
finally: vision.close();del vision;gc.collect();torch.cuda.empty_cache()
scene=make_spatial_provider(CONFIG['detector_checkpoint'],detector='yolo_world',device='cuda:0')
try:
    observation=scene.measure(image)
    print('YOLO-WORLD vocabulary',len(scene.vocabulary),'status',observation.status,'reason',observation.reason,'measurements',observation.values(),flush=True)
    if observation.status!='ok': raise RuntimeError('real YOLO-WORLD first-image preflight failed; stop before teacher labeling')
    if len(scene.vocabulary)!=16: raise RuntimeError('runtime vocabulary changed from declared 16-class specification')
finally:
    if hasattr(scene,'close'): scene.close()
    del scene;gc.collect();torch.cuda.empty_cache()
PRECHECK={'input_images':len(rows),'input_manifest_sha256':hashlib.sha256((INPUT/'input_manifest.json').read_bytes()).hexdigest(),'feature_signature':measured,'first_image_sha256':row['image_sha256'],'detector':'yolo_world','detector_status':observation.status,'detector_vocabulary_count':16,'gpus':GPU_NAMES,'status':'passed'}
Path('/kaggle/working/notebook9_prelabel_preflight.json').write_text(json.dumps(PRECHECK,indent=2))
print('END pre-label real feature/detector checks; teacher cell may now start',flush=True)
'''

# Compose independent stage snippets with CellRecorder and execute under one guarded orchestration cell.
stages = [preflight, *source_codes[:3], input_preflight, *source_codes[3:]]
stage_code = "\n\ntry:\n" + "\n".join(
    f"    with RECORDER.cell({i}):\n        exec(compile(CELL_SOURCES[{i-1 if i>0 else 0}], 'notebook9_source_cell_{i-1}', 'exec'), globals())"
    for i in [])
# Build explicit orchestration calls; stage 0 is dependency preflight; indices 1..3 source cells 0..2;
# stage 4 input preflight; indices 5..8 source cells 3..6.
lines = ['print("START Notebook9 staged run",flush=True)','run_failed=None','try:']
for i,src in enumerate(stages):
    lines.append(f'    with RECORDER.cell({i}):')
    if i==0:
        lines.append('        exec(compile(DEPENDENCY_PREFLIGHT, "dependency_preflight", "exec"), globals())')
    elif i<4:
        lines.append(f'        exec(compile(CELL_SOURCES[{i-1}], "notebook9_source_cell_{i-1}", "exec"), globals())')
    elif i==4:
        lines.append('        exec(compile(INPUT_PREFLIGHT, "input_detector_preflight", "exec"), globals())')
    else:
        lines.append(f'        exec(compile(CELL_SOURCES[{i-2}], "notebook9_source_cell_{i-2}", "exec"), globals())')
lines += [
 'except BaseException as error:',
 '    run_failed=error; print("NOTEBOOK9 RUN FAILED; preserving current outputs",type(error).__name__,str(error),flush=True)',
 'finally:',
 '    run_root=Path("/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v13")',
 '    run_root.mkdir(parents=True,exist_ok=True)',
 '    for src,dst in ((Path("/kaggle/working/notebook9_runtime.json"),run_root/"runtime.json"),(Path("/kaggle/working/notebook9_prelabel_preflight.json"),run_root/"prelabel_preflight.json"),(Path("/kaggle/working/notebook9_config.json"),run_root/"runtime_config.json"),(ROOT/"SOURCE_MANIFEST.json",run_root/"SOURCE_MANIFEST.json")):',
 '        if src.exists(): shutil.copy2(src,dst)',
 '    if run_root.exists():',
 '        payload=[p for p in run_root.rglob("*") if p.is_file() and p.name not in ("retrieval_checksums.json","run_recovery.zip")]',
 '        checks={str(p.relative_to(run_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}',
 '        (run_root/"retrieval_checksums.json").write_text(json.dumps(checks,indent=2))',
 '        with zipfile.ZipFile("/kaggle/working/run_recovery.zip","w",zipfile.ZIP_DEFLATED) as recovery:',
 '            for p in payload: recovery.write(p,str(p.relative_to(run_root)))',
 '            recovery.write(run_root/"retrieval_checksums.json","retrieval_checksums.json")',
 '    print("Recovery archive updated; files",len(payload),"failed",repr(run_failed),flush=True)',
 'if run_failed is not None: raise run_failed',
 'print("END Notebook9 staged run",flush=True)'
]
run_cell = '\n'.join(lines)
compile(run_cell,'notebook9_staged_run','exec')

newnb=nbf.v4.new_notebook()
newnb.metadata=nb.metadata
newnb.cells=[nbf.v4.new_markdown_cell('''# Notebook9 parameter finalization — staged v13 runtime-feature package\n\nExisting private Kaggle identity `satyapaladugu/betterscheduler-spi-20261027finalc` (ID 141089219), version 13. This package enables real execution only in the deployed runtime config. It requires two T4 GPUs and caps server time at 7,200 seconds (4 aggregate GPU-hours). Source snapshot and separately embedded runtime config are SHA-256 verified. The attached 300-image development input is hash checked; first-image HF CLIP feature-signature equality and actual 16-class YOLO-WORLD inference must pass before any Qwen labeling. No synthetic run substitutes for real data.\n\nExpected outputs: `/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v13/`, incremental per-stage logs/status and `executed.ipynb`, plus `/kaggle/working/run_recovery.zip`. A `finally` block refreshes recovery after success or failure. Full run remains subject to cache readiness gates; teacher labels are VLM pseudo-labels, not independent urgency truth.'''),
 nbf.v4.new_code_cell(bootstrap),
 nbf.v4.new_code_cell("DEPENDENCY_PREFLIGHT = "+repr(preflight)+"\nINPUT_PREFLIGHT = "+repr(input_preflight)+"\nCELL_SOURCES = "+repr(source_codes)+"\n"+run_cell)]
newnb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
newnb.metadata.kaggle={'accelerator':'NvidiaTeslaT4','isInternetEnabled':True}
nbf.validate(newnb)
nbf.write(newnb,OUT/'notebook.ipynb')
if (OUT/'notebook.ipynb').stat().st_size >= 900_000:
    raise ValueError('staged notebook exceeds the 900,000-byte payload target; trim snapshot before any submission')
metadata={'id':'satyapaladugu/betterscheduler-spi-20261027finalc','id_no':141089219,'title':'BetterScheduler SPI 20261027finalc','code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,'dataset_sources':['satyapaladugu/betterscheduler-spiv2-inputs-20261027b'],'competition_sources':[],'kernel_sources':[]}
(OUT/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
package={'run_id':'notebook9_20261028_v13','kernel':metadata['id'],'id_no':metadata['id_no'],'version_expected':13,'input':metadata['dataset_sources'][0],'source_sha256':source_sha,'notebook_sha256':hashlib.sha256((OUT/'notebook.ipynb').read_bytes()).hexdigest(),'runtime_config_sha256':config_sha,'source_file_count':len(source_manifest['files']),'server_seconds':7200,'aggregate_gpu_hour_cap':4.0,'accelerator':'2 x Tesla T4 expected; runtime preflight asserts actual count/names','status':'prepared_not_submitted','canonical_notebook_sha256':hashlib.sha256(SOURCE_NOTEBOOK.read_bytes()).hexdigest(),'source_snapshot_head':revision,'source_snapshot_dirty_tree':True,'notes':['No Kaggle submission performed','Configuration deployed separately from canonical source and sets PARAMETER_RUN_REAL=1','Input and CLIP signature plus real detector first-image preflights precede teacher labeling','Asset downloads are size/hash bounded; no local weights uploaded']}
(OUT/'PACKAGE.json').write_text(json.dumps(package,indent=2),encoding='utf-8')
print(json.dumps(package,indent=2))
