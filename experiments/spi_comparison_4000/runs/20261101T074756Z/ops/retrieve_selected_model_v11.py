import hashlib,json,shutil,sys,tarfile
from pathlib import Path,PurePosixPath
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent
OUT=RUN/'downloads/selected_model'
OUT.mkdir(parents=True,exist_ok=True)
OWNER='satyapaladugu';SLUG='crisismmd1000-spi-labeling';VERSION=11
ALLOW=['*.json','*.safetensors','*.bin','*.txt','*.model','*.jinja']
LEVELS=['Low','Medium','High','Critical']
GIB=1024**3

def write_json(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');t.replace(p)
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest().upper()
def fetch_remote(api,remote,local,max_bytes=5*1024**3):
 q=ApiDownloadKernelOutputRequest();q.owner_slug=OWNER;q.kernel_slug=SLUG;q.version_number=VERSION;q.file_path=remote
 with api.build_kaggle_client() as c:redirect=c.kernels.kernels_api_client.download_kernel_output(q)
 response=requests.get(redirect.url,stream=True,timeout=(30,120));response.raise_for_status()
 declared=int(response.headers.get('content-length') or 0)
 if declared<=0 or declared>max_bytes:raise RuntimeError(f'invalid selected artifact size {declared}: {remote}')
 free=shutil.disk_usage(Path(local).parent).free
 if free<=declared+GIB:raise RuntimeError(f'insufficient disk reserve for {remote}: free={free} size={declared}')
 tmp=Path(local).with_suffix(Path(local).suffix+'.partial');h=hashlib.sha256();count=0
 with response:
  with tmp.open('wb') as f:
   for chunk in response.iter_content(8*1024*1024):
    if not chunk:continue
    count+=len(chunk)
    if count>declared:raise RuntimeError('selected artifact exceeded declared length')
    h.update(chunk);f.write(chunk)
   f.flush()
 if count!=declared:raise RuntimeError(f'incomplete selected artifact {count}/{declared}')
 tmp.replace(local)
 return {'remote_path':remote,'local_path':str(local),'bytes':count,'sha256':h.hexdigest().upper()}

def main(archive_path,runtime_path):
 archive=Path(archive_path);runtime_path=runtime_path.strip('/')
 with tarfile.open(archive,'r:') as tf:
  members=[m for m in tf.getmembers() if m.isfile()]
  def json_member(suffix):
   hits=[m for m in members if m.name.endswith(suffix)]
   if len(hits)!=1:raise RuntimeError(f'expected one archive member ending {suffix!r}, got {len(hits)}')
   return json.loads(tf.extractfile(hits[0]).read().decode('utf-8')),hits[0]
  try:pair_selections,selection_member=json_member('comparison/frozen_pair_selections.json')
  except RuntimeError:
   # Not all versions write the concatenated frozen selection; reconstruct only from development candidates.
   pair_selections=[]
   for m in members:
    if m.name.endswith('/candidates.json') and '/features/' not in m.name:
     try:
      rows=json.loads(tf.extractfile(m).read().decode('utf-8'))
      for row in rows:
       if row.get('development'):pair_selections.append(row)
     except Exception:pass
   selection_member=None
  if not pair_selections:raise RuntimeError('no frozen development selections in result archive')
  winner=max(pair_selections,key=lambda r:float(r['development']['macro_f1']))
  encoder=winner['encoder'];head=winner['head'];balance=winner.get('balance');artifact=str(winner.get('artifact',''))
  if not artifact:raise RuntimeError('development winner has no classifier artifact reference')
  tests=[]
  for m in members:
   if m.name.endswith('comparison/final_test_and_head_profiles.json'):
    try:tests=json.loads(tf.extractfile(m).read().decode('utf-8'))
    except Exception:pass
    break
  test_row=next((r for r in tests if r.get('encoder')==encoder and r.get('head')==head and r.get('balance')==balance),None)
  art=artifact.replace('\\','/')
  if '/comparison/' in art:rel='comparison/'+art.split('/comparison/',1)[1]
  elif art.startswith('comparison/'):rel=art
  else:rel='comparison/'+art.lstrip('/')
  model_hits=[m for m in members if m.name.endswith(rel)]
  if len(model_hits)!=1:
   base=PurePosixPath(art).name
   model_hits=[m for m in members if PurePosixPath(m.name).name==base and '/'+balance+'/' in '/'+m.name and '/'+encoder+'/' in '/'+m.name]
  if len(model_hits)!=1:raise RuntimeError(f'could not uniquely resolve selected head artifact {artifact!r}; matches={len(model_hits)}')
  chosen=model_hits[0];head_path=OUT/('classifier_head'+PurePosixPath(chosen.name).suffix)
  with tf.extractfile(chosen) as src,head_path.open('wb') as dst:shutil.copyfileobj(src,dst)
  head_receipt={'archive_member':chosen.name,'local_path':str(head_path),'bytes':head_path.stat().st_size,'sha256':file_sha(head_path)}
  write_json(OUT/'selection.json',{'selection_rule':'highest development macro-F1 among one development-selected balance per encoder/head; test metrics were not used for selection','selection_archive_member':selection_member.name if selection_member else None,'selected':winner,'selected_test_result':test_row})
  write_json(OUT/'class_order.json',[{'class_id':i,'label':name} for i,name in enumerate(LEVELS)])
  project=ROOT.parents[4]
  notebook=json.loads((project/'experiments/spi_comparison_4000/SPI_4000_Comparison.ipynb').read_text(encoding='utf-8'))
  campaign=''.join(notebook['cells'][2]['source']).splitlines()
  token_start=next(i for i,line in enumerate(campaign) if line.startswith('class TokenHead('))
  token_end=next(i for i,line in enumerate(campaign[token_start+1:],token_start+1) if line.startswith('def folds('))
  token_source='import torch\n\n'+'\n'.join(campaign[token_start:token_end])+'\n'
  if winner.get('kind')=='token':(OUT/'token_head_definition.py').write_text(token_source,encoding='utf-8')
  identity_local_source=project/'experiments/spi_comparison_4000/SPI_4000_Comparison.ipynb'
  preprocess={'image_read':'verify SHA-256 against v9 label receipt before use','orientation':'PIL EXIF transpose','color':'RGB','resize_and_padding':'aspect-preserving ImageOps.pad to 224x224, bicubic resampling, RGB(127,127,127) padding','processor':'AutoImageProcessor from selected pinned repo/revision; use_fast=False; do_resize=False; do_center_crop=False when supported; apply processor native normalization','features':'vector heads use L2-normalized global embedding; token heads use float32 spatial tokens and validity mask','class_order':LEVELS,'encoder_identity_preprocessing':identity.get('preprocessing') if 'identity' in locals() else None,'source_notebook_sha256':hashlib.sha256(identity_local_source.read_bytes()).hexdigest().upper()}
  write_json(OUT/'preprocessing_contract.json',preprocess)
  # Keep only the compact evidence needed to reconstruct and audit the winning choice.
  for suffix in ('comparison/support.json','comparison/split_protocol.json','comparison/coverage.json','comparison/study_status.json','comparison/final_test_and_head_profiles.json','comparison/frozen_pair_selections.json','comparison/cost_scope.json'):
   hits=[m for m in members if m.name.endswith(suffix)]
   if len(hits)==1:
    target=OUT/'evidence'/PurePosixPath(hits[0].name).name
    target.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(hits[0]) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
 # Fetch the exact encoder identity saved by this completed v10 run.
 identity_local=OUT/'encoder_identity.json';identity_remote=f"{runtime_path}/comparison/features/{encoder}/identity.json"
 api=KaggleApi();api.authenticate();identity_receipt=fetch_remote(api,identity_remote,identity_local,max_bytes=5*1024**2)
 identity=json.loads(identity_local.read_text(encoding='utf-8'))
 if identity.get('encoder_name')!=encoder:raise RuntimeError('selected encoder identity name mismatch')
 prep=json.loads((OUT/'preprocessing_contract.json').read_text(encoding='utf-8'));prep['encoder_identity_preprocessing']=identity.get('preprocessing');write_json(OUT/'preprocessing_contract.json',prep)
 repo=identity.get('repo');revision=identity.get('revision');expected=identity.get('weights_sha256')
 if not repo or not revision or not isinstance(expected,dict) or not expected:raise RuntimeError('selected encoder identity lacks pinned repo/revision/weight hashes')
 import importlib.metadata
 from huggingface_hub import HfApi,snapshot_download
 hf=HfApi();info=hf.model_info(repo,revision=revision,files_metadata=True)
 if info.sha and info.sha!=revision:raise RuntimeError(f'HF revision mismatch {info.sha} != {revision}')
 files=[]
 for f in info.siblings:
  name=f.rfilename
  if any(PurePosixPath(name).match(pat) for pat in ALLOW):files.append({'path':name,'size':f.size})
 estimated=sum(x['size'] or 0 for x in files)
 encdir=OUT/'encoder_snapshot';free=shutil.disk_usage(OUT).free
 if estimated and free<=estimated+GIB:raise RuntimeError(f'insufficient space for pinned selected encoder files: free={free} need~{estimated}+reserve')
 if estimated>5*GIB:raise RuntimeError(f'pinned selected encoder snapshot exceeds 5 GiB cap: {estimated}')
 snapshot=snapshot_download(repo_id=repo,revision=revision,allow_patterns=ALLOW,local_dir=str(encdir))
 weight_paths=sorted(p for p in encdir.iterdir() if p.is_file() and (p.suffix in ('.safetensors','.bin') or p.name.endswith('.index.json')))
 actual={p.name:file_sha(p) for p in weight_paths}
 if actual!=expected:raise RuntimeError(f'selected checkpoint SHA mismatch for {repo}@{revision}')
 snapshot_files=[]
 for p in sorted(encdir.rglob('*')):
  if p.is_file() and '.cache' not in p.parts:snapshot_files.append({'path':p.relative_to(encdir).as_posix(),'bytes':p.stat().st_size,'sha256':file_sha(p)})
 processor={'verification':'config files included; processor object hash is recorded from exact v10 runtime'}
 local_tf_version=None;processor_sha_observed=None
 try:
  from transformers import AutoImageProcessor
  processor_obj=AutoImageProcessor.from_pretrained(str(encdir),local_files_only=True,use_fast=False)
  proc_dict=processor_obj.to_dict();import hashlib as _h
  processor_sha_observed=_h.sha256(json.dumps(proc_dict,sort_keys=True,default=str).encode()).hexdigest()
  local_tf_version=importlib.metadata.version('transformers')
  processor={'class':type(processor_obj).__name__,'config':proc_dict,'runtime_processor_sha256':identity.get('processor_sha256'),'local_recomputed_processor_sha256':processor_sha_observed,'local_hash_matches_runtime':processor_sha_observed==identity.get('processor_sha256'),'local_transformers_version':local_tf_version,'note':'snapshot files pinned to runtime repo/revision; runtime processor hash remains authoritative'}
  write_json(OUT/'processor_config_observed.json',proc_dict)
 except Exception as e:
  processor.update({'local_processor_instantiation_error':type(e).__name__,'runtime_processor_sha256':identity.get('processor_sha256')})
 write_json(OUT/'processor_identity.json',processor)
 write_json(OUT/'encoder_snapshot_manifest.json',{'repo':repo,'revision':revision,'files':snapshot_files,'weight_sha256_expected':expected,'weight_sha256_actual':actual,'all_weight_hashes_match':actual==expected,'processor_sha256_runtime':identity.get('processor_sha256'),'processor_sha256_local_observed':processor_sha_observed})
 write_json(OUT/'provenance.json',{'kernel':'satyapaladugu/crisismmd1000-spi-labeling','version':11,'source_sha256':'8979A7E9E19C0CD8D29C46EACFA13F3CC0872E59441CB478E402A69467C96166','source_commit':'032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0','input_dataset':'satyapaladugu/spi-4000-training-input','label_source':'v9 saved receipts; no relabeling','selected_encoder':encoder,'repo':repo,'revision':revision,'selection_rule':'development macro-F1 only, no locked-test-based choice','preprocessing':identity.get('preprocessing'),'class_order':LEVELS,'head_artifact':head_receipt,'encoder_identity_file':identity_receipt,'encoder_snapshot':str(encdir),'transformers_setup_pin':'5.5.0','model_weights_verified_by_runtime_file_hashes':actual==expected})
 report={'selected_encoder':encoder,'selected_head':head,'balance':balance,'development_macro_f1':winner.get('development',{}).get('macro_f1'),'test_metrics':test_row.get('test') if test_row else None,'repo':repo,'revision':revision,'head':head_receipt,'encoder_weight_files':len(actual),'all_weight_hashes_match':actual==expected,'encoder_snapshot_bytes':sum(x['bytes'] for x in snapshot_files),'snapshot_path':str(encdir),'identity_file':identity_receipt,'processor':{'runtime_hash':identity.get('processor_sha256'),'local_hash_matches':processor.get('local_hash_matches_runtime'),'class':processor.get('class')}}
 write_json(OUT/'retrieval_receipt.json',report)
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 try:
  main(sys.argv[1],sys.argv[2])
 except Exception as e:
  write_json(OUT/'retrieval_error.json',{'error_type':type(e).__name__,'error':str(e)})
  raise
