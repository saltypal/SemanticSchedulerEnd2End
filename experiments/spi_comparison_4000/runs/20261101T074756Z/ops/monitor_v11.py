import hashlib,json,os,re,shutil,subprocess,time,traceback,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent
KERNEL='satyapaladugu/crisismmd1000-spi-labeling/11'
OWNER='satyapaladugu';SLUG='crisismmd1000-spi-labeling';VERSION=11
SOURCE_SHA256='8979A7E9E19C0CD8D29C46EACFA13F3CC0872E59441CB478E402A69467C96166'
POLL_SECONDS=60;CAP_SECONDS=24000
TERMINAL={'COMPLETE','ERROR','FAILED','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED','KILLED'}
PROGRESS_PATTERN=r'(^|/)(study_status\.json|coverage\.json|support\.json|candidates\.json|selection_frozen\.json|selection\.json|locked_test\.json|locked_test_status\.json|readiness\.json|terminal\.json|profiles\.json|failures\.json|splits\.json|progress\.json|status\.json|results?\.json|summary\.json|environment\.json|artifact_manifest\.json|outer_progress\.json)$'
ENV=os.environ.copy();ENV['PYTHONIOENCODING']='utf-8';ENV['PYTHONUTF8']='1'
STATE=ROOT/'monitor_state.json';HISTORY=ROOT/'status_history.jsonl';LOGS=ROOT/'logs';COMPACT=ROOT/'remote_compact';ARCHIVES=ROOT/'result_archive';BEST=ROOT/'best_model'
for p in (LOGS,COMPACT,ARCHIVES,BEST):p.mkdir(parents=True,exist_ok=True)

def write_json(path,value):
 tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def cli(args,timeout=300):
 return subprocess.run(['kaggle',*args],capture_output=True,text=True,encoding='utf-8',errors='replace',env=ENV,timeout=timeout)
def current_status():
 r=cli(['kernels','status',KERNEL],timeout=120);s=(r.stdout+'\n'+r.stderr).strip();m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',s);return (m.group(1) if m else 'UNKNOWN'),r.returncode,s
def current_logs():
 r=cli(['kernels','logs',KERNEL],timeout=180)
 lines=[x for x in (r.stdout+'\n'+r.stderr).splitlines() if 'outdated `kaggle` version' not in x]
 return '\n'.join(lines).strip(),r.returncode
def save_log(text,stamp):
 name='kernel_log_v11_'+stamp.strftime('%Y%m%dT%H%M%SZ')+'.txt';p=LOGS/name;p.write_text(text+'\n',encoding='utf-8');return p

def output_files(pattern,dest,timeout=600):
 r=cli(['kernels','output',KERNEL,'--path',str(dest),'--file-pattern',pattern,'--page-size','200','--quiet'],timeout=timeout)
 rec={'pattern':pattern,'returncode':r.returncode,'stdout':r.stdout[-3000:],'stderr':r.stderr[-3000:],'retrieved_files':[]}
 for p in sorted(dest.rglob('*')):
  if p.is_file():
   b=p.read_bytes();rec['retrieved_files'].append({'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest().upper()})
 return r.returncode,rec

def retrieve_archive(api,remote_rel,declared_size=None):
 req=ApiDownloadKernelOutputRequest();req.owner_slug=OWNER;req.kernel_slug=SLUG;req.version_number=VERSION;req.file_path=remote_rel
 with api.build_kaggle_client() as client:redir=client.kernels.kernels_api_client.download_kernel_output(req)
 head=requests.head(redir.url,timeout=(20,45),allow_redirects=True);head.raise_for_status()
 size=int(head.headers.get('content-length') or declared_size or 0)
 if size<=0:raise RuntimeError('archive content length unavailable')
 if size>5*1024**3:raise RuntimeError(f'results archive exceeds 5 GiB retrieval cap ({size})')
 free=shutil.disk_usage(ARCHIVES).free
 if free<=size+1024**3:raise RuntimeError(f'insufficient disk reserve: free={free}, archive={size}, reserve=1073741824')
 partial=ARCHIVES/'SPI_comparison_results.tar.partial';target=ARCHIVES/'SPI_comparison_results.tar'
 digest=hashlib.sha256();count=0
 with requests.get(redir.url,stream=True,timeout=(30,180)) as response:
  response.raise_for_status()
  with partial.open('wb') as out:
   for chunk in response.iter_content(8*1024*1024):
    if not chunk:continue
    count+=len(chunk)
    if count>size:raise RuntimeError('archive download exceeded declared size')
    digest.update(chunk);out.write(chunk)
   out.flush();os.fsync(out.fileno())
 if count!=size:raise RuntimeError(f'archive size mismatch {count}/{size}')
 sha=digest.hexdigest().upper();partial.replace(target)
 return target,count,sha,free

def extract_selected_model(archive):
 with tarfile.open(archive,'r:') as tf:
  members=[m for m in tf.getmembers() if m.isfile()]
  sel=[m for m in members if m.name.endswith('/selection_frozen.json') or m.name.endswith('/selection.json')]
  choice=None;selection_member=None
  for m in sel:
   try:
    d=json.loads(tf.extractfile(m).read().decode('utf-8'))
    artifact=d.get('artifact')
    if artifact:
     choice=d;selection_member=m;break
   except Exception:continue
  if not choice:return None
  raw=str(choice['artifact']).replace('\\','/')
  if '/comparison/' in raw:suffix='comparison/'+raw.split('/comparison/',1)[1]
  else:suffix=raw.lstrip('/')
  candidates=[m for m in members if m.name.endswith(suffix)]
  if len(candidates)!=1:
   name=PurePosixPath(raw).name
   candidates=[m for m in members if PurePosixPath(m.name).name==name and 'comparison/' in m.name]
  if len(candidates)!=1:return {'selection_file':selection_member.name,'artifact_reference':raw,'matches':len(candidates)}
  member=candidates[0]
  out=BEST/PurePosixPath(member.name).name;tmp=out.with_suffix(out.suffix+'.partial')
  src=tf.extractfile(member)
  if src is None:raise RuntimeError('selected model archive member could not be read')
  h=hashlib.sha256();count=0
  with tmp.open('wb') as f:
   for chunk in iter(lambda:src.read(8*1024*1024),b''):
    count+=len(chunk);h.update(chunk);f.write(chunk)
   f.flush();os.fsync(f.fileno())
  tmp.replace(out)
  return {'selection_file':selection_member.name,'artifact_reference':raw,'archive_member':member.name,'local_path':str(out),'bytes':count,'sha256':h.hexdigest().upper(),'selection':{k:choice.get(k) for k in ('encoder','head','balance','development','artifact')}}

started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat();last_status=None;last_log_hash=None;latest_log='';last_log_poll=0;runtime_path=None;status='UNKNOWN';log_snapshot=None
write_json(STATE,{'kernel':KERNEL,'version':VERSION,'status':'starting','pid':os.getpid(),'source_sha256':SOURCE_SHA256,'started_utc':started_utc,'cap_seconds':CAP_SECONDS})
api=KaggleApi();api.authenticate();print(f'MONITOR START pid={os.getpid()} kernel={KERNEL}',flush=True)
while time.monotonic()-started<CAP_SECONDS:
 observed=datetime.now(timezone.utc)
 try:status,exit_code,status_text=current_status()
 except Exception as e:status='UNKNOWN';exit_code=-1;status_text=type(e).__name__
 if status!=last_status:
  row={'observed_utc':observed.isoformat(),'kernel':KERNEL,'status':status,'status_exit_code':exit_code};HISTORY.open('a',encoding='utf-8').write(json.dumps(row)+'\n');print(f"[{row['observed_utc']}] status={status}",flush=True);last_status=status
 if status!=last_status or time.monotonic()-last_log_poll>=120:
  last_log_poll=time.monotonic()
  try:log_text,log_exit=current_logs()
  except Exception as e:log_text='';log_exit=-1
  if log_text:
   digest=hashlib.sha256(log_text.encode()).hexdigest()
   if digest!=last_log_hash:
    log_snapshot=save_log(log_text,observed);latest_log=log_text;last_log_hash=digest
    matches=re.findall(r'/kaggle/working/(experiments/spi_comparison_4000/runs/[^\s\'\"\\]+)',log_text)
    if matches:runtime_path=matches[-1].rstrip('.,)')
    print(f'[{observed.isoformat()}] log_bytes={len(log_text.encode())} snapshot={log_snapshot.name}',flush=True)
 state={'kernel':KERNEL,'version':VERSION,'pid':os.getpid(),'status':status,'updated_utc':observed.isoformat(),'runtime_path':runtime_path,'log_snapshot':str(log_snapshot) if log_snapshot else None,'log_bytes':len(latest_log.encode())}
 write_json(STATE,state)
 if status in TERMINAL:break
 time.sleep(POLL_SECONDS)

elapsed=round(time.monotonic()-started,2);ended=datetime.now(timezone.utc).isoformat();terminal_log_exit=None;retrieval=[];archive_info=None;best_info=None;failure=None
if status in TERMINAL:
 try:
  final_log,terminal_log_exit=current_logs()
  if final_log:
   latest_log=final_log;log_snapshot=save_log(final_log,datetime.now(timezone.utc))
   matches=re.findall(r'/kaggle/working/(experiments/spi_comparison_4000/runs/[^\s\'\"\\]+)',final_log)
   if matches:runtime_path=matches[-1].rstrip('.,)')
 except Exception as e:terminal_log_exit=-1;failure={'log_fetch_error':type(e).__name__}
 (LOGS/'terminal_kernel_log_v11.txt').write_text(latest_log+'\n',encoding='utf-8')
 # Retrieve compact study evidence; never request feature banks or checkpoints.
 pattern=PROGRESS_PATTERN
 out_code,receipt=output_files(pattern,COMPACT)
 retrieval.append(receipt)
 # Retry once if outputs are still being published.
 if not receipt['retrieved_files']:
  time.sleep(20);out_code,receipt=output_files(pattern,COMPACT);retrieval.append(receipt)
 if status=='COMPLETE' and runtime_path:
  remote_rel=runtime_path.rstrip('/')+'/SPI_comparison_results.tar'
  try:
   target,size,sha,free=retrieve_archive(api,remote_rel)
   archive_info={'remote_path':remote_rel,'local_path':str(target),'bytes':size,'sha256':sha,'free_bytes_before':free}
   write_json(STATE,{'kernel':KERNEL,'version':VERSION,'pid':os.getpid(),'status':'retrieving_selected_encoder_and_head','runtime_path':runtime_path,'updated_utc':datetime.now(timezone.utc).isoformat()})
   helper=ROOT/'retrieve_selected_model_v11.py';helper_log=ROOT/'selected_model_retrieval.log'
   with helper_log.open('w',encoding='utf-8') as stream:
    proc=subprocess.run([sys.executable,str(helper),str(target),runtime_path],cwd=str(RUN),stdout=stream,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=10800)
   if proc.returncode:
    failure={'selected_model_retrieval_exit_code':proc.returncode,'log':str(helper_log)}
    best_info=None
   else:
    best_info=json.loads((RUN/'downloads/selected_model/retrieval_receipt.json').read_text(encoding='utf-8'))
  except Exception as e:failure={'archive_error_type':type(e).__name__,'archive_error':str(e)}
 receipt_obj={'kernel':KERNEL,'version':VERSION,'source_sha256':SOURCE_SHA256,'status':status,'monitor_started_utc':started_utc,'observed_utc':ended,'elapsed_seconds':elapsed,'cap_seconds':CAP_SECONDS,'runtime_path':runtime_path,'terminal_log_path':str(LOGS/'terminal_kernel_log_v11.txt'),'terminal_log_bytes':len(latest_log.encode()),'terminal_log_sha256':hashlib.sha256(latest_log.encode()).hexdigest().upper() if latest_log else None,'terminal_log_exit_code':terminal_log_exit,'compact_output_receipts':retrieval,'results_archive':archive_info,'best_model':best_info,'retrieval_issue':failure}
 write_json(ROOT/'monitor_receipt.json',receipt_obj)
 print('MONITOR END '+json.dumps(receipt_obj,separators=(',',':')),flush=True)
else:
 write_json(ROOT/'monitor_receipt.json',{'kernel':KERNEL,'version':VERSION,'status':'monitor_cap_reached','elapsed_seconds':elapsed,'cap_seconds':CAP_SECONDS,'runtime_path':runtime_path,'observed_utc':ended})
 print('MONITOR CAP REACHED; Kaggle job untouched',flush=True)

