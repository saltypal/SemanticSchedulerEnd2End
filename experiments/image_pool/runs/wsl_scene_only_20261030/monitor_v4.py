import hashlib,json,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KERNEL='satyapaladuguplus/semantic-image-pool-scene-cache-20261030/4'
SOURCE_SHA256='DD51805844DC2B4B67D613F790795F6D8AEE202BB01C8FC0F1C4C15D553DADC2'
POLL_SECONDS=60
CAP_SECONDS=24600
TERMINAL={'COMPLETE','ERROR','FAILED','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED','KILLED'}
CLI='/home/Satya/.venvs/semantic-kaggle/bin/kaggle'
PATTERN=(r'(^|/)(ACCEPTANCE\.json|artifact_manifest\.json|pool_manifest\.json|source_status\.json|references\.json|source\.json|environment\.json|scene_producer\.json|training_receipt\.json|'
         r'logs/(01_source|02_model_selection|03_acquisition|04_scene_cache|05_spi_cache|06_handoff|07_costs|08_results)\.(json|log)|'
         r'scene/(scene_cache\.sqlite|cache_identity\.json|registry\.json|environment\.json|progress\.json|failures\.json|summary\.json|scene_records\.jsonl|scheduler_arrays\.npz)|'
         r'handoff/(image_parameters\.jsonl|reference_records\.json))$')
def run(args,timeout=180):
 return subprocess.run([CLI,*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
def status():
 r=run(['kernels','status',KERNEL],timeout=120); t=r.stdout+'\n'+r.stderr
 m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',t)
 return (m.group(1) if m else 'UNKNOWN'),r.returncode

def logs():
 r=run(['kernels','logs',KERNEL],timeout=180); t=(r.stdout+'\n'+r.stderr).strip()
 t=re.sub(r'(?i)bearer\s+[A-Za-z0-9._~+/-]+=*','Bearer <redacted>',t)
 t=re.sub(r'(?i)(api[_ -]?key|token|password|secret|authorization)(\s*[:=]\s*)(["\']?)[^\s,\'"}]+',r'\1\2<redacted>',t)
 return t,r.returncode

def digest(p):
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):n+=len(b);h.update(b)
 return n,h.hexdigest().upper()
start=time.monotonic();start_utc=datetime.now(timezone.utc).isoformat();history=ROOT/'status_history_v2.jsonl';last_status=None;last_log_hash=None;last_log_poll=0.0;latest_log='';runtime=None;cur='UNKNOWN'
while time.monotonic()-start<CAP_SECONDS:
 now=datetime.now(timezone.utc);cur,ec=status()
 if cur!=last_status:
  rec={'observed_utc':now.isoformat(),'kernel':KERNEL,'status':cur,'status_exit_code':ec}
  with history.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
  print('STATUS '+json.dumps(rec,separators=(',',':')),flush=True);last_status=cur
 if cur!=last_status or time.monotonic()-last_log_poll>=120:
  last_log_poll=time.monotonic()
  try:t,log_ec=logs()
  except Exception as e:t='';log_ec=-1
  if t:
   lh=hashlib.sha256(t.encode()).hexdigest().upper()
   if lh!=last_log_hash:
    max_chars=10*1024*1024
    saved=t[-max_chars:];truncated=len(t)>len(saved)
    snap=ROOT/f'kaggle_v4_logs_{now.strftime("%Y%m%dT%H%M%SZ")}.txt';snap.write_text(saved+'\n',encoding='utf-8')
    latest_log=saved;last_log_hash=lh
    for line in t.splitlines():
     if '/kaggle/working/experiments/image_pool/runs/' in line:
      m=re.search(r'(/kaggle/working/experiments/image_pool/runs/[^\s\"\\]+)',line)
      if m:runtime=m.group(1)
    print(f'LOG bytes={len(saved.encode())} truncated={truncated} file={snap.name}',flush=True)
 if cur in TERMINAL:break
 time.sleep(POLL_SECONDS)
elapsed=round(time.monotonic()-start,2);ended=datetime.now(timezone.utc);retrieval=ROOT/'retrieved_v4';retrieval.mkdir(exist_ok=True);ret_code=None;ret_out_bytes=0;ret_err_bytes=0
if cur in TERMINAL:
 pattern=PATTERN
 r=run(['kernels','output',KERNEL,'--path',str(retrieval),'--file-pattern',pattern,'--quiet'],timeout=600)
 ret_code=r.returncode;ret_out_bytes=len(r.stdout.encode());ret_err_bytes=len(r.stderr.encode())
files=[];total=0
for p in sorted(retrieval.rglob('*')):
 if not p.is_file():continue
 n,h=digest(p);total+=n;rel=str(p.relative_to(ROOT)).replace('\\','/')
 files.append({'path':rel,'bytes':n,'sha256':h})
 if p.name=='ACCEPTANCE.json':
  try:files[-1]['acceptance']=json.loads(p.read_text(encoding='utf-8'))
  except Exception:files[-1]['parse_error']=True
 if p.name=='source_status.json':
  try:files[-1]['source_status']=json.loads(p.read_text(encoding='utf-8'))
  except Exception:files[-1]['parse_error']=True
receipt={'kernel':KERNEL,'version_number':4,'kernel_id':141684654,'source_sha256':SOURCE_SHA256,'status':cur,'monitor_started_utc':start_utc,'observed_utc':ended.isoformat(),'monitor_elapsed_seconds':elapsed,'cap_seconds':CAP_SECONDS,'runtime_path':runtime,'log_bytes':len(latest_log.encode()),'log_sha256':hashlib.sha256(latest_log.encode()).hexdigest().upper() if latest_log else None,'retrieval_exit_code':ret_code,'retrieval_stdout_bytes':ret_out_bytes,'retrieval_stderr_bytes':ret_err_bytes,'retrieved_file_count':len(files),'retrieved_bytes':total,'retrieved_files':files,'excluded_media_and_weights':True}
(ROOT/'monitor_v4_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print('MONITOR END '+json.dumps(receipt,separators=(',',':')),flush=True)
