import hashlib, json, os, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KERNEL='satyapaladugu/crisismmd1000-spi-labeling/3'
CAP_SECONDS=24000; INTERVAL=60
start=time.monotonic(); start_utc=datetime.now(timezone.utc)
env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['PYTHONUTF8']='1'
last_hash=None; state='UNKNOWN'; log_text=''
while time.monotonic()-start<CAP_SECONDS:
 now=datetime.now(timezone.utc)
 p=subprocess.run(['kaggle','kernels','status',KERNEL],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
 status_text=(p.stdout+'\n'+p.stderr).strip(); match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',status_text); state=match.group(1) if match else 'UNKNOWN'
 lp=subprocess.run(['kaggle','kernels','logs',KERNEL],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
 log_text=lp.stdout or ''
 if lp.stderr.strip() and 'outdated `kaggle` version' not in lp.stderr: log_text+='\n[CLI STDERR]\n'+lp.stderr
 digest=hashlib.sha256(log_text.encode('utf-8')).hexdigest()
 rec={'observed_utc':now.isoformat(),'kernel':KERNEL,'status':state,'status_exit_code':p.returncode,'logs_exit_code':lp.returncode,'log_bytes':len(log_text.encode('utf-8')),'log_sha256':digest}
 with (ROOT/'status_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
 if digest!=last_hash and log_text.strip():
  stamp=now.strftime('%Y%m%dT%H%M%SZ'); (ROOT/f'logs_{stamp}.txt').write_text(log_text,encoding='utf-8'); last_hash=digest
  print(f'[{now.isoformat()}] status={state} log_bytes={rec["log_bytes"]} snapshot=logs_{stamp}.txt',flush=True)
 else: print(f'[{now.isoformat()}] status={state} log_bytes={rec["log_bytes"]}',flush=True)
 if state in {'COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'}:break
 time.sleep(INTERVAL)
elapsed=round(time.monotonic()-start,2)
final={'kernel':KERNEL,'status':state,'monitor_started_utc':start_utc.isoformat(),'observed_utc':datetime.now(timezone.utc).isoformat(),'monitor_elapsed_seconds':elapsed,'cap_seconds':CAP_SECONDS,'notebook_sha256':'A13863DB6FC898927138B877C2F9EABC7A9950828DA293F6D77DB2A59FF21B58','runtime_run_path':None}
for line in log_text.splitlines():
 if '/kaggle/working/experiments/full_medic_labeling/runs/' in line:final['runtime_run_path']=line
(ROOT/'monitor_terminal_receipt.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
print('MONITOR TERMINAL',json.dumps(final),flush=True)
