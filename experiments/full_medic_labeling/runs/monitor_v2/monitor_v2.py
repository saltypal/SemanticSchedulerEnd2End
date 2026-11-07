import hashlib, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parent
KERNEL='satyapaladugu/medic-full-spi-labeling/2'
OWNER='satyapaladugu'; SLUG='medic-full-spi-labeling'
CAP_SECONDS=12000; INTERVAL=60
history=ROOT/'status_history.jsonl'
start=time.monotonic(); start_utc=datetime.now(timezone.utc)
env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['PYTHONUTF8']='1'
api=KaggleApi(); api.authenticate(); req=ApiListKernelSessionOutputRequest(); req.user_name=OWNER; req.kernel_slug=SLUG; req.page_size=200
last_log_hash=None; state='UNKNOWN'; log_text=''
while time.monotonic()-start < CAP_SECONDS:
    now=datetime.now(timezone.utc)
    proc=subprocess.run(['kaggle','kernels','status',KERNEL],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    combined=(proc.stdout+'\n'+proc.stderr).strip()
    import re
    m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',combined)
    state=m.group(1) if m else 'UNKNOWN'
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.list_kernel_session_output(req)
    log_text=response.log or ''
    digest=hashlib.sha256(log_text.encode('utf-8')).hexdigest()
    record={'observed_utc':now.isoformat(),'kernel':KERNEL,'status':state,'status_exit_code':proc.returncode,'log_bytes':len(log_text.encode('utf-8')),'log_sha256':digest}
    with history.open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    if digest!=last_log_hash and log_text:
        stamp=now.strftime('%Y%m%dT%H%M%SZ')
        (ROOT/f'logs_{stamp}.txt').write_text(log_text,encoding='utf-8')
        last_log_hash=digest
        print(f'[{now.isoformat()}] status={state} log_bytes={record["log_bytes"]} snapshot=logs_{stamp}.txt',flush=True)
    else:
        print(f'[{now.isoformat()}] status={state} log_bytes={record["log_bytes"]}',flush=True)
    if state in {'COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'}: break
    time.sleep(INTERVAL)
elapsed=round(time.monotonic()-start,2)
final={'kernel':KERNEL,'status':state,'monitor_started_utc':start_utc.isoformat(),'observed_utc':datetime.now(timezone.utc).isoformat(),'monitor_elapsed_seconds':elapsed,'cap_seconds':CAP_SECONDS,'runtime_run_path':None}
if log_text:
    for line in log_text.splitlines():
        if '/kaggle/working/experiments/full_medic_labeling/runs/' in line:
            final['runtime_run_path']=line
(ROOT/'monitor_terminal_receipt.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
print('MONITOR TERMINAL',json.dumps(final),flush=True)
