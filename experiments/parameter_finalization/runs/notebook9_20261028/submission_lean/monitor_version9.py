from pathlib import Path
import json,hashlib,time,datetime,re,sys,traceback
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parent
SLUG='satyapaladugu/betterscheduler-spi-20261027finalc'
VERSION=9
LOG_DIR=ROOT/'monitor_logs'; LOG_DIR.mkdir(exist_ok=True)
HISTORY=ROOT/'status_history.jsonl'; API_LOG=LOG_DIR/'kaggle_logs_latest.txt'
POLL_SECONDS=60
MAX_MONITOR_SECONDS=9000
TERMINAL={'KernelWorkerStatus.COMPLETE','KernelWorkerStatus.ERROR','KernelWorkerStatus.CANCELLED','COMPLETE','ERROR','CANCELLED'}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def append(row):
    with HISTORY.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def scrub(text):
    text=re.sub(r'(?i)(authorization|token|api[_ -]?key|secret)(["\s:=]+)[^\s,}\]]+',r'\1\2[REDACTED]',text)
    return re.sub(r'(?i)bearer\s+[A-Za-z0-9._~+/=-]+','Bearer [REDACTED]',text)

api=KaggleApi();api.authenticate()
started=time.monotonic();previous=None;previous_log_hash=None;terminal=None
print('Monitor started for existing kernel version 9; unchanged states remain quiet.',flush=True)
while time.monotonic()-started<MAX_MONITOR_SECONDS:
    try:
        response=api.kernels_status(SLUG)
        status=str(response.status)
        failure=getattr(response,'failure_message',None)
        changed=status!=previous
        append({'observed_at_utc':now(),'version':VERSION,'status':status,'failure_message':failure,'status_changed':changed})
        if changed:
            print('STATUS',status,'at',now(),flush=True);previous=status
        try:
            logs=api.kernels_logs(SLUG) or ''
            logs=scrub(str(logs));log_hash=hashlib.sha256(logs.encode()).hexdigest()
            if logs and log_hash!=previous_log_hash:
                API_LOG.write_text(logs,encoding='utf-8')
                print('LOGS_CHANGED',len(logs),'bytes sha256',log_hash,flush=True)
                previous_log_hash=log_hash
        except Exception as log_error:
            append({'observed_at_utc':now(),'version':VERSION,'event':'logs_unavailable','error':type(log_error).__name__+': '+str(log_error)})
        if status in TERMINAL:
            terminal=status;break
    except Exception as error:
        append({'observed_at_utc':now(),'version':VERSION,'event':'status_error','error':type(error).__name__+': '+str(error)})
        print('STATUS_CHECK_ERROR',type(error).__name__,str(error),flush=True)
    time.sleep(POLL_SECONDS)

if terminal is None:
    print('MONITOR_DEADLINE_REACHED; no cancellation issued.',flush=True)
else:
    print('TERMINAL',terminal,'retrieving version-9 outputs.',flush=True)
    output_root=ROOT/'retrieved_version9';output_root.mkdir(exist_ok=False)
    token=None;page=0;all_files=[]
    while True:
        try:
            files,token=api.kernels_output(SLUG,str(output_root),quiet=True,page_token=token,page_size=200)
            page+=1;all_files.extend(files)
            append({'observed_at_utc':now(),'version':VERSION,'event':'output_page','page':page,'files':files,'next_page':bool(token)})
            if not token:break
        except Exception as error:
            append({'observed_at_utc':now(),'version':VERSION,'event':'output_retrieval_error','page':page+1,'error':type(error).__name__+': '+str(error)})
            print('OUTPUT_RETRIEVAL_ERROR',type(error).__name__,str(error),flush=True);break
    receipt={'version':VERSION,'terminal_status':terminal,'retrieved_at_utc':now(),'remote_output_files':all_files,'file_count':len(all_files),'output_root':str(output_root),'kaggle_logs':str(API_LOG) if API_LOG.exists() else None}
    (ROOT/'retrieval_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('RETRIEVAL_DONE files',len(all_files),'root',output_root,flush=True)
