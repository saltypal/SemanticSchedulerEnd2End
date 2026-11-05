"""Finite monitor for accepted Kaggle scheduler version 3; no relaunches."""
from datetime import datetime, timezone
import hashlib, json, re, shutil, subprocess, time
from pathlib import Path

ROOT = Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End')
RUN = ROOT / 'experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z'
MON = RUN / 'monitor'
RETRIEVED = RUN / 'retrieved/version3'
KAGGLE = '/home/Satya/.venvs/semantic-kaggle/bin/kaggle'
KERNEL = 'satyapaladuguplus/semantic-decoder-scheduler-20261030/3'
MAX_SECONDS = 2 * 60 * 60 + 50 * 60
POLL_SECONDS = 60
TERMINAL = {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELLED_BY_USER', 'FAILED'}
PICK_RE = re.compile(r'(^|/)(REPORT\.md|REPORT\.json|manifest\.json|environment\.json|training_environment\.json|cell_status\.json|driver_exit\.json|method_ledger\.json|baseline_summary\.json|training_summary\.json|run_summary\.json|status\.json|artifact_manifest\.json|[^/]+\.log|[^/]+\.ipynb)$', re.I)

MON.mkdir(parents=True, exist_ok=True)
start_utc = datetime.now(timezone.utc).isoformat()
start = time.monotonic()
deadline = start + MAX_SECONDS
(MON / 'monitor_start.json').write_text(json.dumps({'kernel': KERNEL, 'start_utc': start_utc, 'max_seconds': MAX_SECONDS, 'poll_seconds': POLL_SECONDS, 'retrieval_policy': 'selected small manifests, reports, logs, executed notebook only; exclude datasets, model weights, checkpoints'}, indent=2))
status_path = MON / 'status_history.jsonl'
last_status = None
last_log_hash = None
last_quota = 0.0

def run(args, timeout=90):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)

def append(path, record):
    with path.open('a') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')

def parse_status(s):
    m = re.search(r'KernelWorkerStatus\.([A-Z_]+)', s)
    return m.group(1) if m else 'UNKNOWN'

def get_quota():
    p = run(['quota'], timeout=60)
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    if p.returncode == 0:
        m = re.search(r'GPU\s+([0-9.]+)h\s+([0-9.]+)h\s+([0-9.]+)h', p.stdout)
        if m:
            row.update(gpu_used_hours=float(m.group(1)), gpu_remaining_hours=float(m.group(2)), gpu_total_hours=float(m.group(3)))
    append(MON / 'quota_history.jsonl', row)
    return row

def parse_file_page(raw):
    pos = raw.find('[')
    if pos < 0:
        raise ValueError('Kaggle files response did not contain JSON array')
    data = json.loads(raw[pos:])
    token_match = re.search(r'Next Page Token = (.+)', raw[:pos])
    return token_match.group(1).strip() if token_match else None, data

def inventory_and_retrieve():
    pages=[];token=None
    for _ in range(30):
        args=['kernels','files',KERNEL,'--format','json','--page-size','200']
        if token: args.extend(['--page-token', token])
        p=run(args,timeout=120)
        (MON / ('files_page_%02d.txt' % len(pages))).write_text(p.stdout+'\nSTDERR\n'+p.stderr)
        if p.returncode: raise RuntimeError('Kaggle output inventory failed: '+p.stderr[-1000:])
        token,rows=parse_file_page(p.stdout);pages.extend(rows)
        if not token:break
    else: raise RuntimeError('Output inventory pagination exceeded 30 pages')
    total=sum(int(r.get('size') or 0) for r in pages)
    disk=shutil.disk_usage('/mnt/d')
    selected=[r for r in pages if PICK_RE.search(str(r.get('name','')))]
    selected_bytes=sum(int(r.get('size') or 0) for r in selected)
    inventory={'utc':datetime.now(timezone.utc).isoformat(),'kernel':KERNEL,'pages':len(list(MON.glob('files_page_*.txt'))),'files':len(pages),'total_reported_bytes':total,'selected_files':len(selected),'selected_reported_bytes':selected_bytes,'disk_free_before_bytes':disk.free,'disk_free_after_preserve_bytes':max(0,disk.free-1_000_000_000),'download_cap_bytes':min(4_000_000_000,max(0,disk.free-1_000_000_000)),'selected_names':[r.get('name') for r in selected],'files_listing':pages}
    (MON/'output_inventory.json').write_text(json.dumps(inventory,indent=2))
    cap=inventory['download_cap_bytes']
    if selected_bytes>cap:
        inventory['retrieval']='SKIPPED: selected outputs exceed conservative free-space cap'
        (MON/'output_inventory.json').write_text(json.dumps(inventory,indent=2));return inventory
    if not selected:
        inventory['retrieval']='SKIPPED: no selected compact outputs listed'
        (MON/'output_inventory.json').write_text(json.dumps(inventory,indent=2));return inventory
    RETRIEVED.mkdir(parents=True,exist_ok=False)
    pattern=PICK_RE.pattern
    p=run(['kernels','output',KERNEL,'--path',str(RETRIEVED),'--file-pattern',pattern,'--page-size','200'],timeout=600)
    (MON/'retrieval_command.txt').write_text(KAGGLE+' kernels output '+KERNEL+' --path '+str(RETRIEVED)+' --file-pattern '+pattern+' --page-size 200\n')
    (MON/'retrieval_stdout.txt').write_text(p.stdout+'\nSTDERR\n'+p.stderr)
    actual=[]
    for f in RETRIEVED.rglob('*'):
        if f.is_file():actual.append({'path':str(f.relative_to(RETRIEVED)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    actual_bytes=sum(x['bytes'] for x in actual)
    inventory.update(retrieval='COMPLETE' if p.returncode==0 else 'FAILED',retrieval_exit=p.returncode,retrieved_files=actual,retrieved_bytes=actual_bytes,disk_free_after=shutil.disk_usage('/mnt/d').free)
    (MON/'output_inventory.json').write_text(json.dumps(inventory,indent=2))
    return inventory

append(status_path, {'utc':start_utc,'status':'MONITOR_STARTED','max_seconds':MAX_SECONDS})
while time.monotonic() < deadline:
    try:
        p=run(['kernels','status',KERNEL],timeout=90)
        status=parse_status(p.stdout+' '+p.stderr)
        if p.returncode:
            status='STATUS_ERROR'
        record={'utc':datetime.now(timezone.utc).isoformat(),'status':status,'exit':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
        if status!=last_status:
            append(status_path,record);print('STATUS',record['utc'],status,flush=True);last_status=status
        # Poll logs while active; persist only on a changed snapshot.
        if status not in TERMINAL and status!='STATUS_ERROR':
            lp=run(['kernels','logs',KERNEL],timeout=90)
            if lp.returncode==0:
                payload=lp.stdout+'\nSTDERR\n'+lp.stderr
                h=hashlib.sha256(payload.encode()).hexdigest()
                if payload.strip() and h!=last_log_hash:
                    (MON/'latest_remote_logs.txt').write_text(payload)
                    append(MON/'log_changes.jsonl',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':h,'bytes':len(payload),'tail':payload.splitlines()[-20:]})
                    print('REMOTE_LOG_CHANGE',h,flush=True);last_log_hash=h
            # Quota sampling every five minutes.
            if time.monotonic()-last_quota>=300:
                q=get_quota();last_quota=time.monotonic()
                if q.get('gpu_used_hours',0)>6.0:
                    append(status_path,{'utc':datetime.now(timezone.utc).isoformat(),'status':'GPU_CEILING_EXCEEDED','quota':q})
                    raise RuntimeError('Observed Kaggle GPU usage exceeds six-hour program ceiling')
        if status in TERMINAL:
            append(status_path,{'utc':datetime.now(timezone.utc).isoformat(),'status':'TERMINAL','terminal_state':status})
            if p.returncode==0:
                quota=get_quota()
                inv=inventory_and_retrieve()
                print('TERMINAL',status,'GPU_USED_HOURS',quota.get('gpu_used_hours'),'OUTPUT_FILES',inv.get('files'),'RETRIEVAL',inv.get('retrieval'),flush=True)
            break
    except Exception as e:
        append(MON/'monitor_errors.jsonl',{'utc':datetime.now(timezone.utc).isoformat(),'error':type(e).__name__+': '+str(e)})
        print('MONITOR_ERROR',type(e).__name__,str(e),flush=True)
        break
    time.sleep(POLL_SECONDS)
else:
    append(status_path,{'utc':datetime.now(timezone.utc).isoformat(),'status':'MONITOR_DEADLINE_REACHED','max_seconds':MAX_SECONDS})
    print('MONITOR_DEADLINE_REACHED; remote job left untouched',flush=True)
