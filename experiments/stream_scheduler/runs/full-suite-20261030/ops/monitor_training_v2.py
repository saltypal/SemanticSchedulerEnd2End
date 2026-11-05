import json,subprocess,time,datetime
from pathlib import Path
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End')
run=root/'experiments/stream_scheduler/runs/full-suite-20261030'
cli='/home/Satya/.venvs/semantic-kaggle/bin/kaggle'
log=(run/'ops/monitor_training_v2.log').open('a',buffering=1)
start=time.monotonic()
while time.monotonic()-start<11100:
    r=subprocess.run([cli,'kernels','status','satyapaladuguplus/semantic-decoder-scheduler-20261030/2'],capture_output=True,text=True,timeout=45)
    text=r.stdout+r.stderr
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':r.returncode,'status':text.strip()}
    log.write(json.dumps(row)+'\n');(run/'ops/live_status_v2.json').write_text(json.dumps(row,indent=2))
    if r.returncode==0 and ('COMPLETE' in text or 'ERROR' in text or 'CANCEL' in text):
        dest=run/'retrieved_v2';dest.mkdir(exist_ok=True)
        # This kernel outputs only its own logs/source/models; input archive is not copied to outputs.
        with (run/'ops/retrieval_v2.log').open('w') as f:
            result=subprocess.run([cli,'kernels','output','satyapaladuguplus/semantic-decoder-scheduler-20261030/2','-p',str(dest)],stdout=f,stderr=subprocess.STDOUT,timeout=600)
        (run/'ops/retrieval_v2_status.json').write_text(json.dumps({'returncode':result.returncode,'path':str(dest),'verified':False,'scope':'retrieval only; checksums/metrics inspection still required'},indent=2))
        break
    time.sleep(45)
