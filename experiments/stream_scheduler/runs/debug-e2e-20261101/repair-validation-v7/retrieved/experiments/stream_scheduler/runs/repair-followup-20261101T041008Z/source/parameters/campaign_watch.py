"""Bounded unattended stage chaining using saved Kaggle outputs as authority."""
from pathlib import Path
import argparse
import json
import subprocess
import time
from datetime import datetime,timezone


def run(package,kaggle):
    package=Path(package).resolve();config=json.loads((package/'campaign.json').read_text())
    evidence=package/'operations';evidence.mkdir(exist_ok=True)
    def command(args):
        result=subprocess.run([kaggle,*args],capture_output=True,text=True,timeout=180)
        with (evidence/'commands.log').open('a',encoding='utf-8') as log:
            log.write(datetime.now(timezone.utc).isoformat()+' '+repr(args)+'\n'+result.stdout+result.stderr+'\n')
        return result
    def save(value):(evidence/'watch_status.json').write_text(json.dumps(value,indent=2))
    deadline=time.monotonic()+24*3600
    for stage in ('prepare','extract','select'):
        slug='satyapaladugu/'+config['slugs'][stage]
        while time.monotonic()<deadline:
            result=command(['kernels','status',slug]);value=result.stdout+result.stderr
            save({'stage':stage,'slug':slug,'last_status':value,'utc':datetime.now(timezone.utc).isoformat()})
            if 'COMPLETE' in value.upper():break
            if 'ERROR' in value.upper() or 'CANCEL' in value.upper():
                command(['kernels','output',slug,'-p',str(evidence/stage),'--file-pattern',r'.*(status|progress|log|manifest|preflight|REPORT|ipynb).*'])
                raise RuntimeError('Kaggle stage failed: '+stage)
            time.sleep(120)
        else:raise TimeoutError('campaign watcher24-hour cap')
        destination=evidence/stage
        fetched=command(['kernels','output',slug,'-p',str(destination),'--file-pattern',r'.*(status\.json|exposure_audit\.json|producer_profiles\.json|preflight\.json|REPORT\.md|cell_status\.json|executed\.ipynb|artifact_checksums\.json)$'])
        if fetched.returncode:raise RuntimeError('cannot verify saved stage outputs: '+stage)
        statuses=list(destination.glob('**/status.json'))
        if not any(json.loads(path.read_text()).get('status')=='complete' for path in statuses):
            raise RuntimeError('no retrieved complete run receipt: '+stage)
        if stage=='prepare':
            audits=list(destination.glob('**/exposure_audit.json'))
            if not audits:raise RuntimeError('missing persisted exposure audit')
            following='extract'
        elif stage=='extract':following='select'
        else:
            save({'status':'complete','outputs':str(evidence),'utc':datetime.now(timezone.utc).isoformat()});return
        args=['kernels','push','-p',str(package/following),'-t',str(config['time_limits'][following])]
        if following=='extract':args+=['--accelerator','NvidiaTeslaT4']
        pushed=command(args)
        if pushed.returncode:raise RuntimeError('downstream launch failed: '+following)
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True);parser.add_argument('--kaggle',required=True)
    args=parser.parse_args()
    try:run(args.package,args.kaggle)
    except Exception as error:
        path=Path(args.package)/'operations';path.mkdir(exist_ok=True)
        (path/'watch_failure.json').write_text(json.dumps({'error':type(error).__name__,'message':str(error)}));raise
