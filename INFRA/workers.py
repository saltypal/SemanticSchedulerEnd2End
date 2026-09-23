"""Run benchmarks in configured incompatible Python environments, never shell strings."""
import json
from pathlib import Path
import subprocess
from INFRA.dependencies import verify_revision
from INFRA.runs import write_json

def run_benchmark_worker(config,run):
    dependency = config.get('dependencies',{}).get(config['implementation'],{})
    required = ('python','source_root','revision','factory')
    missing = [key for key in required if not dependency.get(key)]
    if missing:
        raise ValueError(f'Worker configuration is missing: {missing}')
    revision = verify_revision(dependency['source_root'],dependency['revision'])
    request = {'config':config,'run_id':run.id,'output':str(run.path),'factory':dependency['factory'],
               'source_root':dependency['source_root'],'revision':revision}
    request_path = run.path/'worker_request.json'
    write_json(request_path,request)
    entry = Path(__file__).resolve().parents[1]/'EncDec/worker.py'
    with (run.path/'worker.log').open('w',encoding='utf-8') as log:
        subprocess.run([dependency['python'],str(entry),'--request',str(request_path)],cwd=entry.parents[1],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=dependency.get('timeout_seconds',3600))
    return json.loads((run.path/'worker_result.json').read_text(encoding='utf-8'))
