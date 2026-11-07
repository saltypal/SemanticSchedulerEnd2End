"""Frozen CPU development comparison for corrective protocol m. Pilot remains closed."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def snapshot(output,upstream):
    source=output/'source';source.mkdir();hashes={}
    for folder in ('Evaluation','Scheduler','Archived.owners_before_simplification_20261027.SemParam'):
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or 'tests' in path.parts:continue
            if path.suffix not in ('.py','.json'):continue
            relative=path.relative_to(ROOT);target=source/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            hashes[relative.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    frozen=json.loads((upstream/'upstream_manifest.json').read_text())
    for relative,expected in frozen['files'].items():
        path=(upstream/relative).resolve()
        if not path.is_relative_to(upstream.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('upstream snapshot identity mismatch')
        target=source/'upstream'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes['upstream/'+relative]=expected
    shutil.copy2(upstream/'upstream_manifest.json',source/'upstream/upstream_manifest.json')
    hashes['upstream/upstream_manifest.json']=hashlib.sha256((source/'upstream/upstream_manifest.json').read_bytes()).hexdigest()
    launcher=source/'scripts'/Path(__file__).name;launcher.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,launcher)
    hashes[launcher.relative_to(source).as_posix()]=hashlib.sha256(launcher.read_bytes()).hexdigest()
    return source,hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--train-traces',type=int,default=32);p.add_argument('--eval-traces',type=int,default=32)
    p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--algorithms',nargs='+',default=['ranknet','lambdamart','priority_classifier','masked_ppo','masked_double_dqn'])
    p.add_argument('--q-mode',choices=['bounded_nonnegative_v1','unbounded'],default='bounded_nonnegative_v1')
    a=p.parse_args();a.output=a.output.resolve();a.data=a.data.resolve();a.upstream=a.upstream.resolve()
    a.output.mkdir(parents=True,exist_ok=False);begun=time.perf_counter()
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    print('START corrective CPU development comparison',a.output,flush=True)
    source,hashes=snapshot(a.output,a.upstream)
    sys.path.insert(0,str(source))
    from Scheduler.training.scene_study import run_study
    import Scheduler.training.scene_study as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(source):raise RuntimeError('mutable source imported')
    import torch
    torch.set_num_threads(1)
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scikit-learn','lightgbm','jsonschema','Pillow')}
    write(a.output/'environment.json',{'python':sys.executable,'version':platform.python_version(),
        'packages':versions,'used_device':'cpu','platform':platform.platform(),'processor':platform.processor(),
        'cuda_available_but_unused':torch.cuda.is_available()})
    (a.output/'requirements-observed.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n')
    manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'dirty':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'source_files':hashes,'data_files':{name:hashlib.sha256((a.data/name).read_bytes()).hexdigest()
            for name in ('records.json','features.jsonl')},'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'scope':'One-seed development feasibility, opaque payloads, synthetic service time, no pilot or teacher inference'}
    write(a.output/'manifest.json',manifest)
    status={'status':'running','pilot_opened':False};write(a.output/'status.json',status)
    try:
        rows=run_study(a.data,source/'upstream',a.output/'study',epochs=a.epochs,steps=a.steps,
            train_traces=a.train_traces,eval_traces=a.eval_traces,algorithms=tuple(a.algorithms),
            seconds=max(1,a.seconds-(time.perf_counter()-begun)),q_value_mode=a.q_mode,
            collection_policy='random_damage_v1',training_stream_mode='fresh',rl_patience=3)
        statuses=[r['status'] for r in rows]
        status['status']='failed' if 'failed' in statuses else ('partial' if 'partial' in statuses else 'complete')
        status['methods']={r['algorithm']:r['status'] for r in rows}
    except Exception as error:
        status.update(status='failed',error=type(error).__name__,reason=str(error));raise
    finally:
        status['seconds']=time.perf_counter()-begun;write(a.output/'status.json',status)
        checks={path.relative_to(a.output).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in a.output.rglob('*') if path.is_file() and path.name!='artifact_checksums.json'}
        write(a.output/'artifact_checksums.json',checks)
        print('END corrective comparison',status,flush=True)


if __name__=='__main__':main()
