"""Bounded subprocess orchestration; source methods remain in reusable workers."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
from .artifacts import Run,write_json
from .semantic_validation_run import materialize_panel,select_review_cohort


def stream_process(command,log_path,seconds,env=None):
    """Persist child output incrementally and enforce a process wall deadline."""
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();position=0;heartbeat=0
    with log_path.open('w',encoding='utf-8') as handle:
        process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
        try:
            while process.poll() is None:
                if time.monotonic()-started>seconds:raise TimeoutError('worker process wall budget')
                with log_path.open(encoding='utf-8',errors='replace') as reader:
                    reader.seek(position);text=reader.read();position=reader.tell()
                if text:print(text,end='',flush=True)
                if time.monotonic()-heartbeat>=30:
                    print('WORKER ACTIVE',log_path.name,'elapsed',round(time.monotonic()-started,1),flush=True)
                    heartbeat=time.monotonic()
                time.sleep(2)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    with log_path.open(encoding='utf-8',errors='replace') as reader:
        reader.seek(position);print(reader.read(),end='',flush=True)
    return {'returncode':process.returncode,'seconds':time.monotonic()-started,'log':str(log_path)}


def run(config,output,server_seconds=None):
    import torch
    if torch.cuda.device_count()!=2:raise RuntimeError('two assigned GPUs required')
    spec=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    limits=spec['limits']
    server_seconds=limits['server_seconds'] if server_seconds is None else server_seconds
    if not limits['internal_seconds']<server_seconds<=limits['server_seconds']:
        raise ValueError('declared server cap must cover internal work and respect the specification maximum')
    run=Run(output,{'seconds':limits['internal_seconds'],'declared_server_seconds':server_seconds,'specification':spec})
    deadline=run.started+limits['internal_seconds'];outcomes={}
    environment={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'PYTHONUTF8':'1',
                 'HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'}
    def remaining(cap):
        seconds=min(cap,int(deadline-time.monotonic()))
        if seconds<30:raise TimeoutError('campaign time budget exhausted')
        return seconds
    try:
        with run.stage('00_new_test_source_panel'):
            records=materialize_panel(spec,run.path/'data')
            write_json(run.path/'hardware.json',{'gpu_names':[torch.cuda.get_device_name(i) for i in (0,1)],
                'torch':torch.__version__,'cuda':torch.version.cuda,'enable_thinking':False})
        with run.stage('01_parameter_extraction'):
            command=[sys.executable,'-u','-m','parameters.semantic_validation_run','--stage','extract',
                     '--root',str(run.path),'--config',str(config)]
            outcomes['extraction']=stream_process(command,run.path/'logs/extraction_worker.log',remaining(240),environment)
            if outcomes['extraction']['returncode']:raise RuntimeError('parameter worker failed; model stages not started')
            print('REVIEW COHORT',select_review_cohort(run.path,spec['review_parents_per_stratum']),flush=True)
        with run.stage('02_isolated_9b_environment'):
            # The environment is disposable; do not publish thousands of package
            # files as notebook outputs or crowd out the actual run artifacts.
            target=Path('/kaggle/temp/qwen35_env')
            if target.exists():raise ValueError('isolated 9B environment path already exists; no overwrite')
            creation=stream_process([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(target)],
                run.path/'logs/qwen35_venv.log',remaining(120),environment)
            if creation['returncode']:raise RuntimeError('isolated environment creation failed')
            interpreter=target/'bin/python'
            install=stream_process([sys.executable,'-m','pip','--python',str(interpreter),'install','--disable-pip-version-check',
                'transformers==5.3.0','accelerate==1.12.0','bitsandbytes==0.49.2','huggingface-hub==1.5.0'],
                run.path/'logs/qwen35_dependencies.log',remaining(180),environment)
            if install['returncode']:raise RuntimeError('9B dependency installation failed')
            check="import importlib.util,torch,transformers; assert transformers.__version__=='5.3.0'; assert not any(importlib.util.find_spec(k) is not None for k in ('fla','causal_conv1d')); from transformers import Qwen3_5ForConditionalGeneration; import bitsandbytes; print('9B runtime imports verified',transformers.__version__,torch.__version__,bitsandbytes.__version__,flush=True)"
            preflight=stream_process([str(interpreter),'-u','-c',check],run.path/'logs/qwen35_import_preflight.log',remaining(90),environment)
            if preflight['returncode']:raise RuntimeError('9B runtime import preflight failed before reviewer weight downloads')
        with run.stage('03_baseline_4b'):
            seconds=remaining(350)
            command=[sys.executable,'-u','-m','parameters.semantic_validation_run','--stage','review',
                     '--root',str(run.path),'--config',str(config),'--model','qwen4b','--seconds',str(seconds-10)]
            outcomes['qwen4b']=stream_process(command,run.path/'logs/qwen4b_worker.log',seconds,environment)
        with run.stage('04_challenger_9b'):
            seconds=remaining(950)
            command=[str(interpreter),'-u','-m','parameters.semantic_validation_run','--stage','review',
                     '--root',str(run.path),'--config',str(config),'--model','qwen9b','--seconds',str(seconds-10)]
            outcomes['qwen9b']=stream_process(command,run.path/'logs/qwen9b_worker.log',seconds,environment)
        statuses={}
        for key in ('qwen4b','qwen9b'):
            path=run.path/key/'status.json'
            statuses[key]=json.loads(path.read_text()) if path.exists() else {'status':'missing_worker_status'}
        state='complete' if all(row['status']=='complete' for row in statuses.values()) else 'partial'
        write_json(run.path/'status.json',{'status':state,'model_status':statuses,'processes':outcomes,
            'seconds':time.monotonic()-run.started,'no_policy_training':True,'enable_thinking':False})
    except Exception as error:
        traceback.print_exc()
        write_json(run.path/'status.json',{'status':'failed','error':type(error).__name__,'message':str(error),
            'processes':outcomes,'seconds':time.monotonic()-run.started,'enable_thinking':False})
    finally:
        import hashlib
        checks={str(p.relative_to(run.path)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in run.path.rglob('*') if p.is_file() and p.name!='checksums.json'}
        write_json(run.path/'checksums.json',checks)
    return run.path
