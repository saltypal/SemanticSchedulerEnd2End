"""Run identity, incremental stage logs, budgets, and atomic JSON persistence."""
from pathlib import Path
from contextlib import contextmanager,redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


class Tee:
    def __init__(self,terminal,file):self.terminal,self.file=terminal,file
    def write(self,value):
        # Third-party loggers can retain a stream installed during a stage
        # after that stage closes its file. Keep forwarding to the live
        # terminal, but do not let a stale file handle break model loading.
        self.terminal.write(value)
        if not self.file.closed:
            self.file.write(value)
            self.file.flush()
        return len(value)
    def flush(self):
        self.terminal.flush()
        if not self.file.closed:self.file.flush()


class Run:
    def __init__(self,path,config):
        self.path=Path(path).resolve();self.started=time.monotonic();self.config=config
        self.path.mkdir(parents=True,exist_ok=False)
        for name in ('logs','metrics','figures','models','data','features'): (self.path/name).mkdir()
        root=Path(__file__).resolve().parents[1]
        def git(*args):
            try:return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            except (subprocess.CalledProcessError,FileNotFoundError):return None
        files=list((root/'parameters').glob('*.py'))
        self.manifest={'run_id':self.path.name,'utc':datetime.now(timezone.utc).isoformat(),'root':str(root),
                       'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),
                       'git_commit':git('rev-parse','HEAD'),'dirty_state':git('status','--porcelain'),
                       'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                       'packages':{},'scope':'parameter research; complete messages; no radio allocation'}
        snapshot=os.environ.get('BETTER_SOURCE_MANIFEST')
        if snapshot and Path(snapshot).is_file():
            self.manifest['uploaded_source_snapshot']=json.loads(Path(snapshot).read_text())
        for package in ('numpy','torch','transformers','scikit-learn','Pillow','nbformat','nbclient','psutil'):
            try:self.manifest['packages'][package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:self.manifest['packages'][package]=None
        write_json(self.path/'manifest.json',self.manifest);write_json(self.path/'config.json',config)
        self.stages=[]

    def budget(self):
        if time.monotonic()-self.started>self.config.get('seconds',3600):raise TimeoutError('declared wall-compute budget reached')

    @contextmanager
    def stage(self,name):
        self.budget();began=time.monotonic();status='complete'
        with (self.path/'logs'/(name+'.log')).open('w',encoding='utf-8') as handle:
            with redirect_stdout(Tee(sys.stdout,handle)),redirect_stderr(Tee(sys.stderr,handle)):
                print('START',name,'output',self.path,flush=True)
                try:yield
                except Exception:
                    status='failed';traceback.print_exc();raise
                finally:
                    row={'stage':name,'status':status,'seconds':time.monotonic()-began}
                    self.stages.append(row);write_json(self.path/'progress.json',self.stages)
                    print('END',name,row,flush=True)
