"""Traceable run storage, kept separate from source and historical outputs."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[1]

def provenance():
    def git(*args):
        result = subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','-C',str(ROOT),*args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for name in ('numpy','torch','torchvision','transformers','timm','PyYAML','Pillow','scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    digest = hashlib.sha256()
    for folder in ('EncDec','SemParam','Scheduler','INFRA'):
        for path in sorted((ROOT/folder).rglob('*.py')):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return {'git_commit':git('rev-parse','HEAD'), 'git_status':git('status','--porcelain'),
            'source_sha256':digest.hexdigest(), 'python':sys.version, 'executable':sys.executable,
            'platform':platform.platform(), 'packages':versions}

def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (Path, datetime)):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False), encoding='utf-8')

class Run:
    def __init__(self, config):
        self.id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        self.path = Path(config['paths']['results']) / config['module'] / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config
        (self.path/'config.resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        write_json(self.path/'provenance.json', provenance())
        write_json(self.path/'status.json', {'status':'running','run_id':self.id})
        random.seed(config['seed'])

    def finish(self, result):
        write_json(self.path/'result.json', result)
        write_json(self.path/'status.json', {'status':'completed_with_failures' if result.get('failed',0) else 'completed','run_id':self.id})

    def fail(self, error):
        write_json(self.path/'status.json', {'status':'failed','run_id':self.id,'error':f'{type(error).__name__}: {error}'})

def select_device(requested):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if ':' in requested and int(requested.split(':')[1]) >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device does not exist: {requested}')
    return requested

def seed_models(seed):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
