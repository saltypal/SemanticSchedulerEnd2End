"""Record the exact local handoff checks using the active project interpreter."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'BetterScheduler'
OUTPUT = Path(__file__).resolve().parent/'validation'
OUTPUT.mkdir(exist_ok=True)


def source_hashes():
    result = {}
    for path in PACKAGE.rglob('*'):
        relative = path.relative_to(PACKAGE)
        if (path.is_file() and path.suffix in ('.py','.json')
                and path.name not in ('VALIDATION.json','PUBLICATION_SCAN.json')
                and not set(relative.parts)&{'results','staging','__pycache__','validation','evidence'}):
            result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


started = datetime.now(timezone.utc).isoformat()
before = source_hashes()
environment = {**os.environ,'PYTHONUTF8':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
command = [sys.executable,'-m','pytest','BetterScheduler/tests','-q']
result = subprocess.run(command,cwd=ROOT,env=environment,text=True,encoding='utf-8',errors='replace',
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
(OUTPUT/'pytest.log').write_text(result.stdout,encoding='utf-8',newline='\n')
after = source_hashes()
receipt = dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
    python=sys.executable,python_version=sys.version,command=command,exit_code=result.returncode,
    source_unchanged=before==after,source_hashes=after,
    changed_source_paths=sorted(name for name in set(before)|set(after) if before.get(name)!=after.get(name)),
    base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    test_summary=result.stdout.strip().splitlines()[-1] if result.stdout.strip() else 'no output',
    scope='local CPU tests including schema, routing, policy scoring, package contracts and fixture recovery; no VLM inference or GPU run')
(OUTPUT.parent/'VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(result.stdout)
print('Receipt:',OUTPUT.parent/'VALIDATION.json')
if before!=after: raise SystemExit('Source changed during validation; do not publish this checkpoint.')
raise SystemExit(result.returncode)
