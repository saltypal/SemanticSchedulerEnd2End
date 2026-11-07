"""Package only the additive study and its verified artifacts; never legacy work."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.build_decoder_v3_notebook import SOURCES
TESTS = ['Archived/owners_before_simplification_20261027/SemParam/tests/test_decoder_parameters.py','Archived/owners_before_simplification_20261027/SemParam/tests/test_decoder_costs.py',
         'Scheduler/tests/test_decoder_scheduler.py','Evaluation/tests/test_decoder_v3_extract.py',
         'Evaluation/tests/test_decoder_v3_study.py','Evaluation/tests/test_decoder_queue.py',
         'Evaluation/tests/test_decoder_v3_notebook.py','Evaluation/tests/test_decoder_v3_analysis.py']
RUNS = ['20261022_stageA_v3','20261023_stageB_v1/full','20261023_stageC_v2',
        '20261023_analysis_v1','20261023_analysis_v2']


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('handoff archives are immutable; choose a new filename')
    output.parent.mkdir(parents=True,exist_ok=True)
    canonical = ROOT/'Scheduler/notebooks/05_disaster_decoder_v3.ipynb'
    notebook = json.loads(canonical.read_text(encoding='utf-8'))
    code = [cell for cell in notebook['cells'] if cell['cell_type']=='code']
    if len(code)!=3 or any(cell.get('execution_count') is None for cell in code):
        raise ValueError('actual executed final notebook required')
    if any(item['output_type']=='error' for cell in code for item in cell.get('outputs',[])):
        raise ValueError('notebook contains an execution error')
    entries = {}
    def add(path,name=None):
        if not path.is_file() or path.suffix in ('.partial','.pyc','.tmp') or '__pycache__' in path.parts:
            return
        archive_name = name or path.relative_to(ROOT).as_posix()
        entries[archive_name] = path.read_bytes()
    for name in SOURCES+TESTS+['scripts/build_decoder_v3_notebook.py','scripts/build_decoder_v3_handoff.py',
                              'Scheduler/notebooks/05_disaster_decoder_v3.ipynb']:
        add(ROOT/name)
    for folder in (ROOT/'docs/disaster_decoder_v3',ROOT/'kaggle/disaster-decoder-v3-final'):
        for path in folder.rglob('*'): add(path)
    for name in RUNS:
        folder = ROOT/'results/disaster_decoder_v3'/name
        if not folder.is_dir(): raise FileNotFoundError(folder)
        for path in folder.rglob('*'): add(path)
    for name in ('Archived.owners_before_simplification_20261027.SemParam','Scheduler','Evaluation','Scheduler/training','Evaluation/tests','Archived/owners_before_simplification_20261027/SemParam/tests','Scheduler/tests','scripts'):
        entries[name+'/__init__.py'] = b'"""Isolated additive disaster decoder study package."""\n'
    entries['README_FIRST.md'] = (ROOT/'docs/disaster_decoder_v3/HANDOFF.md').read_bytes()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='additive prototype, executed notebook, frozen features, trained artifacts and honest evaluation',
        credentials_included=False,raw_source_images_included=False,
        legacy_work_modified_or_removed=False,
        files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(entries.items())})
    entries['HANDOFF_MANIFEST.json'] = json.dumps(manifest,indent=2).encode()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(entries.items()): archive.writestr(name,data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None: raise RuntimeError('handoff archive failed CRC verification')
        for name,record in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=record['sha256']:
                raise RuntimeError('handoff entry hash mismatch: '+name)
    summary = dict(path=str(output),bytes=output.stat().st_size,files=len(entries),
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),verified=True)
    output.with_suffix('.audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    build(parser.parse_args().output)
