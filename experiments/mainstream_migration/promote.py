"""One-time source promotion. Never touches historical run outputs."""
from pathlib import Path
import hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/mainstream_migration/runs/promotion-20261105'
RUN.mkdir(parents=True,exist_ok=True)
groups={'experiments/stream_scheduler':'Scheduler/training/stream_suite',
        'experiments/master_study':'Scheduler/training/study'}
mapping={str(p.relative_to(ROOT)).replace('\\','/'):f'{dest}/{p.name}'
         for old,dest in groups.items() for p in (ROOT/old).glob('*.py')}
manifest=[]
for old,new in mapping.items():
 source=ROOT/old;target=ROOT/new
 if target.exists():raise RuntimeError(f'Refusing to overwrite {target}')
 text=source.read_text(encoding='utf-8-sig')
 for a,b in groups.items():text=text.replace(a.replace('/','.'),b.replace('/','.'))
 text=text.replace('Path(__file__).resolve().parents[2]','Path(__file__).resolve().parents[3]')
 for a,b in mapping.items():text=text.replace(a,b)
 text=text.replace('ROOT/"experiments/stream_scheduler"/script','ROOT/"Scheduler/training/stream_suite"/script')
 text=text.replace('SCRIPTS=ROOT/"experiments/stream_scheduler"','SCRIPTS=ROOT/"Scheduler/training/stream_suite"')
 if source.name=='build_paired_notebooks.py':
  text=text.replace('DEST=ROOT/"experiments/master_study/paired_kaggle"','DEST=ROOT/"Notebooks/SchedulerSuite/paired"')
  text=text.replace('CACHE=json.loads((DEST/"inputs/attachment.json").read_text())','CACHE=json.loads((ROOT/"experiments/master_study/paired_kaggle/inputs/attachment.json").read_text())')
  text=text.replace('    out=io.BytesIO();hashes={}', '    names.update(str(p.relative_to(ROOT)).replace("\\\\", "/") for owner in ("stream_suite", "study") for p in (ROOT/"Scheduler/training"/owner).glob("*.py"))\n    out=io.BytesIO();hashes={}')
 if source.name=='build_teacher_recovery_notebook.py':
  text=text.replace('DEST=ROOT/"experiments/master_study/gap_recovery"','DEST=ROOT/"Notebooks/SchedulerSuite/recovery"')
  text=text.replace('DEST.mkdir(exist_ok=True)','DEST.mkdir(parents=True,exist_ok=True)')
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8')
 manifest.append({'old':old,'new':new,'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'promoted_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
 module=new[:-3].replace('/','.')
 # Direct execution preserves __main__; imports share the actual module object.
 wrapper=f'''"""Compatibility entry point. Implementation: {new}."""
from pathlib import Path
import importlib, runpy, sys
_root=Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:sys.path.insert(0,str(_root))
if __name__ == '__main__':
    runpy.run_path(str(_root / {new!r}), run_name='__main__')
else:
    sys.modules[__name__]=importlib.import_module({module!r})
'''
 source.write_text(wrapper,encoding='utf-8')
for dest in groups.values():(ROOT/dest/'__init__.py').write_text('"""Canonical scheduler suite implementation. Importing does not start training."""\n')
# Import shared modules through the canonical owner in live source and tests.
for folder in ['Scheduler','Evaluation','Test','experiments/stream_scheduler/tests','experiments/master_study/tests']:
 for p in (ROOT/folder).rglob('*.py'):
  text=p.read_text(encoding='utf-8-sig');updated=text
  for a,b in groups.items():updated=updated.replace(a.replace('/','.'),b.replace('/','.'))
  if updated!=text:p.write_text(updated,encoding='utf-8')
(RUN/'migration.json').write_text(json.dumps({'mapping':manifest,'preserved':'All runs, checkpoints, old notebooks and external account identities remain in place'},indent=2))
print('Promoted',len(manifest),'source modules')
