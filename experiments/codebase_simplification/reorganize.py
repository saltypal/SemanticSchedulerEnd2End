"""Preserve old owners and reorganize sources; never execute project code."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'experiments/codebase_simplification/runs/20261027_layout'
ARCHIVE = ROOT / 'Archived/owners_before_simplification_20261027'
RUN.mkdir(parents=True, exist_ok=False)
ARCHIVE.mkdir(parents=True, exist_ok=False)
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'tests_run': False, 'inference_run': False, 'files': [], 'rewritten': []}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def persist():
    (RUN / 'migration.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

# Archive the exact original owners, including their notebooks and tests.
for name in ('INFRA', 'EncDec', 'SemParam'):
    source = ROOT / name
    target = ARCHIVE / name
    for file in source.rglob('*'):
        if file.is_file():
            manifest['files'].append({'original': str(file.relative_to(ROOT)),
                'archive': str((target / file.relative_to(source)).relative_to(ROOT)),
                'sha256': sha(file), 'bytes': file.stat().st_size})
    shutil.move(str(source), str(target))
    persist()

# Preserve exact editable originals before touching top-level integration files.
for name in ('pyproject.toml', 'setup.py', 'README.md', 'PROJECT_MAP.md', 'AGENTS.md', 'E2EMain.py'):
    shutil.copy2(ROOT / name, ARCHIVE / name)

# Reuse the existing implementations in the new owners. No models are rerun.
ignore = shutil.ignore_patterns('__pycache__', '*.pyc', 'tests', 'notebooks')
for old, new in (('EncDec', 'Codec'), ('INFRA', 'Evaluation')):
    shutil.copytree(ARCHIVE / old, ROOT / new, ignore=ignore)
shutil.move(str(ROOT / 'BetterScheduler'), str(ROOT / 'parameters'))
shutil.move(str(ROOT / 'Evaluation/learning'), str(ROOT / 'Scheduler/training'))

# The old src/Evaluation compatibility package conflicts with the new owner.
shutil.move(str(ROOT / 'src/Evaluation'), str(ARCHIVE / 'src_Evaluation_compatibility'))

replacements = ((r'\bINFRA\.learning\b', 'Scheduler.training'),
                (r'\bEvaluation\.learning\b', 'Scheduler.training'),
                (r'\bINFRA\b', 'Evaluation'), (r'\bEncDec\b', 'Codec'),
                (r'\bBetterScheduler\b', 'parameters'),
                (r'\bSemParam\b', 'Archived.owners_before_simplification_20261027.SemParam'))
excluded = {'results', 'staging', 'notebooks', 'historical', '__pycache__', 'docs', 'handoff'}
for folder in ('Codec', 'Evaluation', 'parameters', 'Scheduler', 'scripts', 'src', 'semcom_benchmark'):
    for file in (ROOT / folder).rglob('*'):
        if not file.is_file() or file.suffix not in {'.py', '.ps1'}:
            continue
        if any(p in excluded for p in file.relative_to(ROOT / folder).parts[:-1]):
            continue
        original = file.read_text(encoding='utf-8-sig')
        updated = original
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, updated)
        if updated != original:
            # Store the exact pre-edit source for reversibility.
            backup = ARCHIVE / 'pre_edit' / file.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, backup)
            file.write_text(updated, encoding='utf-8')
            manifest['rewritten'].append(str(file.relative_to(ROOT)))

entry = ROOT / 'E2EMain.py'
text = entry.read_text(encoding='utf-8')
for pattern, replacement in replacements:
    text = re.sub(pattern, replacement, text)
entry.write_text(text, encoding='utf-8')

# Model hashes remain untouched; only its relocated local filename changes.
release = ROOT / 'parameters/config/parameter_release_v2.json'
config = json.loads(release.read_text(encoding='utf-8'))
config['spi']['model_path'] = config['spi']['model_path'].replace('BetterScheduler', 'parameters')
release.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

# Make archived imports resolvable for retained source and model references.
(ROOT / 'Archived/__init__.py').write_text('"""Preserved research; no active pipeline entry point."""\n', encoding='utf-8')
(ARCHIVE / '__init__.py').write_text('"""Exact old owner snapshots from the layout migration."""\n', encoding='utf-8')
manifest['status'] = 'files_reorganized_no_runtime_validation'
persist()
print(json.dumps({'status': manifest['status'], 'archived_files': len(manifest['files']),
                  'rewritten_sources': len(manifest['rewritten']),
                  'manifest': str(RUN / 'migration.json')}))
