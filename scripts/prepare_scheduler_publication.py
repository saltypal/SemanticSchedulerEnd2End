"""Inventory and credential-pattern-check the explicit scheduler publication scope."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ['Scheduler', 'Archived.owners_before_simplification_20261027.SemParam', 'Evaluation', 'docs/disaster_decoder_v3',
           'docs/disaster_scheduler', 'kaggle', 'deliverables',
           'results/disaster_decoder_v3', 'results/baseline_scheduling',
           'results/disaster_study_preflight', 'results/ladi_v2_e2e_semantic_channel_v1']
EXTRA = ['.gitattributes', 'PROJECT_MAP.md', 'SCHEDULER_START_HERE.md',
         'Notebooks/03_LADI_v2_Baseline_Scheduling.ipynb',
         'Notebooks/04_LADI_v2_End_to_End_Semantic_Channel_Scheduling.ipynb']
BLOCKED_PARTS = {'__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
MANIFEST = 'docs/disaster_decoder_v3/PUBLICATION_MANIFEST.json'
SECRET_PATTERNS = [rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{30,}',
                   rb'hf_[A-Za-z0-9]{30,}', rb'AIza[0-9A-Za-z_-]{30,}',
                   rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']


def selected_files():
    folders = FOLDERS + [p.relative_to(ROOT).as_posix()
                         for p in (ROOT/'results').glob('disaster_evidence_seven_version*')]
    files = set()
    for name in folders:
        for p in (ROOT/name).rglob('*'):
            if p.is_file():
                files.add(p)
    files.update(ROOT/name for name in EXTRA)
    files.update((ROOT/'scripts').glob('*decoder*.py'))
    files.update((ROOT/'scripts').glob('*disaster*.py'))
    files.update((ROOT/'scripts').glob('*qwen*.py'))
    files.update(ROOT/'scripts'/name for name in
                 ('compare_original_schedulers.py', 'export_scheduler_session.py',
                  'prepare_scheduler_publication.py'))
    return sorted(p for p in files if p.is_file()
        and not BLOCKED_PARTS.intersection(p.parts)
        and p.suffix not in ('.pyc', '.pyo', '.partial', '.tmp')
        and p.name not in ('.env', 'kaggle.json')
        and p.relative_to(ROOT).as_posix() != MANIFEST)


def scan(data, name):
    if any(re.search(pattern, data) for pattern in SECRET_PATTERNS):
        raise ValueError('possible credential in '+name+'; content not printed')


def main(use_index=False):
    files, entries = selected_files(), {}
    for p in files:
        name = p.relative_to(ROOT).as_posix()
        data = p.read_bytes()
        scan(data, name)
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as archive:
                for entry in archive.infolist():
                    if not entry.is_dir():
                        scan(archive.read(entry), name+'::'+entry.filename)
        if len(data) >= 100*1024**2:
            raise ValueError('file exceeds normal GitHub limit; explicit LFS rule required: '+name)
        entries[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if use_index:
        # Hash the exact staged bytes, resolving LFS pointers to their payload OID.
        # Preserve the local-byte hash where Git normalizes text line endings.
        blobs = {}
        for row in subprocess.check_output(['git', 'ls-files', '-s', '-z'], cwd=ROOT).split(b'\0'):
            if row:
                info, name = row.split(b'\t', 1)
                blobs[name.decode('utf-8')] = info.split()[1].decode()
        process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for name, record in entries.items():
                if name not in blobs:
                    raise ValueError('publication file not staged/tracked: '+name)
                process.stdin.write((blobs[name]+'\n').encode()); process.stdin.flush()
                header = process.stdout.readline().split()
                size = int(header[2])
                blob = process.stdout.read(size)
                assert process.stdout.read(1) == b'\n'
                if blob.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                    oid = re.search(rb'oid sha256:([0-9a-f]{64})', blob).group(1).decode()
                    if oid != record['sha256']:
                        raise ValueError('LFS pointer does not match local payload: '+name)
                    record['git_lfs'] = True
                else:
                    digest = hashlib.sha256(blob).hexdigest()
                    if digest != record['sha256']:
                        record['worktree_sha256'] = record['sha256']
                        record['worktree_bytes'] = record['bytes']
                        record['sha256'], record['bytes'] = digest, size
                        record['normalization'] = 'Git text normalization; sha256 describes repository bytes'
                record['git_blob'] = blobs[name]
        finally:
            process.stdin.close(); process.stdout.close(); process.wait()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        branch='new-scheduler', base_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='scheduler modules, historical/current scheduler code, notebooks and all saved scheduler results',
        excluded=['raw dataset downloads', 'codec-only working changes', 'credentials',
                  'private Codex session export', 'temporary transfer fragments', 'Python caches'],
        secret_pattern_scan='passed; heuristic, not a guarantee',
        hash_scope='staged repository bytes / resolved LFS payload' if use_index else 'local worktree bytes',
        file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries.values()), files=entries)
    (ROOT/MANIFEST).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', action='store_true')
    main(parser.parse_args().index)
