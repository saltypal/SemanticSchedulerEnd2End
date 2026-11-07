"""Package changed source/docs only, excluding data, weights, credentials/results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(['git', '-C', str(ROOT), *arguments], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT/'results/decoder_context/delivery'/stamp
    if output.exists() and any(output.iterdir()):
        raise ValueError('delivery output must be new/empty')
    output.mkdir(parents=True, exist_ok=True)
    modified = set(git('diff', '--name-only', 'HEAD').splitlines())
    added = set(git('ls-files', '--others', '--exclude-standard').splitlines())
    permitted = {'Evaluation', 'Scheduler', 'Archived.owners_before_simplification_20261027.SemParam', 'Codec', 'scripts', 'docs', 'environment'}
    sources = sorted(name for name in modified | added
                     if (Path(name).parts[0] in permitted or name == 'PROJECT_MAP.md')
                     and Path(name).suffix.lower() in ('.py', '.md', '.json', '.txt', '.yaml')
                     and (ROOT/name).is_file())
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    manifest = dict(created_utc=stamp, base_revision=git('rev-parse', 'HEAD').strip(),
                    dirty_state=git('status', '--porcelain'), files=hashes,
                    source_root=str(ROOT), tracked_patch='baseline.patch',
                    datasets_and_weights_included=False, checkpoints_committed=False,
                    scope='local_reviewable_source_snapshot_not_a_journal_result')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    patch = git('diff', '--binary', 'HEAD')
    (output/'baseline.patch').write_text(patch, encoding='utf-8', newline='\n')
    archive = output/'decoder_context_source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
        for name in sources:
            handle.write(ROOT/name, name)
        handle.write(output/'manifest.json', 'DELIVERY_MANIFEST.json')
        handle.write(output/'baseline.patch', 'baseline.patch')
    with zipfile.ZipFile(archive) as handle:
        for name, expected in hashes.items():
            if hashlib.sha256(handle.read(name)).hexdigest() != expected:
                raise RuntimeError(f'delivery archive hash mismatch: {name}')
    print(json.dumps(dict(output=str(output.resolve()), files=len(sources),
                          archive_bytes=archive.stat().st_size,
                          archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          archive_verified=True), indent=2))


if __name__ == '__main__':
    main()
