"""Restore the small published pilot fixture for a teammate; never loads models."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from .artifacts import write_json

COMPATIBLE_OUTPUT = ('results/ops/20261025T2256Z/semantic_validation_h3/monitor/'
                     'retrieved/results/semantic_validation/20261026h3')


def restore_fixture(output=None, *, download_images=False, image_source=None):
    """Copy hash-checked metadata and optionally restore 12 public image files.

    Missing images must be explicitly sourced. Original metadata bytes remain
    untouched to preserve the frozen upstream identities used by integration tests.
    Existing output directories are refused, never reset or merged.
    """
    if bool(download_images) == bool(image_source):
        raise ValueError('choose exactly one: public image download or existing image source')
    package = Path(__file__).resolve().parent
    source = package/'handoff/evidence/h3'
    fixture = json.loads((source/'FIXTURE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in fixture['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('fixture checksum mismatch: '+name)
    destination = Path(output).resolve() if output else package/COMPATIBLE_OUTPUT
    if destination.exists():
        raise FileExistsError('fixture output exists; verify it or use a fresh --output, never overwrite')
    destination.mkdir(parents=True)
    write_json(destination/'RESTORE_STATUS.json', {'status': 'incomplete', 'scope': 'restored h3 subset, not a new run'})
    try:
        for name in fixture['files']:
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
        records = json.loads((source/'data/records.json').read_text(encoding='utf-8'))
        if download_images:
            from .semantic_validation_run import materialize_panel
            # Bounded official metadata refresh/download path; no encoder inference.
            materialize_panel({'panel_records': records}, destination/'public_image_download')
            images = destination/'public_image_download/images'
        else:
            images = Path(image_source).resolve()
            if (images/'data/images').is_dir(): images = images/'data/images'
        target_images = destination/'data/images'
        target_images.mkdir(parents=True, exist_ok=True)
        for row in records:
            name = row['image_id']+'.jpg'
            content = (images/name).read_bytes()
            if hashlib.sha256(content).hexdigest() != row['image_id']:
                raise ValueError('pilot image checksum mismatch: '+name)
            (target_images/name).write_bytes(content)
        receipt = dict(status='complete', images=len(records), metadata_files=len(fixture['files']),
            image_bytes=sum((target_images/(r['image_id']+'.jpg')).stat().st_size for r in records),
            public_images_downloaded=download_images, model_inference=False,
            scope='restored consumed h3 fixture subset; not a rerun or complete original run output')
        write_json(destination/'RESTORE_STATUS.json', receipt)
        return receipt
    except Exception as error:
        write_json(destination/'RESTORE_STATUS.json', dict(status='failed', error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--download-images', action='store_true')
    group.add_argument('--image-source')
    args = parser.parse_args()
    print(json.dumps(restore_fixture(args.output, download_images=args.download_images,
                                     image_source=args.image_source), indent=2))
