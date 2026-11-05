"""Review a checksum-pinned parameter cache without repeating visual extraction."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from .artifacts import Run, write_json
from .semantic_validation_driver import stream_process


def copy_review_inputs(source, destination, expected):
    source, destination = Path(source), Path(destination)
    for name, digest in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError('upstream identity mismatch: ' + name)
    checks = json.loads((source / 'checksums.json').read_text())
    records = json.loads((source / 'data/records.json').read_text())
    names = ['data/records.json', 'data/review_cohort.json',
             'data/controls_evaluator_only.json', 'data/actual_evaluator_only.json',
             'data/full_controls_evaluator_only.json', 'data/full_actual_evaluator_only.json',
             'extraction/status.json', 'extraction/trace.json']
    names += ['data/images/' + row['image_id'] + '.jpg' for row in records]
    verified = {}
    for name in names:
        path = source / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if checks.get(name) != digest:
            raise ValueError('upstream artifact mismatch: ' + name)
        target = destination / name
        if target.exists():
            raise FileExistsError('refuse to replace existing review input: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        verified[name] = digest
    images = {row['image_id']: str((destination / 'data/images' / (row['image_id'] + '.jpg')).resolve())
              for row in records}
    for identity, path in images.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != identity:
            raise ValueError('image identity mismatch')
    # Change filesystem routing only. Candidate predictions/reference facts remain exact.
    for name in ('records', 'controls_evaluator_only', 'actual_evaluator_only',
                 'full_controls_evaluator_only', 'full_actual_evaluator_only'):
        path = destination / 'data' / (name + '.json')
        rows = json.loads(path.read_text())
        for row in rows:
            row['image_path'] = images[row['image_id']]
        write_json(path, rows)
    write_json(destination / 'upstream_receipt.json', {
        'source': str(source), 'pinned_identity': expected, 'verified_artifacts': verified,
        'extraction_reused': True, 'routing_paths_rewritten': True,
        'candidate_values_modified': False, 'native_annotations_modified': False})


def run(config, source, output):
    import torch
    spec = json.loads(Path(config).read_text())
    if torch.cuda.device_count() != 2:
        raise RuntimeError('two assigned T4 GPUs required; reviewer uses GPU1')
    limits = spec['limits']
    experiment = Run(output, {'seconds': limits['internal_seconds'], 'specification': spec})
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
                   'PYTHONUTF8': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'HF_HUB_DISABLE_PROGRESS_BARS': '1'}
    try:
        with experiment.stage('00_verify_reused_parameters'):
            copy_review_inputs(source, experiment.path, spec['upstream_identity'])
            write_json(experiment.path / 'hardware.json', {
                'gpu_names': [torch.cuda.get_device_name(i) for i in (0, 1)],
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'reviewer_device': 'cuda:1', 'gpu0_inference': False,
                'extraction_reused': True, 'enable_thinking': False})
        with experiment.stage('01_qwen9b_review'):
            seconds = int(limits['internal_seconds'] - (time.monotonic() - experiment.started))
            command = [sys.executable, '-u', '-m', 'parameters.semantic_validation_run',
                       '--stage', 'review', '--root', str(experiment.path), '--config', str(config),
                       '--model', 'qwen9b', '--seconds', str(seconds - 15)]
            process = stream_process(command, experiment.path / 'logs/qwen9b_worker.log', seconds, environment)
            path = experiment.path / 'qwen9b/status.json'
            result = json.loads(path.read_text()) if path.is_file() else {'status': 'missing_worker_status'}
            if process['returncode']:
                result = {**result, 'status': 'failed', 'worker_returncode': process['returncode']}
            write_json(experiment.path / 'status.json', {
                'status': result['status'], 'model_status': {'qwen9b': result}, 'process': process,
                'seconds': time.monotonic() - experiment.started, 'enable_thinking': False,
                'extraction_reused': True, 'no_policy_training': True})
    except Exception as error:
        traceback.print_exc()
        write_json(experiment.path / 'status.json', {
            'status': 'failed', 'error': type(error).__name__, 'message': str(error),
            'enable_thinking': False, 'seconds': time.monotonic() - experiment.started})
    finally:
        write_json(experiment.path / 'checksums.json', {
            str(path.relative_to(experiment.path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in experiment.path.rglob('*') if path.is_file() and path.name != 'checksums.json'})
    return experiment.path
