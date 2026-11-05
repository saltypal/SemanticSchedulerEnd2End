"""Publish a bounded evidence subset into Git; preserve original local artifacts."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys
from .artifacts import write_json


def export():
    root = Path(__file__).resolve().parent
    target = root/'handoff/evidence'
    target.mkdir(parents=True, exist_ok=False)
    evidence = {}
    def copy(source, relative):
        source, destination = root/source, target/relative
        content = source.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        evidence[str(relative).replace('\\','/')] = dict(source=str(source.relative_to(root)).replace('\\','/'),
            sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    h3 = Path('results/ops/20261025T2256Z/semantic_validation_h3/monitor/retrieved/results/semantic_validation/20261026h3')
    fixture_names = ['manifest.json','checksums.json','config.json','status.json',
        'data/records.json','data/review_cohort.json','data/controls_evaluator_only.json',
        'data/actual_evaluator_only.json','data/full_controls_evaluator_only.json',
        'data/full_actual_evaluator_only.json','extraction/features.jsonl','extraction/trace.json',
        'extraction/status.json','extraction/producer.json']
    for name in fixture_names: copy(h3/name, Path('h3')/name)
    write_json(target/'h3/FIXTURE_MANIFEST.json', dict(
        scope='exact consumed metadata/predictions subset; images recoverable separately; not complete original run',
        source_run='satyapaladugu/betterscheduler-semantic-validation-20261026h3',
        files={name:hashlib.sha256((target/'h3'/name).read_bytes()).hexdigest() for name in fixture_names}))
    i = Path('results/ops/20261025T2356Z/review_only_package/retrieved/results/semantic_validation/20261026i')
    for name in ('status.json','config.json','hardware.json','manifest.json','checksums.json','upstream_receipt.json',
                 'qwen9b/reviews.jsonl','qwen9b/metrics.json','qwen9b/model.json','qwen9b/status.json',
                 'qwen9b/technical_preflight.json','qwen9b/processor_preflight.json',
                 'data/controls_evaluator_only.json','data/actual_evaluator_only.json'):
        if (root/i/name).is_file(): copy(i/name,Path('review_i')/name)
    for path in sorted((root/i/'qwen9b/raw').glob('*.txt')):
        copy(path.relative_to(root),Path('review_i/qwen9b/raw')/path.name)
    local = Path('results/local/json_v2_20261026')
    for folder in ('converted_summary/documents','settings','validation'):
        for path in sorted((root/local/folder).glob('*')):
            if path.is_file() and path.suffix in ('.json','.log','.yaml'):
                copy(path.relative_to(root),Path('json_v2')/path.relative_to(root/local))
    if (root/local/'converted_summary/REPORT.json').is_file():
        copy(local/'converted_summary/REPORT.json',Path('json_v2/converted_summary/REPORT.json'))
    operations = [
        ('results/ops/20261025T2356Z/review_only_package/FINAL_REPORT.md','review_i/FINAL_REPORT.md'),
        ('results/ops/20261025T2356Z/review_only_package/terminal_audit.json','review_i/terminal_audit.json'),
        ('results/ops/20261025T2356Z/review_only_package/terminal_checksums_verified.json','review_i/terminal_checksums_verified.json'),
        ('results/ops/20261025T2356Z/review_only_package/source_manifest_archive_verified.json','review_i/source_manifest_archive_verified.json')]
    for source, destination in operations:
        if (root/source).is_file(): copy(Path(source),Path(destination))
    versions = {}
    for package in ('numpy','Pillow','scikit-learn','matplotlib','psutil','PyYAML','torch','torchvision',
                    'transformers','huggingface-hub','jsonschema','pytest','nbformat','nbclient'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    write_json(target/'LOCAL_ENVIRONMENT.json', dict(python=sys.version, interpreter=sys.executable,
        packages=versions, scope='observed laptop environment, not a portable lockfile or Kaggle environment'))
    write_json(target/'EVIDENCE_MANIFEST.json',dict(
        files=evidence, excluded=['raw dataset images','model weights','generated staging ZIPs',
        'full prepared multi-GB datasets','credentials','redundant audit sidecars'],
        note='Copied subsets have their own file hashes. Original run manifests may list files not included here. '
             'All original artifacts remain local; KAGGLE_RUNS contains remote retrieval locations.'))
    return {'files':len(evidence),'copied_bytes':sum(v['bytes'] for v in evidence.values()),'destination':str(target)}


if __name__ == '__main__': print(json.dumps(export(),indent=2))
