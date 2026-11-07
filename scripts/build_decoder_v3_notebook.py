"""Build resumable private Kaggle notebooks from canonical V3 module sources."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['Archived/owners_before_simplification_20261027/SemParam/decoder_parameters.py', 'Archived/owners_before_simplification_20261027/SemParam/decoder_extractors.py',
           'Archived/owners_before_simplification_20261027/SemParam/decoder_costs.py', 'Scheduler/training/decoder_v3_ollama.py',
           'Scheduler/decoder_scheduler.py', 'Evaluation/decoder_queue.py', 'Scheduler/training/decoder_v3_extract.py',
           'Scheduler/training/decoder_v3_study.py','Scheduler/training/decoder_v3_analysis.py']


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build(stage='extract'):
    if stage not in {'extract', 'train', 'all','final'}:
        raise ValueError('stage must be extract/train/all')
    stream = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            path = ROOT/name
            if not path.exists():
                if name.endswith('decoder_v3_study.py') and stage == 'extract':
                    continue
                raise FileNotFoundError(path)
            data = path.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        for name in ('Archived/owners_before_simplification_20261027/SemParam/__init__.py','Scheduler/__init__.py','Evaluation/__init__.py','Scheduler/training/__init__.py'):
            archive.writestr(name, '"""Isolated canonical V3 source snapshot."""\n')
    encoded = base64.b64encode(stream.getvalue()).decode('ascii')
    manifest = dict(schema='decoder_v3_source_snapshot', source_files=hashes,
                    zip_sha256=hashlib.sha256(stream.getvalue()).hexdigest(), stage=stage,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    dirty_state=subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True))
    bootstrap = '''import os, sys, subprocess, json, platform, base64, hashlib, zipfile, io
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.1',
                'datasets==4.3.0', 'huggingface-hub==0.35.3', 'accelerate>=1.10,<2',
                'lightgbm==4.6.0', 'pillow>=11.3', 'matplotlib', 'scikit-learn', 'zstandard'], check=True)
'''
    bootstrap += f"source_manifest = {manifest!r}\narchive_bytes = base64.b64decode({encoded!r})\n"
    bootstrap += '''assert hashlib.sha256(archive_bytes).hexdigest() == source_manifest['zip_sha256']
work = Path('/kaggle/working')
code_root = work / 'canonical_source'
code_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    archive.extractall(code_root)
sys.path.insert(0, str(code_root))
for name, expected in source_manifest['source_files'].items():
    assert hashlib.sha256((code_root/name).read_bytes()).hexdigest() == expected
(work/'source_manifest.json').write_text(json.dumps(source_manifest, indent=2))
(work/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
import torch
hardware = dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.cuda.is_available(),
                gpu_count=torch.cuda.device_count(),gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
(work/'hardware.json').write_text(json.dumps(hardware, indent=2))
print(hardware)
assert hardware['gpu_count'] >= 2, 'The requested execution profile requires two GPUs.'
'''
    execute = f"STAGE = {stage!r}\n"
    execute += '''import time, traceback
started = time.time()
status = {'status':'running','stage':STAGE}
try:
    if STAGE == 'final':
        import shutil
        from Scheduler.training.decoder_v3_study import load_cache, verify_extraction_cache
        caches = [p.parent for p in Path('/kaggle/input').rglob('features.jsonl')
                  if (p.parent/'extraction_summary.json').exists()]
        assert len(caches) == 1, caches
        cache = caches[0]
        verified = verify_extraction_cache(cache, load_cache(cache))
        origins = [p.parent for p in Path('/kaggle/input').rglob('study/status.json')
                   if json.loads(p.read_text()).get('status') == 'completed']
        assert len(origins) == 1, origins
        origin = origins[0]
        (work/'resume_provenance.json').write_text(json.dumps(dict(
            mode='resume completed training; no test-based retuning',cache_verification=verified,
            origin_study=str(origin),origin_comparison_sha256=hashlib.sha256((origin/'comparison.csv').read_bytes()).hexdigest(),
            original_epoch_training_recorded_in='disaster-eight-field-scheduler version 1 logs/checkpoints'),indent=2))
        shutil.copytree(origin,work/'study',dirs_exist_ok=True)
        from Scheduler.training.decoder_v3_study import run_study
        run_study(cache,work/'study',config={'max_epochs':60,'patience':10,'train_queues':128,'seed':0})
        import csv
        old_rows = list(csv.DictReader((origin/'comparison.csv').open()))
        new_rows = list(csv.DictReader((work/'study/comparison.csv').open()))
        assert len(old_rows) == len(new_rows) == 39
        for old,new in zip(old_rows,new_rows):
            assert (old['task'],old['method']) == (new['task'],new['method'])
            for metric in ('A10','coverage5','coverage10','ndcg10','high_recall10','redundancy10'):
                assert abs(float(old[metric])-float(new[metric])) < 1e-12, (old['task'],old['method'],metric)
        from Scheduler.training.decoder_v3_analysis import analyze_run
        analyze_run(work,cache,work/'analysis')
    if STAGE in ('extract','all'):
        from Scheduler.training.decoder_v3_extract import run_extraction
        partial = [path.parent for path in Path('/kaggle/input').rglob('features.jsonl')
                   if (path.parent/'labels.jsonl').exists()]
        assert len(partial) <= 1, f'Ambiguous partial cache inputs: {partial}'
        run_extraction(work/'extraction', '/kaggle/temp/decoder_v3_hf', device='auto', qwen_images=32,
                       seed_cache=partial[0] if partial else None)
    if STAGE in ('train','all'):
        from Scheduler.training.decoder_v3_study import run_study
        if STAGE == 'all':
            cache = work/'extraction'
        else:
            caches = sorted(Path('/kaggle/input').rglob('features.jsonl'))
            caches = [p.parent for p in caches if (p.parent/'labels.jsonl').exists() and (p.parent/'extraction_summary.json').exists()]
            assert len(caches) == 1, f'Expected exactly one compatible extraction input, got {caches}'
            cache = caches[0]
        run_study(cache, work/'study', config={'max_epochs':60, 'patience':10, 'train_queues':128, 'seed':0})
    status.update(status='completed',elapsed_seconds=time.time()-started)
except Exception as error:
    status.update(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
    raise
finally:
    (work/'run_status.json').write_text(json.dumps(status, indent=2))
print(status)
'''
    presentation = '''from IPython.display import display, Image, Markdown
import pandas as pd
if STAGE in ('train','all','final'):
    result_status = json.loads((work/'study/status.json').read_text())
    assert result_status['status'] == 'completed', result_status
    display(Markdown('## Held-out comparison: controlled observation coverage'))
    display(pd.read_csv(work/'study/comparison.csv'))
    figure_root = work/'analysis' if (work/'analysis/figure1_coverage.png').exists() else work/'study'
    display(Image(filename=str(figure_root/'figure1_coverage.png')))
    display(Image(filename=str(figure_root/'figure2_paired_comparison.png')))
    display(Markdown('Source-image observation identities are not independently verified physical facts. '
                     'Service ticks are simulated; real codec reconstruction is not evaluated. '
                     'Qwen annotations are silver and do not define the primary reference labels.'))
else:
    display(json.loads((work/'extraction/extraction_summary.json').read_text()))
    display(json.loads((work/'extraction/qwen_status.json').read_text()))
'''
    raw_sources = [bootstrap,execute,presentation]
    capture_setup = ('from IPython.utils.capture import capture_output\n'
                     'import json, nbformat\n'
                     f'_report_sources = {raw_sources!r}\n'
                     '_captured_cells = []\n')
    wrapped_cells = []
    for index,source in enumerate(raw_sources):
        wrapper = (capture_setup if index == 0 else '')
        wrapper += (f'with capture_output() as _captured:\n'
                    f'    exec(_report_sources[{index}], globals())\n'
                    f'_captured_cells.append((_report_sources[{index}], get_ipython().execution_count, _captured))\n'
                    '_captured.show()\n')
        wrapped_cells.append(cell('code',wrapper))
    save_executed = '''_nb = nbformat.v4.new_notebook()
_nb.metadata.kernelspec = dict(display_name='Python 3',language='python',name='python3')
_execution_note = ('This notebook reused verified extraction and resumed the completed primary training run. '
    'Original full epoch histories/checkpoints are included in the study artifact. '
    if STAGE == 'final' else 'Actual execution stage: '+STAGE+'. ')
_nb.cells.append(nbformat.v4.new_markdown_cell('# Complete disaster decoder study — executed Kaggle artifact\\n\\n'
    +_execution_note+'Actual captured outputs are preserved below; no test-based retuning was performed.'))
for _source, _count, _capture in _captured_cells:
    _outputs = []
    if _capture.stdout:
        _outputs.append(nbformat.v4.new_output('stream',name='stdout',text=_capture.stdout))
    if _capture.stderr:
        _outputs.append(nbformat.v4.new_output('stream',name='stderr',text=_capture.stderr))
    for _rich in _capture.outputs:
        _outputs.append(nbformat.v4.new_output('display_data',data=_rich.data,metadata=_rich.metadata))
    _nb.cells.append(nbformat.v4.new_code_cell(_source,execution_count=_count,outputs=_outputs))
nbformat.validate(_nb)
nbformat.write(_nb,work/'05_complete_executed_disaster_scheduler.ipynb')
print('Saved actual executed notebook:',work/'05_complete_executed_disaster_scheduler.ipynb')
'''
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
                    cells=[cell('markdown', '# Eight-parameter disaster decoder scheduler\n\n'
                                'Canonical source snapshot; complete arrived messages; codec integration deferred.\n'
                                'Controlled observation coverage is distinct from verified physical-fact coverage.\n'
                                f'Execution stage: **{stage}**. All checkpoints and evidence are saved under `/kaggle/working`.\n'),
                           *wrapped_cells,cell('code',save_executed)])
    folder = ROOT/'kaggle'/f'disaster-decoder-v3-{stage}'
    folder.mkdir(parents=True, exist_ok=True)
    notebook_path = folder/'05_disaster_decoder_v3.ipynb'
    notebook_path.write_text(json.dumps(notebook,indent=1), encoding='utf-8')
    suffix = {'extract':'ollama','train':'scheduler','all':'end-to-end','final':'final-notebook'}[stage]
    title_suffix = {'extract':'Ollama','train':'Scheduler','all':'End To End','final':'Final Notebook'}[stage]
    metadata = dict(id=f'hmmmkk/disaster-eight-field-{suffix}',
                    title=f'Disaster Eight Field {title_suffix}',
                    code_file=notebook_path.name,language='python',kernel_type='notebook',is_private=True,
                    enable_gpu=True,enable_internet=True,
                    dataset_sources=['hmmmkk/disaster-eight-field-feature-cache-v1'] if stage=='extract' else [],
                    competition_sources=[],model_sources=[],
                    kernel_sources=(['hmmmkk/disaster-eight-field-ollama','hmmmkk/disaster-eight-field-scheduler']
                                    if stage=='final' else ['hmmmkk/disaster-eight-field-ollama'] if stage=='train' else []))
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if stage == 'all':
        canonical = ROOT/'Scheduler/notebooks/05_disaster_decoder_v3_source.ipynb'
        canonical.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print(folder)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['extract','train','all','final'],default='extract')
    build(parser.parse_args().stage)
