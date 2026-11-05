"""Package one bounded Qwen9B-only review of pinned, already extracted inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import nbformat as nbf
from .artifacts import write_json
from .build_annotation_notebook import build as build_source


def build(output, tag):
    output = build_source(output, tag, scope='review-only9B; cached parameters; thinking OFF')
    previous = nbf.read(output / 'notebook.ipynb', as_version=4)
    setup = [c.source for c in previous.cells if c.cell_type == 'code'][0].split('\n', 1)[1]
    setup = setup.replace('Configuration:GPU0CLIP+YOLOWorld,GPU1Qwen4B;server timeout3600s;internal3000s',
                          'Configuration:cached extraction;GPU1Qwen3.5-9BNF4;thinkingOFF;server1800s/internal1650s')
    dependencies = '''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata
    print('START reviewer-only runtime; keep managed torch/CUDA; no visual extraction',flush=True)
    command=[sys.executable,'-m','pip','install','--disable-pip-version-check',
        'transformers==5.3.0','huggingface-hub==1.5.0','accelerate==1.12.0',
        'bitsandbytes==0.49.2','sentencepiece==0.2.2','nbformat']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    print(result.stdout,flush=True);result.check_returncode()
    import torch,transformers,importlib.util
    from transformers import Qwen3_5ForConditionalGeneration
    assert torch.cuda.device_count()==2,'Two assigned GPUs required'
    assert not any(importlib.util.find_spec(name) is not None for name in ('fla','causal_conv1d'))
    packages={name:metadata.version(name) for name in ('torch','torchvision','transformers','huggingface-hub','accelerate','bitsandbytes')}
    print('Actual interpreter',sys.executable,sys.version,flush=True)
    print('Actual packages',packages,'GPUs',[torch.cuda.get_device_name(i) for i in (0,1)],flush=True)
    from parameters.artifacts import write_json
    write_json(Path('/kaggle/working/dependency_receipt.json'),{'packages':packages,'cuda':torch.version.cuda,'enable_thinking':False})
    os.environ['HF_HOME']='/kaggle/temp/hf_review'
    os.environ['PYTHONUTF8']='1'
    print('END reviewer runtime ready; no weights loaded yet',flush=True)'''
    execute = f'''with recorder.cell(2):
    from parameters.cloud_notebook import locate
    from parameters.review_only import run
    SOURCE_RUN=locate('data/review_cohort.json').parents[1]
    print('START Qwen9B review-only; same30cases; largest actual JSON first',SOURCE_RUN,flush=True)
    OUTPUT=run(ROOT/'parameters/config/review_only_v1.json',SOURCE_RUN,Path('/kaggle/working/results/semantic_validation/{tag}'))
    print('ACTUAL experiment outcome',json.loads((OUTPUT/'status.json').read_text()),flush=True)'''
    summary = '''with recorder.cell(3):
    from parameters.artifacts import write_json
    for name in ('qwen9b/status.json','qwen9b/metrics.json','qwen9b/technical_preflight.json'):
        path=OUTPUT/name
        if path.is_file():print(name,json.loads(path.read_text()),flush=True)
    write_json(Path('/kaggle/working/artifact_checksums.json'),{
        str(path.relative_to(OUTPUT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in OUTPUT.rglob('*') if path.is_file()})
    print('END saved review outputs; inspect actual coverage and independent metrics; no automatic retry',flush=True)'''
    sources = [setup, dependencies, execute, summary]
    cells = [nbf.v4.new_markdown_cell(f'''# Qwen9B parameter review — {tag}

User-authorized review-only rerun. Reuse checksum-pinned h3 images and parameter
JSONs: six balanced parents,24controls+6actual cases. No extraction,4Breview,
parameter selection or scheduler training. Largest actual JSON first,then shared
technical controls,then remaining cases. Raw candidate values and reference labels
are unchanged. Prompt semantic_contract_v5 remains fixed; no tuning for approvals.

Qwen3.5-9B,NF4/FP16 compute,SDPA,native Torch fallback,thinking OFF on GPU1.
GPU0 is assigned but performs no inference; resource cap accounts for both GPUs.
Managed Kaggle runtime;setup prints exact source,interpreter,packages and devices.
Server1800s/internal1650s:at most1aggregateGPUhour. No automatic retry.

Inputs:completed h3 notebook outputs,verified against pinned manifest/checksum
identities and image hashes before weights load. Output:/kaggle/working/results/
semantic_validation/{tag}/. Executed notebook and four cell logs:cell_evidence/.
Model weights remain in/kaggle/temp. Retrieval/checksum verification is required.
This is development evidence,not final validation of all semantic parameters.''')]
    for index, source in enumerate(sources):
        compile(source, 'review-cell-' + str(index), 'exec')
        cells.append(nbf.v4.new_code_cell(('CELL_SOURCES=' + repr(sources) + '\n' if index == 0 else '') + source))
    notebook = nbf.v4.new_notebook(cells=cells, metadata=previous.metadata)
    nbf.write(notebook, output / 'notebook.ipynb')
    slug = 'satyapaladugu/betterscheduler-review-' + tag
    write_json(output / 'kernel-metadata.json', {
        'id': slug, 'title': 'Betterscheduler Qwen9B Review ' + tag, 'code_file': 'notebook.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
        'machine_shape': 'NvidiaTeslaT4', 'enable_internet': True, 'dataset_sources': [],
        'competition_sources': [],
        'kernel_sources': ['satyapaladugu/betterscheduler-semantic-validation-20261026h3']})
    review_manifest = {
        'tag': tag, 'slug': slug, 'server_seconds': 1800, 'internal_seconds': 1650,
        'maximum_allocated_gpu_hours': 1, 'planned_reviews': 30, 'enable_thinking': False,
        'source_zip_sha256': hashlib.sha256((output / 'source.zip').read_bytes()).hexdigest(),
        'upstream': 'satyapaladugu/betterscheduler-semantic-validation-20261026h3'}
    write_json(output / 'review_manifest.json', review_manifest)
    # The shared source packager also creates this established manifest. Keep its
    # source identity but replace the inherited annotation-pilot run settings.
    pilot_manifest = json.loads((output / 'pilot_manifest.json').read_text())
    pilot_manifest.update(review_manifest)
    write_json(output / 'pilot_manifest.json', pilot_manifest)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    print(build(args.output, args.tag))
