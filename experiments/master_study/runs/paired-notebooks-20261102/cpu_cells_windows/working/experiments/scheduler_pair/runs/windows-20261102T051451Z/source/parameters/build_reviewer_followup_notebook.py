"""Package one bounded reviewer-only follow-up; never submit implicitly."""
from pathlib import Path
import argparse
import json
import nbformat as nbf
from .build_annotation_notebook import build as build_base
from .artifacts import write_json


def build(output,tag):
    output=build_base(output,tag,scope='paired reviewer repair:72calls;no extraction/training')
    notebook=nbf.read(output/'notebook.ipynb',as_version=4)
    setup=[c.source for c in notebook.cells if c.cell_type=='code'][0].split('\n',1)[1]
    setup=setup.replace('Configuration:GPU0CLIP+YOLOWorld,GPU1Qwen4B;server timeout3600s;internal3000s',
                        'Configuration:GPU1Qwen4B;GPU0idle;server1200s;internal900s;72calls')
    dependencies='''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata
    print('START active-kernel dependencies; no detector or CLIP installation',flush=True)
    result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
        'transformers==4.57.6','huggingface-hub==0.36.2','accelerate>=1.1,<2','sentencepiece==0.2.2','nbformat'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    print(result.stdout,flush=True);result.check_returncode()
    import torch
    print('Assigned GPUs',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
    assert torch.cuda.device_count()==2,'Two actual GPUs required before model loading'
    versions={p:metadata.version(p) for p in ('torch','torchvision','numpy','Pillow','transformers','huggingface-hub','accelerate','nbformat')}
    print('Actual versions',versions,flush=True)
    from parameters.preflight_reviewer_processor import run as processor_preflight
    processor_preflight('/kaggle/working/processor_preflight.json')
    from parameters.artifacts import write_json
    write_json(Path('/kaggle/working/dependency_receipt.json'),{'packages':versions,'cuda':torch.version.cuda,'gpu_count':2})'''
    execute=f'''with recorder.cell(2):
    from parameters.cloud_notebook import locate
    from parameters.reviewer_followup import run
    print('START 72-call paired reviewer repair; no new data download',flush=True)
    OUTPUT=run(locate('data/records.json'),Path('/kaggle/working/results/reviewer_followup/{tag}'),
        ROOT/'parameters/config/reviewer_followup_cohort.json',seconds=900)
    print('ACTUAL outcome',json.loads((OUTPUT/'status.json').read_text()),flush=True)'''
    summary='''with recorder.cell(3):
    from parameters.artifacts import write_json
    hashes={str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.rglob('*') if p.is_file()}
    write_json(Path('/kaggle/working/artifact_checksums.json'),hashes)
    print('Review summaries',json.loads((OUTPUT/'summary.json').read_text()),flush=True)
    print('Saved raw responses, evaluator cases, source/model provenance and checksums. No automatic prompt selection.',flush=True)'''
    sources=[setup,dependencies,execute,summary]
    notebook.cells=[nbf.v4.new_markdown_cell(f'''# Reviewer repair — {tag}

Compare original v2 and explicit v4 on six exposed cases with/without images
(24 calls) and 24 fresh paired controls per prompt (48 calls). Prior 56 control
groups excluded from the fresh cohort. Fixed Qwen3-VL-4B, FP16 SDPA on GPU1,
greedy decoding, 192 output tokens, 512-side contain, 262144 effective pixels.
GPU0 is allocated but idle; budgets account for both assigned GPUs.

Managed Kaggle Python environment; active-kernel dependency installation and
processor preflight print actual versions/hardware. Prepared private 20261025b
input only. Server timeout1200s, internal900s, at most0.667 aggregate GPU-hours.
No automatic retry, extraction, training or model expansion.

Outputs: /kaggle/working/results/reviewer_followup/{tag}/, cell_evidence/ with
executed notebook and per-cell logs, dependency/processor receipts and checksums.
Raw responses flush per case. Inspect experiment status beyond kernel status.
Annotation-assisted judgments are not independent visual recognition truth.
This run qualifies only the same exact-key control protocol on eight new groups.''')]
    for i,source in enumerate(sources):
        compile(source,'reviewer_followup_cell_'+str(i),'exec')
        notebook.cells.append(nbf.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if i==0 else '')+source))
    nbf.write(notebook,output/'notebook.ipynb')
    metadata=json.loads((output/'kernel-metadata.json').read_text())
    slug='satyapaladugu/betterscheduler-reviewer-repair-'+tag
    metadata.update(id=slug,title='Betterscheduler Reviewer Repair '+tag)
    write_json(output/'kernel-metadata.json',metadata)
    manifest=json.loads((output/'pilot_manifest.json').read_text())
    manifest.update(slug=slug,server_seconds=1200,internal_seconds=900,maximum_allocated_gpu_hours=2/3,
                    stage='paired_reviewer_repair',max_review_calls=72)
    write_json(output/'pilot_manifest.json',manifest)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args();print(build(args.output,args.tag))
