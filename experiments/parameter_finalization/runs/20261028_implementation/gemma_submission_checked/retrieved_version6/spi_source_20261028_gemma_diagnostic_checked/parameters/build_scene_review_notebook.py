"""Immutable 44-call scene-v2 pilot; no extraction or paid API."""
from pathlib import Path
import argparse
import hashlib
import json
import nbformat as nbf
from .build_review_only_notebook import build as base_build
from .artifacts import write_json


def build(output, tag):
    output = base_build(output, tag)
    notebook = nbf.read(output/'notebook.ipynb', as_version=4)
    sources = [c.source for c in notebook.cells if c.cell_type=='code']
    sources[0] = sources[0].split('\n',1)[1]
    sources[1] = sources[1].replace("'bitsandbytes==0.49.2','sentencepiece==0.2.2','nbformat'",
        "'bitsandbytes==0.49.2','sentencepiece==0.2.2','nbformat','jsonschema>=4.23,<5'")
    sources[2] = f'''with recorder.cell(2):
    from parameters.cloud_notebook import locate
    from parameters.scene_review_run import run
    SOURCE_RUN=locate('data/review_cohort.json').parents[1]
    print('START sceneJSONv2 pilot: six parents;30audits,12QA,2synthetic context calls',flush=True)
    OUTPUT=run(ROOT/'parameters/config/scene_review_v1.json',SOURCE_RUN,Path('/kaggle/working/results/scene_review/{tag}'))
    print('ACTUAL outcome',json.loads((OUTPUT/'status.json').read_text()),flush=True)'''
    sources[3] = '''with recorder.cell(3):
    from parameters.artifacts import write_json
    for name in ('status.json','metrics.json','technical_preflight.json'):
        path=OUTPUT/name
        if path.is_file():print(name,json.loads(path.read_text()),flush=True)
    write_json(Path('/kaggle/working/artifact_checksums.json'),{
        str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUTPUT.rglob('*') if p.is_file()})
    print('END persisted cases,documents,audit,rawreview,metrics,model provenance and cell logs',flush=True)
    print('Policy controls are synthetic; no real-world urgency or whole-image completeness claim.',flush=True)'''
    notebook.cells = [nbf.v4.new_markdown_cell(f'''# Scene JSON v2 offline pilot — {tag}

Reuses pinned h3runtime evidence; no extraction or model training. Six exposed
parents,30audits(actual+fourcontrolledvariants),12separate image/JSON-only QA calls,
and2synthetic-time policy controls. Native12image labels score supported facts;
relationships/physicaldepth/realoperatorurgency remain unvalidated. Qwen sees
annotations only on the audit route. JSON-only route has no pixels or labels.
Operator policy is a declared engineering fixture, not medical/emergency advice.

Qwen3.5-9B NF4/FP16 compute,thinkingOFF,GPU1. Two assignedT4s are charged to the
budget; GPU0 performs no inference. Managed Kaggle Python, pinned model revision,
printed dependencies and source identity. Server1800s/internal1650s <=1GPUhour.
No paid API or automatic retry. Largest actual JSON preflight,then each QA route.

Outputs:/kaggle/working/results/scene_review/{tag}/;cell_evidence has the executed
notebook and four cell logs. Partial raw responses/metrics flush after each call.
Saved output retrieval/checksums remain necessary. Models cache outside results.''')]
    for i, source in enumerate(sources):
        compile(source,'scene-review-cell-'+str(i),'exec')
        notebook.cells.append(nbf.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if i==0 else '')+source))
    nbf.write(notebook,output/'notebook.ipynb')
    slug='satyapaladugu/betterscheduler-scene-v2-'+tag
    metadata=json.loads((output/'kernel-metadata.json').read_text())
    metadata.update(id=slug,title='Betterscheduler Scene V2 '+tag)
    write_json(output/'kernel-metadata.json',metadata)
    manifest=json.loads((output/'pilot_manifest.json').read_text())
    manifest.update(tag=tag,slug=slug,planned_reviews=44,server_seconds=1800,internal_seconds=1650,
        maximum_allocated_gpu_hours=1,source_zip_sha256=hashlib.sha256((output/'source.zip').read_bytes()).hexdigest(),
        protocol='scene_v2_offline_v1',urgency_scope='synthetic policy controls only')
    write_json(output/'pilot_manifest.json',manifest);write_json(output/'review_manifest.json',manifest)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args();print(build(args.output,args.tag))
