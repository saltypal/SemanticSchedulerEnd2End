"""Immutable balanced-semantic panel, real extraction, and non-thinking model comparison."""
from pathlib import Path
import argparse
import json
import nbformat as nbf
from .build_annotation_notebook import build as build_base
from .artifacts import write_json


def build(output,tag,server_seconds=None):
    spec=json.loads((Path(__file__).parent/'config/semantic_validation_v1.json').read_text(encoding='utf-8-sig'))
    internal=spec['limits']['internal_seconds']
    server_seconds=spec['limits']['server_seconds'] if server_seconds is None else server_seconds
    if not internal<server_seconds<=spec['limits']['server_seconds']:raise ValueError('server cap outside frozen budget')
    review_parents=3*spec['review_parents_per_stratum']
    review_calls=2*5*review_parents
    output=build_base(output,tag,scope='balanced semantic contract;real extraction;non-thinking4Bvs9BNF4')
    notebook=nbf.read(output/'notebook.ipynb',as_version=4)
    setup=[c.source for c in notebook.cells if c.cell_type=='code'][0].split('\n',1)[1]
    setup=setup.replace('Configuration:GPU0CLIP+YOLOWorld,GPU1Qwen4B;server timeout3600s;internal3000s',
        f'Configuration:GPU0extractor,GPU1sequential4B/9BNF4;thinkingOFF;server{server_seconds}s/internal{internal}s')
    setup=setup.replace("os.environ['YOLO_CONFIG_DIR']='/tmp/better_yolo_config'",
        "os.environ['YOLO_CONFIG_DIR']='/tmp/better_yolo_config'\nos.environ['HF_HOME']='/kaggle/temp/better_hf'")
    dependencies='''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata
    print('START baseline dependency setup; matched Kaggle Torch/CUDA retained',flush=True)
    command=[sys.executable,'-m','pip','install','--disable-pip-version-check',
        'transformers==4.57.6','huggingface-hub==0.36.2','accelerate==1.12.0','sentencepiece==0.2.2','ftfy','nbformat']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    print(result.stdout,flush=True);result.check_returncode()
    command=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
        'ultralytics==8.3.237','git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    print(result.stdout,flush=True);result.check_returncode()
    import torch
    versions={p:metadata.version(p) for p in ('torch','torchvision','numpy','Pillow','transformers','huggingface-hub','accelerate','ultralytics')}
    print('Actual environment',sys.executable,versions,flush=True)
    print('Actual GPUs',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
    assert torch.cuda.device_count()==2,'Two T4s required before expensive inference'
    from parameters.preflight_reviewer_processor import run as preflight
    preflight('/kaggle/working/baseline_processor_preflight.json')
    from parameters.artifacts import write_json
    write_json(Path('/kaggle/working/dependency_receipt.json'),{'packages':versions,'cuda':torch.version.cuda,'enable_thinking':False})'''
    execute=f'''with recorder.cell(2):
    from parameters.semantic_validation_driver import run
    print('START balanced official-test-source pilot; masks/rules/facts evaluated separately',flush=True)
    OUTPUT=run(ROOT/'parameters/config/semantic_validation_v1.json',Path('/kaggle/working/results/semantic_validation/{tag}'),server_seconds={server_seconds})
    print('ACTUAL experiment outcome',json.loads((OUTPUT/'status.json').read_text()),flush=True)'''
    summary='''with recorder.cell(3):
    from parameters.artifacts import write_json
    for name in ('extraction/status.json','qwen4b/status.json','qwen4b/metrics.json','qwen9b/status.json','qwen9b/metrics.json'):
        path=OUTPUT/name
        if path.exists():print(name,json.loads(path.read_text()),flush=True)
    hashes={str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.rglob('*') if p.is_file()}
    write_json(Path('/kaggle/working/artifact_checksums.json'),hashes)
    present={name:(OUTPUT/name).is_file() for name in ('extraction/trace.json','extraction/features.jsonl',
        'data/controls_evaluator_only.json','qwen4b/reviews.jsonl','qwen9b/reviews.jsonl','checksums.json')}
    print('ACTUAL saved artifact availability',present,flush=True)
    print('Inspect complete/partial/failed status; no automatic model promotion or retry.',flush=True)'''
    sources=[setup,dependencies,execute,summary]
    notebook.cells=[nbf.v4.new_markdown_cell(f'''# Balanced semantic validation — {tag}

Purpose: distinguish literal field availability, declared semantic recovery and
factual correctness. Twelve new official LADI test-source images, four per native
state:absent,present-unaffected,affected. Known/near pixel matches excluded locally;
runtime pixels, native labels and source revision reverified. This small panel is
development evidence; event/site independence and external pretraining are unknown.

GPU0:CLIP,YOLO-World and native LADI classifier extraction. Three independent
constructed UE histories,15observations including exact repeats. Native annotations
never enter image extraction. Spatial/complexity remain unverified box proxies.
GPU1:sequential Qwen3-VL-4B FP16 and Qwen3.5-9B NF4,thinking explicitly OFF.
Each sees the same{4*review_parents}constructed controls and{review_parents}actual parameter JSONs
from{review_parents}balanced parents;full12image extraction is retained.9Bhas an
isolated Transformers5.3 runtime;baseline stays4.57.6. This compares deployed
model/runtime configurations, not parameter count alone. Technical/memory/time
preflight can stop9Bearly; partial results must remain visibly partial.

Managed Kaggle Python with printed package/source/device identity. Server{server_seconds}s,
internal{internal}s,at most{server_seconds/1800:.3f}aggregateGPUhours. Baseline setup occurs before internal
timer but inside the server cap. No automatic follow-on runs or scheduler training.
Images require about5.5MB from signed public dataset-viewer routes refreshed at
runtime; signed URLs are not persisted. Model weights are cached outside results.

Outputs:/kaggle/working/results/semantic_validation/{tag}/; cell_evidence/ has
executed notebook and per-cell logs. Child logs/JSONL flush incrementally. Root
checksums.json persists on failure; global artifact_checksums.json is saved at
summary. Verify saved-version retrieval before reporting completion.''')]
    for i,source in enumerate(sources):
        compile(source,'semantic_validation_cell_'+str(i),'exec')
        notebook.cells.append(nbf.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if i==0 else '')+source))
    nbf.write(notebook,output/'notebook.ipynb')
    metadata=json.loads((output/'kernel-metadata.json').read_text())
    slug='satyapaladugu/betterscheduler-semantic-validation-'+tag
    metadata.update(id=slug,title='Betterscheduler Semantic Validation '+tag,kernel_sources=[])
    write_json(output/'kernel-metadata.json',metadata)
    manifest=json.loads((output/'pilot_manifest.json').read_text())
    manifest.update(slug=slug,server_seconds=server_seconds,internal_seconds=internal,maximum_allocated_gpu_hours=server_seconds/1800,
                    upstream=None,stage='balanced_semantic_validation',max_review_calls=review_calls)
    write_json(output/'pilot_manifest.json',manifest)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--server-seconds',type=int)
    args=parser.parse_args();print(build(args.output,args.tag,args.server_seconds))
