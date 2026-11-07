"""Build one private, immutable SPI teacher/student notebook. Does not launch it."""
from pathlib import Path
import argparse
import hashlib
import json
import nbformat as nbf
from .artifacts import write_json
from .build_annotation_notebook import build as source_package


def build(output,tag,input_dataset,server_seconds=3600):
    if not 600<=server_seconds<=3600:raise ValueError("SPI server budget must be600..3600seconds")
    internal=server_seconds-300
    output=source_package(output,tag,scope="SPI pseudo-label teacher and frozen-feature student; no scene model or scheduler fitting")
    old=nbf.read(output/"notebook.ipynb",as_version=4)
    setup=next(c.source for c in old.cells if c.cell_type=="code").split("\n",1)[1]
    setup="import os\nos.environ['HF_HOME']='/kaggle/temp/spi_models'\n"+setup
    setup=setup.replace("Configuration:GPU0CLIP+YOLOWorld,GPU1Qwen4B;server timeout3600s;internal3000s",
                        f"Configuration:cached CLIP, GPU1 Qwen3.5-9B NF4, thinking OFF, CPU SPI head, server{server_seconds}s")
    dependencies='''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata,importlib.util
    import torch
    print('START SPI runtime setup; retain managed torch/CUDA',flush=True)
    assert torch.cuda.device_count()==2,'Exactly two GPUs are required by the pinned Qwen9B loader'
    result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
        'transformers==5.3.0','huggingface-hub==1.5.0','accelerate==1.12.0',
        'bitsandbytes==0.49.2','sentencepiece==0.2.2'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    print(result.stdout,flush=True);result.check_returncode()
    from transformers import Qwen3_5ForConditionalGeneration
    assert not any(importlib.util.find_spec(x) is not None for x in ('fla','causal_conv1d'))
    packages={x:metadata.version(x) for x in ('torch','transformers','huggingface-hub','accelerate','bitsandbytes','numpy')}
    from parameters.artifacts import write_json
    write_json(Path('/kaggle/working/runtime.json'),{'python':sys.version,'interpreter':sys.executable,
        'packages':packages,'devices':[torch.cuda.get_device_name(i) for i in (0,1)],'thinking':False})
    os.environ['HF_HOME']='/kaggle/temp/spi_models'
    print('END runtime',packages,flush=True)'''
    execute=f'''with recorder.cell(2):
    import signal,time
    started=time.monotonic()
    def cutoff(signum,frame):raise TimeoutError('SPI whole-cell {internal}-second budget reached')
    signal.signal(signal.SIGALRM,cutoff);signal.alarm({internal})
    from parameters.cloud_notebook import locate
    from parameters.pilot_producers import snapshot_model
    from parameters.qwen35_reviewer import Qwen35AnnotationReviewer,MODEL_ID,MODEL_REVISION
    from parameters.model_settings import ReviewerSettings
    from parameters.spi_campaign import execute,verify_inputs
    INPUT=locate('input_manifest.json').parent
    _,receipt=verify_inputs(INPUT)
    print('Verified input BEFORE model download',receipt,flush=True)
    print('START model download; pinned Qwen9B; no local laptop weights',flush=True)
    model_path,revision=snapshot_model(MODEL_ID,MODEL_REVISION)
    reviewer=Qwen35AnnotationReviewer(model_path,revision,
        settings=ReviewerSettings(max_new_tokens=192,max_pixels=262144,image_max_side=512,prompt_variant='semantic_contract_v5'))
    reviewer.maximum_generation_seconds=30;reviewer.maximum_input_tokens=4096;reviewer.require_finite_preflight=True
    OUTPUT=execute(INPUT,Path('/kaggle/working/results/spi/{tag}'),reviewer,seconds=max(1,int({internal-100}-(time.monotonic()-started))))
    signal.alarm(0)
    print('END actual SPI status',json.loads((OUTPUT/'status.json').read_text()),flush=True)'''
    summary='''with recorder.cell(3):
    import zipfile
    root=Path('/kaggle/working/results/spi')
    files=[p for p in root.rglob('*') if p.is_file() and '/data/images/' not in str(p)]
    write_json(Path('/kaggle/working/spi_checksums.json'),{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with zipfile.ZipFile('/kaggle/working/spi_recovery.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,str(p.relative_to(root)))
    print('END saved models, raw labels, losses, statuses and cell logs; inspect experiment status',flush=True)'''
    sources=[setup,dependencies,execute,summary]
    for i,s in enumerate(sources):compile(s,f'spi-cell-{i}','exec')
    cells=[nbf.v4.new_markdown_cell(f'''# Fixed SPI development campaign — {tag}

One frozen CLIP ViT-B/32 feature cache, one four-class 128-unit GELU head.
Offline Qwen3.5-9B NF4 labels under config/spi_rubric_v1.json; thinking OFF.
No native source annotations or parameter predictions are teacher inputs.
Managed Kaggle Python environment: setup prints actual versions and devices.
No local virtual environment is assumed to transfer to Kaggle.

Input: private {input_dataset}; checksum-verified real images/cached features,
locked development groups. Native labels excluded. This development cohort is
exposed; held-out teacher agreement is not independent human urgency accuracy.
All four classes require actual support; absent classes do not receive invented labels.

Runtime: two GPUs allocated, GPU1 runs teacher; GPU0 unused because CLIP features
are cached. Both allocated devices count: server{server_seconds}s <={server_seconds*2/3600:g} aggregate GPU-hours.
No paid API and no automatic retry. CPU trains the head after GPU labels finish.

Outputs: /kaggle/working/results/spi/{tag}/, /kaggle/working/cell_evidence/,
runtime.json, spi_checksums.json and spi_recovery.zip. CellRecorder saves a live
executed notebook and cell logs. Recovery export after cell2 failure may require
running the final export cell; incremental labels/logs are persisted beforehand.
The server must expose saved outputs for durable retrieval; verify after completion.''')]
    for i,s in enumerate(sources):cells.append(nbf.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if i==0 else '')+s))
    notebook=nbf.v4.new_notebook(cells=cells,metadata=old.metadata)
    nbf.write(notebook,output/'notebook.ipynb')
    write_json(output/'kernel-metadata.json',{'id':'satyapaladugu/betterscheduler-spi-'+tag,
        'title':'parameters SPI '+tag,'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
        'dataset_sources':[input_dataset],'competition_sources':[],'kernel_sources':[]})
    write_json(output/'SPI_PACKAGE.json',{'tag':tag,'source_zip_sha256':hashlib.sha256((output/'source.zip').read_bytes()).hexdigest(),
        'input_dataset':input_dataset,'server_seconds':server_seconds,'aggregate_gpu_hour_cap':server_seconds*2/3600,'thinking':False,
        'state':'prepared_not_submitted'})
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--input-dataset',required=True);p.add_argument('--server-seconds',type=int,default=3600)
    a=p.parse_args();print(build(a.output,a.tag,a.input_dataset,a.server_seconds))
