"""Build the v14 Kaggle wrapper around the staged immutable source ZIP."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import zipfile
from pathlib import Path


RUN = Path(__file__).resolve().parent
PACKAGE = RUN / "submission_v14_cnn_compare"
SOURCE_ZIP = PACKAGE / "source.zip"
CANONICAL = (
    RUN.parents[1]
    / "notebooks"
    / "notebook9_efficientnet_b1_comparison.ipynb"
)


def source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


source_bytes = SOURCE_ZIP.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
package_b64 = base64.b64encode(source_bytes).decode("ascii")
bootstrap = f'''import os,sys,json,base64,hashlib,zipfile,subprocess,importlib.metadata as metadata,time,shutil,socket,traceback,io
from pathlib import Path
print('START v14 immutable CNN comparison package bootstrap',flush=True)
PACKAGE_BYTES=base64.b64decode({package_b64!r})
EXPECTED_PACKAGE_SHA256={source_sha256!r}
if hashlib.sha256(PACKAGE_BYTES).hexdigest()!=EXPECTED_PACKAGE_SHA256: raise RuntimeError('embedded source package hash mismatch')
PROJECT=Path('/kaggle/working/project'); PROJECT.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(PACKAGE_BYTES)) as archive:
    bad=archive.testzip()
    if bad is not None: raise RuntimeError('source zip CRC failure: '+bad)
    for member in archive.namelist():
        path=Path(member)
        if path.is_absolute() or '..' in path.parts: raise RuntimeError('unsafe source archive member '+member)
    archive.extractall(PROJECT)
SOURCE_MANIFEST=json.loads((PROJECT/'SOURCE_MANIFEST.json').read_text())
for rel,expected in SOURCE_MANIFEST['source_files'].items():
    actual=hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()
    if actual!=expected: raise RuntimeError('source hash mismatch '+rel)
LABEL_PATH=PROJECT/SOURCE_MANIFEST['embedded_v13_teacher_labels']['path']
if hashlib.sha256(LABEL_PATH.read_bytes()).hexdigest()!=SOURCE_MANIFEST['embedded_v13_teacher_labels']['sha256']: raise RuntimeError('embedded teacher-label digest mismatch')
sys.path.insert(0,str(PROJECT))
RUN_ROOT=Path('/kaggle/working/experiments/spi_classifier_repair/runs/notebook9_20261028_v14')
RUN_ROOT.mkdir(parents=True,exist_ok=False)
print('Source package bytes',len(PACKAGE_BYTES),'sha256',EXPECTED_PACKAGE_SHA256,'files',SOURCE_MANIFEST['source_file_count'],flush=True)
print('Output root',RUN_ROOT,'embedded labels',SOURCE_MANIFEST['embedded_v13_teacher_labels'],flush=True)
'''

setup = '''import importlib.metadata as md, json, torch, torchvision, numpy, PIL, socket, subprocess, sys
print('START environment preflight',flush=True)
if torch.cuda.device_count()!=2: raise RuntimeError('comparison requires exactly two assigned GPUs')
GPU_NAMES=[torch.cuda.get_device_name(i) for i in range(2)]
if any('T4' not in x for x in GPU_NAMES): raise RuntimeError('expected two Tesla T4 GPUs: '+repr(GPU_NAMES))
for requirement in ('scikit-learn==1.7.2','nbformat==5.10.4'):
    command=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',requirement]
    print('RUN pinned active-kernel dependency install',command,flush=True)
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    print(result.stdout,flush=True);result.check_returncode()
import sklearn
if md.version('scikit-learn')!='1.7.2': raise RuntimeError('requires pinned scikit-learn 1.7.2; found '+md.version('scikit-learn'))
if md.version('nbformat')!='5.10.4': raise RuntimeError('requires pinned nbformat 5.10.4 for executed notebook capture; found '+md.version('nbformat'))
# Preserve Kaggle's installed Torch/TorchVision pair; never pip-upgrade GPU packages.
socket.setdefaulttimeout(90)
ENV={'python':__import__('sys').version,'interpreter':__import__('sys').executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'torch_cuda':torch.version.cuda,'numpy':numpy.__version__,'scikit-learn':sklearn.__version__,'Pillow':PIL.__version__,'nbformat':md.version('nbformat'),'gpus':GPU_NAMES,'train_device':'cuda:0','device_parallelism':False}
print('Observed environment',ENV,flush=True)
if not hasattr(torchvision.models,'EfficientNet_B1_Weights'): raise RuntimeError('installed TorchVision lacks pinned EfficientNet-B1 weights enum')
print('END environment preflight',flush=True)
'''

input_stage = '''import hashlib,json
from pathlib import Path
from parameters.artifacts import write_json
from parameters.dataset_pipeline import discover_input_root,materialize_current_300
print('START immutable input and label verification',flush=True)
INPUT_ROOT=discover_input_root('/kaggle/input',SOURCE_MANIFEST['input']['manifest_sha256'])
records,unused_features=materialize_current_300(INPUT_ROOT,RUN_ROOT/'materialized_images')
if len(records)!=300: raise RuntimeError('expected exactly 300 input records')
labels=json.loads(LABEL_PATH.read_text(encoding='utf-8'))
if len(labels)!=300: raise RuntimeError('expected exactly 300 embedded v13 labels')
INPUT_RECEIPT={'root':str(INPUT_ROOT),'manifest_sha256':hashlib.sha256((INPUT_ROOT/'input_manifest.json').read_bytes()).hexdigest(),'record_count':len(records),'embedded_labels_sha256':hashlib.sha256(LABEL_PATH.read_bytes()).hexdigest(),'label_count':len(labels),'label_source':'verified v13 teacher pseudo-labels','feature_cache_used':False}
write_json(RUN_ROOT/'input_receipt.json',INPUT_RECEIPT)
print('Input verified',INPUT_RECEIPT,flush=True)
print('END immutable input and label verification',flush=True)
'''

cnn_stage = '''import json,importlib.util
from pathlib import Path
print('START bounded EfficientNet-B1 transfer comparison; CPU orchestration; train on cuda:0',flush=True)
cnn_path=PROJECT/'experiments/spi_classifier_repair/cnn_compare.py'
cnn_spec=importlib.util.spec_from_file_location('spi_cnn_compare',cnn_path)
cnn_compare=importlib.util.module_from_spec(cnn_spec)
cnn_spec.loader.exec_module(cnn_compare)
comparison_root=RUN_ROOT/'efficientnet_b1'
cnn_compare.run(records,labels,comparison_root,device='cuda:0',seconds=1500,max_epochs=20)
metrics=json.loads((comparison_root/'validation.json').read_text())
print('CNN validation summary',metrics,flush=True)
print('END EfficientNet-B1 comparison',flush=True)
'''

runner = '''from parameters.cloud_notebook import CellRecorder
CELL_SOURCES=[SETUP_SOURCE,INPUT_SOURCE,CNN_SOURCE]
RECORDER=CellRecorder(CELL_SOURCES,RUN_ROOT/'cell_evidence')
RUN_ERROR=None
try:
    for index,stage_source in enumerate(CELL_SOURCES):
        with RECORDER.cell(index): exec(compile(stage_source,'spi_cnn_stage_'+str(index),'exec'),globals())
except BaseException as error:
    RUN_ERROR=error
    print('COMPARISON FAILED; preserving incremental outputs',type(error).__name__,str(error),flush=True)
finally:
    runtime={'run_id':'notebook9_20261028_v14','source_package_sha256':EXPECTED_PACKAGE_SHA256,'environment':globals().get('ENV'),'failure':None if RUN_ERROR is None else {'type':type(RUN_ERROR).__name__,'message':str(RUN_ERROR)},'status':'complete' if RUN_ERROR is None else 'failed'}
    (RUN_ROOT/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\\n')
    shutil.copy2(PROJECT/'SOURCE_MANIFEST.json',RUN_ROOT/'SOURCE_MANIFEST.json')
    payload=[p for p in RUN_ROOT.rglob('*') if p.is_file() and p.name not in ('retrieval_checksums.json','run_recovery.zip')]
    checks={str(p.relative_to(RUN_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}
    (RUN_ROOT/'retrieval_checksums.json').write_text(json.dumps(checks,indent=2)+'\\n')
    with zipfile.ZipFile('/kaggle/working/run_recovery.zip','w',zipfile.ZIP_DEFLATED) as recovery:
        for p in payload: recovery.write(p,str(p.relative_to(RUN_ROOT)))
        recovery.write(RUN_ROOT/'retrieval_checksums.json','retrieval_checksums.json')
    print('Recovery archive finalized',len(payload),'payload files','failed',repr(RUN_ERROR),flush=True)
if RUN_ERROR is not None: raise RUN_ERROR
'''

code = "\n".join(
    [
        bootstrap,
        "SETUP_SOURCE=" + repr(setup),
        "INPUT_SOURCE=" + repr(input_stage),
        "CNN_SOURCE=" + repr(cnn_stage),
        runner,
    ]
)
ast.parse(code)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "v14-purpose",
            "metadata": {},
            "source": [
                "# Notebook9 v14 — EfficientNet-B1 SPI comparison\n",
                "\n",
                "Canonical repository: `SemanticSchedulerEnd2End`, source commit `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0` plus the exact working-tree snapshot and file hashes embedded in this notebook.\n",
                "\n",
                "Execution target: the existing Kaggle notebook identity only. Environment: Kaggle-managed Python 3.13; use its preinstalled compatible Torch/TorchVision CUDA pair without upgrading it, require scikit-learn 1.7.2 and nbformat 5.10.4, and record observed versions. No local virtual environment transfers to Kaggle.\n",
                "\n",
                "Staged for the existing private notebook identity; **not submitted**. Compares one image-only transfer-learning candidate against the saved frozen-CLIP-feature baseline using the same 300 fixed images and verified v13 pseudo-labels. No VLM inference, new labels, parameter release, or readiness promotion.\n",
                "\n",
                "Requires exactly two Tesla T4 GPUs; training uses `cuda:0`. Kaggle timeout is 1,800 seconds (at most 1 aggregate GPU-hour for two assigned devices). The model stage is capped at 1,500 seconds and 20 epochs. Internal grouped stopping uses training groups only; the exposed 96-row outer validation is evaluated once after refitting on the 204-row train split. No test split is used.\n",
                "\n",
                "Source, labels, and input manifest are SHA-256 pinned. Runtime records Python, Torch, TorchVision, CUDA, scikit-learn, and hardware. Recovery outputs are written incrementally under `/kaggle/working/experiments/spi_classifier_repair/runs/notebook9_20261028_v14/`.",
            ],
        },
        {"cell_type": "code", "id": "v14-runner", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "codex": {"purpose": "staged version14 existing identity only; pending Medium and main clearance", "source_package_sha256": source_sha256, "timeout_seconds": 1800, "aggregate_gpu_hour_cap": 1},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
raw = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
CANONICAL.parent.mkdir(parents=True, exist_ok=True)
CANONICAL.write_bytes(raw)
(PACKAGE / "notebook.ipynb").write_bytes(raw)

