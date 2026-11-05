"""Build private Kaggle notebooks from one hashed minimal source snapshot."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import subprocess
import zipfile
import nbformat as nbf
from .artifacts import write_json
from .exposure import build_ledger


def build(output,tag):
    root=Path(__file__).resolve().parents[1];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    ledger=build_ledger(root,output/'exposure_ledger.json')
    files={str(p.relative_to(root)).replace('\\','/'):p.read_bytes() for p in (root/'parameters').glob('*.py')}
    files.update({str(p.relative_to(root)).replace('\\','/'):p.read_bytes()
                  for p in (root/'parameters/config').glob('*.json')})
    for name in ('Evaluation/decoder_semantic_benchmark.py','Evaluation/decoder_datasets.py',
                 'Scheduler/training/decoder_context_training.py','Scheduler/contextual_decoder_scheduler.py','Scheduler/decoder_scheduler.py'):
        files[name]=(root/name).read_bytes()
    stubs=['Evaluation/__init__.py','Scheduler/training/__init__.py','Scheduler/__init__.py']
    for name in stubs:files[name]=b'"""Isolated experiment namespace; selected canonical modules copied verbatim."""\n'
    files['exposure_ledger.json']=json.dumps(ledger,indent=2).encode()
    manifest={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
              'namespace_stubs':stubs,'exposure_ledger_sha256':ledger['sha256'],'campaign_tag':tag}
    files['SOURCE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in files.items():archive.writestr(name,value)
    (output/'source.zip').write_bytes(buffer.getvalue())
    payload=base64.b64encode(buffer.getvalue()).decode()
    slugs={stage:'better-scheduler-'+stage+'-'+tag for stage in ('prepare','extract','select')}
    for stage in slugs:
        folder=output/stage;folder.mkdir()
        bootstrap=f'''import os,sys,json,base64,zipfile,io,hashlib,time
from pathlib import Path
print("BOOTSTRAP {stage}: verifying immutable source archive",flush=True)
ROOT=Path('/kaggle/working/better_source');ROOT.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
    for name in archive.namelist():
        if not (ROOT/name).resolve().is_relative_to(ROOT):raise ValueError("unsafe archive path")
    archive.extractall(ROOT)
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name,expected in manifest['files'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError("source checksum mismatch")
sys.path.insert(0,str(ROOT));os.environ['BETTER_SOURCE_MANIFEST']=str(ROOT/'SOURCE_MANIFEST.json')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from parameters.cloud_notebook import CellRecorder
recorder=CellRecorder(CELL_SOURCES,Path('/kaggle/working/cell_evidence'))
with recorder.cell(0):
    print("Source manifest verified",manifest['git_commit'],manifest['campaign_tag'])
    print("Interpreter",sys.executable,sys.version)
    print("Source path",ROOT,"Ledger",manifest['exposure_ledger_sha256'])'''
        setup='''with recorder.cell(1):
    import subprocess,importlib.metadata as metadata
    print("Installing into active Kaggle kernel; keeping managed torch/CUDA",flush=True)
    command=[sys.executable,'-m','pip','install','--disable-pip-version-check','transformers==4.57.6',
        'huggingface-hub==0.36.2','sentencepiece==0.2.2','scikit-learn>=1.4,<2','pillow>=10','pyarrow','nbformat','pytest']
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
    print(result.stdout,flush=True);result.check_returncode()
    for package in ('torch','transformers','huggingface-hub','scikit-learn','numpy','Pillow','pyarrow'):
        print(package,metadata.version(package))
    import torch
    print("Actual CUDA:",torch.cuda.is_available(),"Assigned GPUs:",torch.cuda.device_count())
'''
        if stage=='extract':setup+='    assert torch.cuda.device_count()==2,"Exactly two GPUs required; stopping before extraction"\n'
        run=f'''with recorder.cell(2):
    from parameters.cloud_notebook import execute
    RESULT=execute({stage!r},ROOT,Path('/kaggle/working/results/{stage}/{tag}'))
    print("Stage artifact path:",RESULT,flush=True)'''
        finish='''with recorder.cell(3):
    from parameters.artifacts import write_json
    status=json.loads((RESULT/'status.json').read_text())
    print("Stage completion record:",status)
    receipt={str(path.relative_to(RESULT)):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in RESULT.rglob('*') if path.is_file() and path.stat().st_size<5_000_000}
    write_json(Path('/kaggle/working/artifact_checksums.json'),receipt)
    print("Executed notebook, cell logs, run manifests and checksums preserved.")
    print("Saved-version outputs require independent retrieval; runtime paths alone are not durability proof.")'''
        sources=[bootstrap,setup,run,finish]
        cells=[nbf.v4.new_markdown_cell(f'''# parameters — {stage}

Private campaign {tag}. CPU data preparation, dual-GPU DataParallel extraction,
then CPU nested grouped selection and separately recorded locked final study.
Limits:10,000parents,20GB source downloads,8aggregateGPUhours. Managed Kaggle
Python/runtime; dependencies install in the active kernel. Source identity and
actual package/backend versions print before work. Outputs:
`/kaggle/working/results/{stage}/{tag}`, `cell_evidence/executed.ipynb`, cell logs,
status and artifact checksums. Partial logs and feature shards persist during
work; saved-version retrieval is checked separately. Initial bootstrap messages
before recorder setup and native-process messages may have limited capture.
No codec, natural cross-view identity or operational urgency claim.''')]
        for index,value in enumerate(sources):
            if index==0:value='CELL_SOURCES='+repr(sources)+'\n'+value
            compile(value,'cloud_cell_'+str(index),'exec')
            cells.append(nbf.v4.new_code_cell(value))
        notebook=nbf.v4.new_notebook(cells=cells);notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
        nbf.write(notebook,folder/'notebook.ipynb')
        upstream=[] if stage=='prepare' else ['satyapaladugu/'+slugs['prepare']]
        if stage=='select':upstream+=['satyapaladugu/'+slugs['extract']]
        metadata={'id':'satyapaladugu/'+slugs[stage],'title':'parameters '+stage+' '+tag,
                  'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
                  'enable_gpu':stage=='extract','enable_internet':True,'dataset_sources':[],'competition_sources':[],
                  'kernel_sources':upstream}
        if stage=='extract':metadata['machine_shape']='NvidiaTeslaT4'
        write_json(folder/'kernel-metadata.json',metadata)
    write_json(output/'campaign.json',{'tag':tag,'slugs':slugs,'source_manifest':manifest,
               'limits':{'parents':10000,'download_bytes':20_000_000_000,'aggregate_gpu_hours':8},
               'time_limits':{'prepare':10800,'extract':14400,'select':21600}})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args();print(build(args.output,args.tag))
