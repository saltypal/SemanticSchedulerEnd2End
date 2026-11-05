"""Generate both account notebooks from one canonical source snapshot. Never submits."""
from pathlib import Path
import ast,base64,hashlib,io,json,re,textwrap,zipfile
ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/"Notebooks/SchedulerSuite/paired"
CACHE=json.loads((ROOT/"Notebooks/SchedulerSuite/inputs/attachment.json").read_text())

def markdown(text):return {"cell_type":"markdown","metadata":{},"source":text.strip().splitlines(keepends=True)}
def cell(text):return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(text).strip().splitlines(keepends=True)}
def make_snapshot():
    old=json.loads((ROOT/"Notebooks/SchedulerSuite/snapshot_template.ipynb").read_text())
    code="".join(old["cells"][2]["source"]);blob=base64.b64decode(re.search("base64.b64decode\\('([^']+)'\\)",code)[1])
    previous=zipfile.ZipFile(io.BytesIO(blob))
    names=set(previous.namelist())|{"Scheduler/training/study/paired_runtime.py","Scheduler/training/study/requirements-kaggle.txt","Scheduler/training/study/summarize_saved.py","Scheduler/training/study/merge_paired_runs.py","Scheduler/training/stream_suite/profile_exports.py",
        "experiments/master_study/tests/test_paired_notebooks.py"}
    names.add("Scheduler/training/study/requirements-kaggle.txt")
    names.update(str(p.relative_to(ROOT)).replace("\\", "/") for owner in ("stream_suite", "study") for p in (ROOT/"Scheduler/training"/owner).glob("*.py"))
    out=io.BytesIO();hashes={}
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):
            path=ROOT/name
            if not path.exists() and name not in previous.namelist():continue
            data=path.read_bytes() if path.is_file() else previous.read(name)
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=zipfile.ZipInfo(name,(2026,11,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    return out.getvalue(),hashes

def build(profile,blob,hashes,*,persist=True):
    owner="satyapaladuguplus" if profile=="wsl" else "satyapaladugu"
    title="WSL — ranking and VLM supervision" if profile=="wsl" else "Windows — reinforcement learning"
    allocation=("GPU0:10 supervised D methods. GPU1:Qwen candidate/prefix/replay labels,7 P students,2 feedback RL methods, each gated."
        if profile=="wsl" else "GPU0:Double DQN. GPU1:Deep Sets initialization, PPO, TD-PER and warmstarted PPO.")
    intro=f"""
# {title}
**Prepare/read first. This notebook has not been submitted or run by Codex.**

Launch account: **{owner}**. Both notebooks execute in managed **Kaggle Linux**; WSL/Windows identify the launch account, not the cloud operating system.

{allocation}

**Single trial:** seed1729, at most50 supervised epochs with patience8;20,000 RL decisions per method.
**Clock:** stop learner work at10.5hours from first cell, finish reports by11.5hours, leaving30minutes before Kaggle's observed12hour kill.
Setup/model loading count against that clock. A partial checkpoint is reported as partial, not convergence.

**Real scope:** corrected CrisisMMD silver-priority D study. General multi-domain urgency and independent mission-E training remain blocked by missing independent labels.
The ordinary image pool is not relabeled Low. Teacher controls, actual swap consistency, strict label support and reward-validation gates are mandatory.
Qwen2.5-VL-7B at immutable revisioncc594898137f460bfe9f0759e9844b3ce807cfb5 uses NF4; no Qwen fine-tuning.
SQ-GAN is frozen, digital-byte source coding. Scheduler learning uses synthetic unit service; codec timing is separate.

## Required private inputs
1. Mini_dataset_6000.tar (SHA2560f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c).
2. Scheduler_shared_cache_20261102.tar (SHA256{CACHE['sha256']}). It supplies identical frozen embeddings/mission vectors and SQ-GAN reference source.
3. WSL only:attach the published SQ-GAN teammate-2 output (existing semantic-notebook-2). No codec asset/training is required by Windows RL.

The Windows account currently cannot read the plus account's private cache. Attach private copies under Windows; do not make inputs public.
Neither notebook waits for the other. The Windows Deep Sets fit is an initialization prerequisite/frozen PPO comparator; do not count it as another independent WSL training seed.

Both notebooks profile exported policies on CPU:queue sizes4/16/64/256, selection milliseconds, parameter/weight size and supported-operator FLOPs. These FLOPs exclude some operations and are not a total compute claim. WSL also runs the baseline UE-rate study when training-window time remains.
"""
    cells=[markdown(intro)]
    setup=f"""
SESSION_START=__import__('time').monotonic()
import os,sys,json,time,platform,hashlib,base64,zipfile,tarfile,io,shutil,datetime,contextlib,traceback
from pathlib import Path
os.environ.update({{name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}})
PROFILE={profile!r}
RESUME_INPUT=''  # Optional path to prior RUN folder attached read-only under /kaggle/input.
TRAINING_STOP=SESSION_START+10.5*3600
FINAL_STOP=SESSION_START+11.5*3600
SOURCE_SHA={hashlib.sha256(blob).hexdigest()!r}
WORK=Path('/kaggle/working')
if RESUME_INPUT:
    previous=Path(RESUME_INPUT)
    prior=json.loads((previous/'run_config.json').read_text())
    if prior['profile']!=PROFILE or prior['source_snapshot_sha256']!=SOURCE_SHA:
        raise ValueError('Resume requires the same profile and immutable source snapshot')
    RUN=(WORK/prior['relative_run']).resolve()
    if not RUN.is_relative_to(WORK/'experiments/scheduler_pair/runs') or RUN.exists():
        raise ValueError('Resume destination must be a new, verified working run path')
    shutil.copytree(previous,RUN)
else:
    RUN=WORK/'experiments/scheduler_pair/runs'/(PROFILE+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    RUN.mkdir(parents=True,exist_ok=False)
SESSION=RUN/'sessions'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
SESSION.mkdir(parents=True,exist_ok=False)
(SESSION/'logs').mkdir()
CELL_RECORDS=[]
class Tee:
    def __init__(self,console,file):self.console,self.file=console,file
    def write(self,text):self.console.write(text);self.file.write(text);self.file.flush();return len(text)
    def flush(self):self.console.flush();self.file.flush()
    def isatty(self):return False
def persist_cells():
    (SESSION/'cells.json').write_text(json.dumps(CELL_RECORDS,indent=2))
    exported=[]
    for r in CELL_RECORDS:
        outputs=[]
        log=Path(r['log'])
        if log.exists():outputs.append({{'output_type':'stream','name':'stdout','text':log.read_text(errors='replace').splitlines(keepends=True)}})
        exported.append({{'id':r['stage'],'cell_type':'code','metadata':{{'stage':r['stage'],'status':r['status']}},'source':r['source'].splitlines(keepends=True),'execution_count':r['execution_count'],'outputs':outputs}})
    (SESSION/'executed_notebook.ipynb').write_text(json.dumps({{'nbformat':4,'nbformat_minor':5,'metadata':{{'capture':'cell source/stdout/stderr and errors; rich tables/figures saved separately; platform notebook is authoritative rich capture'}},'cells':exported}},indent=1))
@contextlib.contextmanager
def stage(name):
    shell=get_ipython()
    rec={{'stage':name,'execution_count':shell.execution_count,'source':shell.history_manager.input_hist_raw[-1],
         'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','log':str(SESSION/'logs'/(name+'.log'))}}
    CELL_RECORDS.append(rec);persist_cells();started=time.monotonic()
    with Path(rec['log']).open('w',buffering=1) as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        print('START',name,'run',RUN,flush=True)
        try:
            yield
        except BaseException:
            rec['status']='failed';traceback.print_exc();raise
        else:rec['status']='complete'
        finally:
            rec['seconds']=time.monotonic()-started
            print('END',name,rec['status'],'seconds',round(rec['seconds'],2),flush=True)
            persist_cells()
with stage('01_session'):
    print({{'profile':PROFILE,'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),
           'training_stop_hours':10.5,'finalization_stop_hours':11.5,'platform_limit_hours':12,
           'host_environment':'managed Kaggle, not a transferred local venv'}},flush=True)
"""
    cells += [markdown('## 1. Session, budget and recovery\nSet RESUME_INPUT only to a complete prior run folder. Source/observations/schedules must match; otherwise start a new run. No files are deleted.'),cell(setup)]
    source=f"""
with stage('02_pinned_source'):
    payload=base64.b64decode({base64.b64encode(blob).decode()!r})
    if hashlib.sha256(payload).hexdigest()!=SOURCE_SHA:raise ValueError('source capsule mismatch')
    (RUN/'source_snapshot.zip').write_bytes(payload)
    SOURCE=RUN/'source';SOURCE.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for member in z.infolist():
            if not (SOURCE/member.filename).resolve().is_relative_to(SOURCE):raise ValueError('unsafe source member')
        z.extractall(SOURCE)
    source_hashes={hashes!r}
    for name,expected in source_hashes.items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=expected:raise ValueError('source mismatch:'+name)
    sys.path.insert(0,str(SOURCE))
    import importlib.util
    spec=importlib.util.spec_from_file_location('runtime_bootstrap',SOURCE/'Scheduler/training/stream_suite/runtime_bootstrap.py')
    bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap);bootstrap.install(SOURCE)
    cfg={{'profile':PROFILE,'relative_run':str(RUN.relative_to(WORK)),'source_snapshot_sha256':SOURCE_SHA,
         'source_hashes':source_hashes,'seed':1729,'epochs_max':50,'rl_decisions_max':20000}}
    (RUN/'run_config.json').write_text(json.dumps(cfg,indent=2))
    print('Verified source files',len(source_hashes),flush=True)
"""
    cells += [markdown('## 2. Frozen source snapshot\nBoth notebooks contain the same canonical source capsule. This is a disposable execution copy, not a second development codebase.'),cell(source)]
    inputs=f"""
with stage('03_inputs_and_cache'):
    archives=[p for name in ('Mini_dataset_6000.tar','Mini_dataset_6000.tar.bin') for p in Path('/kaggle/input').rglob(name)]
    shared=[p for name in ('Scheduler_shared_cache_20261102.tar','Scheduler_shared_cache_20261102.tar.bin') for p in Path('/kaggle/input').rglob(name)]
    if len(archives)!=1 or len(shared)!=1:raise ValueError('Attach exactly one mini archive and one shared cache; found '+str((len(archives),len(shared))))
    ARCHIVE=archives[0]
    from Scheduler.training.study.paired_runtime import digest
    if digest(ARCHIVE)!='0f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c':raise ValueError('image archive mismatch')
    if digest(shared[0])!={CACHE['sha256']!r}:raise ValueError('shared cache mismatch')
    COMMON=RUN/'shared'
    if not (COMMON/'shared_manifest.json').exists():
        COMMON.mkdir(exist_ok=True)
        with tarfile.open(shared[0]) as tar:
            for item in tar.getmembers():
                if item.issym() or item.islnk() or not (COMMON/item.name).resolve().is_relative_to(COMMON):raise ValueError('unsafe cache member')
            tar.extractall(COMMON,filter='data')
    common_manifest=json.loads((COMMON/'shared_manifest.json').read_text())
    for name,expected in common_manifest['files'].items():
        if digest(COMMON/name)!=expected:raise ValueError('cache mismatch:'+name)
    VECTORS=COMMON/'vectors/vectors.npz'
    MISSIONS=COMMON/'missions/mission_vectors.npy'
    identity={{'archive':str(ARCHIVE),'archive_sha256':digest(ARCHIVE),
        'shared_sha256':digest(shared[0]),'vectors_sha256':digest(VECTORS),'mission_vectors_sha256':digest(MISSIONS)}}
    if RESUME_INPUT:
        previous_identity=json.loads((RUN/'input_manifest.json').read_text())
        if previous_identity!=identity:raise ValueError('Resume input identity/path mismatch')
    (RUN/'input_manifest.json').write_text(json.dumps(identity,indent=2))
    print('Image archive and frozen caches verified. No SPI/scene/CLIP extraction is rerun.',flush=True)
"""
    cells += [markdown('## 3. Input identity\nFail before GPU work if either private input is absent or wrong. The same cache avoids producer differences across accounts.'),cell(inputs)]
    env="""
with stage('04_environment'):
    import subprocess
    dependencies=[line.strip() for line in (SOURCE/'Scheduler/training/study/requirements-kaggle.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
    command=[sys.executable,'-m','pip','install',*dependencies]
    print('Install into active kernel:',command,flush=True)
    subprocess.run(command,check=True,timeout=600)
    import torch,numpy,lightgbm,transformers
    actual={'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'lightgbm':lightgbm.__version__,
            'transformers':transformers.__version__,'cuda':torch.cuda.is_available(),
            'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    if RESUME_INPUT:
        previous_env=[p for p in RUN.glob('sessions/*/environment.json') if p.parent!=SESSION]
        if not previous_env:raise ValueError('Resume has no prior environment identity')
        prior_environment=json.loads(sorted(previous_env)[-1].read_text())
        for key in ('python','torch','numpy','lightgbm','transformers'):
            if prior_environment[key]!=actual[key]:raise ValueError('Resume environment differs:'+key+'; use an explicitly separate weights-only experiment instead')
    (SESSION/'environment.json').write_text(json.dumps(actual,indent=2));print(actual,flush=True)
    if torch.cuda.device_count()!=2:raise RuntimeError('This allocation requires two CUDA devices; refuse CPU fallback')
    for i in range(2):
        probe=torch.ones(1,device='cuda:'+str(i));assert probe.device.index==i
        torch.cuda.synchronize(i)
    del probe
    print('Managed Torch/CUDA retained; no core-framework replacement. Fresh kernel required if dependencies were previously imported.',flush=True)
"""
    cells += [markdown('## 4. Managed environment and real hardware\nBoth require Kaggle T4×2 and internet for pinned model/config downloads. Package setup counts toward the session budget.'),cell(env)]
    check="""
with stage('05_contracts_and_reference_scope'):
    from Scheduler.training.stream_suite.input_preflight import check
    coverage=check(ARCHIVE,RUN/'reference_preflight',['crisismmd'])
    if coverage['status']=='blocked_references':raise RuntimeError('Even the scoped priority cohort is unavailable')
    print('Only CrisisMMD has D references. Ordinary-domain and independent mission-E claims remain blocked.',flush=True)
    command=[sys.executable,'-m','pytest','-q','-ra','-c','/dev/null','--rootdir',str(SOURCE),'--confcutdir',str(SOURCE),
             str(SOURCE/'experiments/master_study/tests/test_scientific_repairs.py'),
             str(SOURCE/'experiments/master_study/tests/test_controller.py')]
    subprocess.run(command,cwd=SOURCE,env={**os.environ,'PYTHONPATH':str(SOURCE)},check=True,timeout=600)
    spec=importlib.util.spec_from_file_location('sq01_packet_preflight',COMMON/'sqgan_reference/src/bitstream.py')
    packet_format=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet_format)
    test_packet=packet_format.encode_packet([0,511],[0,1023],[1],[17])
    assert packet_format.decode_packet(test_packet)==[([0,511],[0,1023]),([1],[17])]
    corrupted=bytearray(test_packet);corrupted[-1]^=1
    try:packet_format.decode_packet(bytes(corrupted))
    except ValueError:pass
    else:raise AssertionError('SQ01 corruption was accepted')
    print('Packaged regression subset and SQ01 CRC checks passed. No analog channel is used.',flush=True)
"""
    cells += [markdown('## 5. Numerical/lifecycle tests and reference audit\nCheck both CUDA allocations, SQ01 digital-packet CRC, scientific repair regressions and controller execution. No scheduler real-data fitting occurs in this cell.'),cell(check)]
    if profile=="wsl":
        codec="""
import contextlib,tempfile,shutil
with stage('06_frozen_sqgan_payload_check'), contextlib.ExitStack() as cleanup:
    from Codec.integration.sqgan_payload import SQGANDigitalCodec
    candidates=[p.parent for p in Path('/kaggle/input').rglob('bundle.json')
                if p.parent.name=='20261028T130100Z-eea4f3aa' and p.parent.parent.name=='SQ-GAN']
    if not candidates:
        flat=list(Path('/kaggle/input').rglob('scheduler_sqgan_flat_manifest.json'))
        if len(flat)!=1:raise ValueError('Attach exactly one verified SQ-GAN bundle or flat package')
        mapping=json.loads(flat[0].read_text())
        if mapping.get('schema')!='scheduler_sqgan_flat_v1' or mapping.get('run_id')!='20261028T130100Z-eea4f3aa':
            raise ValueError('Unexpected flat SQ-GAN package identity')
        source=flat[0].parent
        manifest_path=(source/mapping['bundle_manifest']).resolve()
        if manifest_path.parent!=source.resolve() or digest(manifest_path)!='8b482d80a1adc7dde7a2725461a17346aca49d6628d4f78f88f1619df7448c21':
            raise ValueError('Original SQ-GAN bundle manifest mismatch')
        original=json.loads(manifest_path.read_text())
        required=sum(entry['bytes'] for entry in original['files'])
        if shutil.disk_usage('/kaggle/working').free<required+512*1024*1024:
            raise ValueError('Insufficient Kaggle disk for verified codec materialization')
        temporary=Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix='sqgan-input-',dir='/kaggle/working')))
        bundle=temporary/'SQ-GAN'/mapping['run_id'];bundle.mkdir(parents=True)
        shutil.copyfile(manifest_path,bundle/'bundle.json')
        for entry in original['files']:
            member=mapping['files'][entry['path']]
            if isinstance(member,dict):
                if member.get('bytes')!=entry['bytes'] or member.get('sha256')!=entry['sha256']:
                    raise ValueError('Flat entry disagrees with original bundle manifest')
                filename=member.get('file')
            else:
                filename=member
            if not isinstance(filename,str):raise ValueError('Flat entry filename must be a string')
            uploaded=(source/filename).resolve()
            destination=(bundle/entry['path']).resolve()
            if uploaded.parent!=source.resolve() or not destination.is_relative_to(bundle):
                raise ValueError('Unsafe flat SQ-GAN member')
            if uploaded.stat().st_size!=entry['bytes']:
                raise ValueError('Incomplete uploaded SQ-GAN member: '+entry['path'])
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(uploaded,destination)
            print('Materialized codec input',entry['path'],entry['bytes'],flush=True)
        candidates=[bundle]
    if len(candidates)!=1:raise ValueError('Attach the selected teammate2 SQ-GAN output; found '+str(len(candidates)))
    # Verify one image, then five more. Cache each complete packet once.
    import tarfile,numpy as np
    from PIL import Image,ImageOps
    codec=SQGANDigitalCodec(bundle=candidates[0],source_root=COMMON/'sqgan_reference',device='cuda:0')
    destination=RUN/'sqgan_payloads';destination.mkdir(exist_ok=True)
    records=[]
    with tarfile.open(ARCHIVE) as tar:
        index=json.load(tar.extractfile('Mini_dataset/portable_image_index.json'))
        for j,item in enumerate(index[:6]):
            with Image.open(tar.extractfile('Mini_dataset/'+item['relative_path'])) as opened:
                image=ImageOps.exif_transpose(opened).convert('RGB')
                width,height=image.size;factor=min(512/width,256/height)
                resized=image.resize((max(1,round(width*factor)),max(1,round(height*factor))))
                canvas=Image.new('RGB',(512,256),(127,127,127));offset=((512-resized.width)//2,(256-resized.height)//2);canvas.paste(resized,offset)
            x=torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1).float()[None].to('cuda:0')/255
            torch.cuda.synchronize();start=time.perf_counter();semantic=codec.semantic_map(x)
            torch.cuda.synchronize();teacher_seconds=time.perf_counter()-start
            start=time.perf_counter();packet=codec.encode(x,semantic);torch.cuda.synchronize();encode_seconds=time.perf_counter()-start
            path=destination/(str(j)+'.sq01');path.write_bytes(packet)
            start=time.perf_counter();decoded=codec.decode(packet);torch.cuda.synchronize();decode_seconds=time.perf_counter()-start
            Image.fromarray(np.round(decoded[0].detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)).save(destination/(str(j)+'_reconstruction.png'))
            records.append({'image_id':item['image_id'],'payload_bytes':len(packet),'payload_sha256':digest(path),
                            'teacher_seconds':teacher_seconds,'encode_seconds':encode_seconds,'decode_seconds':decode_seconds,
                            'geometry':{'source_width':width,'source_height':height,'scale':factor,'offset':offset},
                            'checkpoint_sha256':codec.checkpoint_sha256})
            (destination/'manifest.json').write_text(json.dumps({'scope':'frozen source codec; not lossless/radio/final semantic validation','records':records},indent=2))
            print('SQGAN',j+1,'of6','bytes',len(packet),'decode_seconds',round(decode_seconds,4),flush=True)
    del codec,x,semantic,decoded
    import gc;gc.collect();torch.cuda.empty_cache()
"""
        cells += [markdown('## 6. SQ-GAN: one image, then five\nFrozen checkpoint only. Record real encode/decode times and byte payloads; these do not replace synthetic service times in learning. Native256×512 preprocessing is recorded. This small check does not establish p95 latency or disaster semantic quality.'),cell(codec)]
    else:
        cells += [markdown('## 6. Codec boundary\nThis notebook trains receiver policies, not the codec. SQ-GAN is the selected digital codec. The WSL notebook owns the bounded actual payload check; this RL study uses declared synthetic unit service and does not claim real decoder latency.')]
    train="""
with stage('07_training_and_conditional_labels'):
    from Scheduler.training.study.paired_runtime import PairRun
    config={'profile':PROFILE,'run':str(RUN),'archive':str(ARCHIVE),'vectors':str(VECTORS),'mission_vectors':str(MISSIONS),
            'training_stop_monotonic':TRAINING_STOP,'final_stop_monotonic':FINAL_STOP}
    (SESSION/'resolved_execution.json').write_text(json.dumps(config,indent=2))
    runner=PairRun(config)
    try:runner.execute()
    finally:
        runner.report()
        print('Training stage exited; consult per-method state, not only process exit.',flush=True)
"""
    cells += [markdown(f"""## 7. Assigned training
{allocation}
Baseline policies are evaluated before learned methods. Checkpoints contain optimizer/progress/RNG state and are atomically replaced. A cutoff marks unfinished methods partial. VLM label failure blocks only dependent methods; no invented labels are substituted."""),cell(train)]
    reports="""
with stage('08_tables_figures_and_recovery'):
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    for path in sorted((RUN/'reports').glob('*/delay_comparison.csv')):
        table=pd.read_csv(path)
        if table.empty:
            print('No shared-D rows for',path.parent.name,'; see its method/status report.',flush=True);continue
        print('Shared D validation:',path.parent.name,flush=True);display(table)
        table.to_html(SESSION/(path.parent.name+'_delay.html'),index=False)
        ax=table.set_index('method')['mean_wait_penalty'].plot.barh(figsize=(9,max(4,.3*len(table))))
        ax.set_xlabel('D normalized waiting penalty (lower better; synthetic time)')
        ax.set_title(path.parent.name+' — development/silver results')
        ax.figure.tight_layout();ax.figure.savefig(SESSION/(path.parent.name+'_delay.png'),dpi=150);plt.show();plt.close(ax.figure)
    status=json.loads((RUN/'RUN_STATUS.json').read_text())
    display(pd.DataFrame.from_dict(json.loads((RUN/'method_ledger.json').read_text()),orient='index'))
    print('RUN STATUS',status,flush=True)
    print('Runtime/partial-FLOPs profiles:',list((RUN/'profiles').glob('*.json')),flush=True)
    print('Important:Critical waits/misses are in critical_safeguards.json; aggregate D is not a Critical guarantee.',flush=True)
    print('Retrieve the ENTIRE run folder, including source/cache identity, session logs, raw teacher records, optimizer checkpoints and metrics.',flush=True)
    print('Resume:attach that run under /kaggle/input and set RESUME_INPUT. Preserve original input mount names and source snapshot.',flush=True)
    (SESSION/'artifact_index.json').write_text(json.dumps({str(p.relative_to(RUN)):{'bytes':p.stat().st_size}
          for p in RUN.rglob('*') if p.is_file()},indent=2))
    print('Incremental executed_notebook.ipynb is in',SESSION,'; platform executed notebook/rich output must also be retrieved.',flush=True)
"""
    cells += [markdown('## 8. Results and recovery\nDevelopment-only comparisons, class-delay/miss safeguards, raw status tables and plots. Do not rank feedback versus D from different traces. One seed is a trial, not stability evidence. Keep failures and blocked methods visible.'),cell(reports)]
    n={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"scheduler_profile":profile},"cells":cells}
    for i,c in enumerate(cells):
        c["id"]=f"{profile}-{i:02d}"
        if c["cell_type"]=="code":compile("".join(c["source"]),f"{profile}-cell-{i}","exec")
    if not persist:return n
    folder=DEST/profile;folder.mkdir(parents=True,exist_ok=True)
    filename="Scheduler_WSL_Ranking_VLM.ipynb" if profile=="wsl" else "Scheduler_Windows_RL.ipynb"
    path=folder/filename;path.write_text(json.dumps(n,indent=1))
    if path.stat().st_size>950000:raise ValueError("Kaggle source size too large")
    metadata={"id":("satyapaladuguplus/semantic-decoder-scheduler-20261030" if profile=="wsl" else "satyapaladugu/semantic-decoder-scheduler-rl"),
      "title":("Semantic Decoder Scheduler 20261030" if profile=="wsl" else "Semantic Decoder Scheduler RL"),
      "code_file":filename,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,
      "dataset_sources":[],"competition_sources":[],"kernel_sources":(["satyapaladuguplus/semantic-image-pool-scene-cache-20261030","satyapaladuguplus/semantic-notebook-2"] if profile=="wsl" else [])}
    meta_path=folder/"kernel-metadata.json"
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        # Rebuilding source must preserve verified account identity and private attachments.
        for key in ("id","dataset_sources","kernel_sources","competition_sources"):
            if key in previous:metadata[key]=previous[key]
    meta_path.write_text(json.dumps(metadata,indent=2))
    return {"profile":profile,"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"code_cells":sum(c["cell_type"]=="code" for c in cells),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_snapshot_sha256":hashlib.sha256(blob).hexdigest(),"submitted":False}
def main():
    blob,hashes=make_snapshot();results=[build(profile,blob,hashes) for profile in ("wsl","windows")]
    (DEST/"notebook_manifest.json").write_text(json.dumps({"notebooks":results,"source_hashes":hashes,"shared_cache":CACHE},indent=2))
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
