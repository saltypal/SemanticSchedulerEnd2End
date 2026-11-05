"""One readable multi-source cache notebook; no full archives or local image downloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import nbformat as nb

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experiments/image_pool'
config=json.loads((HERE/'config.json').read_text())
selected=json.loads((ROOT/'experiments/full_medic_labeling/runs/crisismmd4000_selection/selected_1000.json').read_text())
def key(r):return hashlib.sha256(('29:'+r['image_id']).encode()).hexdigest()
crisis=sorted(selected,key=key)
crisis=[{k:r[k] for k in ('image_id','image_path','split','tweet_id')} for r in crisis]
files=['parameters/scene_vocabulary.py','parameters/scene_arrays.py',
       'experiments/scene_array_cache/cache.py','experiments/full_medic_labeling/costs.py',
       'experiments/spi_classifier_repair/medic_campaign.py','experiments/spi_classifier_repair/head_matrix.py',
       'experiments/spi_classifier_repair/medic_training_setup.py','experiments/spi_classifier_repair/resume_store.py',
       'experiments/full_medic_labeling/balancing.py',
       'parameters/spi.py','parameters/artifacts.py','parameters/contracts.py','parameters/spi_labels.py','parameters/data.py',
       'experiments/image_pool/sources.py','experiments/image_pool/student.py','experiments/image_pool/runtime.py']
snap={p:hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode()).hexdigest() for p in files}
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Diverse image pool — scene arrays and frozen SPI predictions

**One shared pipeline.** Cloud-only tiny samples, individual images or already accessible selected frames.
Store image-dependent scene evidence and experimental student SPI once. The simulator later computes
mission relevance, context centroids and causal UE novelty. No VLM labeling or student training here.

```
Small source selections → source/hash/clip groups → frozen scene detector + frozen SPI student
 → separate caches → source-qualified scheduler JSONL + scene arrays + offline references
```

The real run requires the **completed 4,000-image training output**, then chooses by development score.
It never silently substitutes the exposed 1,000-image model. Predictions are not native urgency truth.
''')
md('''## 0. Environment and bounded execution

Canonical project: SemanticSchedulerEnd2End. Laptop uses its existing `.venv`; Kaggle has its own
managed kernel. Fixture mode executes software tests only and cannot download any media.
Real mode runs only on Kaggle. Keep managed torch/CUDA. Source snapshots below are readable and hashed.
''')
code('''import os,sys,time,json,hashlib,subprocess
from pathlib import Path
print('SETUP START',sys.executable,sys.version,flush=True)
IS_KAGGLE=Path('/kaggle/working').exists()
MODE=os.environ.get('IMAGE_POOL_MODE','real' if IS_KAGGLE else 'fixture')
SCENE_ONLY=os.environ.get('IMAGE_POOL_SCENE_ONLY','0')=='1'
if MODE not in ('real','fixture'):raise ValueError('unknown mode')
if MODE=='real' and not IS_KAGGLE:raise RuntimeError('real acquisition is cloud-only')
PROJECT=Path('/kaggle/working') if IS_KAGGLE else next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'parameters/scene_arrays.py').is_file())
RUN=Path(os.environ.get('IMAGE_POOL_RESUME_DIR',str(PROJECT/'experiments/image_pool/runs'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))))
RUN.mkdir(parents=True,exist_ok=True)
if IS_KAGGLE:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps','ultralytics==8.3.237',
       'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==5.5.0','huggingface-hub==1.5.0',
       'scikit-learn==1.7.2','timm','joblib','ftfy','regex','tqdm','psutil==7.2.2'])
BOOT=RUN/'source_snapshot'
for package in ('parameters','experiments','experiments/scene_array_cache','experiments/spi_classifier_repair',
                'experiments/full_medic_labeling','experiments/image_pool'):
    (BOOT/package).mkdir(parents=True,exist_ok=True)
    (BOOT/package/'__init__.py').write_text('# Runtime snapshot package.\\n')
os.chdir(BOOT);sys.path.insert(0,str(BOOT))
print('SETUP END',MODE,'output',RUN,flush=True)
''')
md('## 1. Exact shared source snapshot — no second implementation')
for f in files:code('%%writefile '+f+'\n'+(ROOT/f).read_text(encoding='utf8'))
code('''from experiments.scene_array_cache.cache import (stage,write_json,file_hash,code_hash,environment,
    Detector,WEIGHTS_SHA256,cache_all,profile,report)
from experiments.image_pool.sources import build_pool,group_pool
from experiments.image_pool.student import freeze_student,Student
from experiments.image_pool.runtime import cache_student,join_runtime,profile_spi
from parameters.scene_arrays import REGISTRY_SHA256
SOURCE_HASHES='''+repr(snap)+'''
with stage(RUN,'01_source'):
    for f,h in SOURCE_HASHES.items():
        if code_hash(BOOT/f)!=h:raise ValueError('runtime source differs: '+f)
    write_json(RUN/'source.json',{'commit':''' +repr(commit)+''','source_hashes':SOURCE_HASHES,'scope':'selected dirty-checkout source files'})
    write_json(RUN/'environment.json',environment());print(environment(),flush=True)
''')
md('''## 2. Fixed source caps and frozen training model gate

Core sources: 200 each COCO, OpenImages, Places365 and CrisisMMD. BDD100K/DoTA/UCF-Crime:
at most five clips × ten ordered frames each, only from accessible managed selections. Specialist
sources: 50 each. Total cap 1,150 image/frame records and 2GiB new data transfers, excluding model
weights. Places uses at most a 64MiB official TAR prefix (a convenience sample, not uniform sampling).

Set `IMAGE_POOL_TRAINING_ROOT` to the completed 4,000-image run mounted in Kaggle. Its `progress.json`,
`selected_manifest.json` and `comparison/` artifacts must be present. No bulk image acquisition starts
before the SPI model-selection gate passes. The current running SPI job is never interrupted.
''')
code('CONFIG='+repr(config)+'''
CRISIS_MANIFEST='''+repr(crisis)+'''
with stage(RUN,'02_model_selection'):
    write_json(RUN/'config.json',CONFIG)
    if SCENE_ONLY:
        BUNDLE=None
        print('Scene-only run: SPI deferred until completed training artifacts exist',flush=True)
    elif MODE=='real':
        root=os.environ.get('IMAGE_POOL_TRAINING_ROOT')
        if not root:raise RuntimeError('Completed4000 SPI training not supplied. Mount it and set IMAGE_POOL_TRAINING_ROOT; no historical fallback.')
        TRAINING_ROOT=Path(root)
        progress=json.loads((TRAINING_ROOT/'progress.json').read_text())
        training_cohort=json.loads((TRAINING_ROOT/'selected_manifest.json').read_text())
        if progress.get('processed_rows')!=4000 or not progress.get('coverage_complete') or len(training_cohort)!=4000:
            raise RuntimeError('4000image labeling/training cohort incomplete')
        receipt={'status':'complete','cohort_size':4000,'training_root':str(TRAINING_ROOT),
                 'cohort_sha256':file_hash(TRAINING_ROOT/'selected_manifest.json'),
                 'progress_sha256':file_hash(TRAINING_ROOT/'progress.json'),
                 'comparison_status_sha256':file_hash(TRAINING_ROOT/'comparison/study_status.json'),
                 'claim':'completed experimental comparison, not validated urgency reliability'}
        write_json(RUN/'training_receipt.json',receipt)
        BUNDLE=freeze_student(TRAINING_ROOT/'comparison',RUN/'student',RUN/'training_receipt.json')
        print('VALIDATION SELECTED STUDENT',json.loads(BUNDLE.read_text()),flush=True)
    else:
        BUNDLE=None;print('Fixture mode: no model selection or inference claim',flush=True)
''')
md('''## 3. Cloud-only small source acquisition

Optional managed inputs: set `IMAGE_POOL_MANAGED_MANIFESTS` to a JSON file mapping source names to
JSONL manifests. Required per row: source_id, image_path, source_revision, source_split, source_page.
Temporal rows additionally need actual clip_id/frame_index, and original-video parent_id where known.
Licensed ImageNet/xBD inputs need recorded legitimate access. No archive-download fallback is used.
Missing access is saved as input_required/unavailable, not treated as a successful source.

Published source categories never become priority labels. References remain outside runtime records.
''')
code('''with stage(RUN,'03_acquisition'):
    if MODE=='real':
        managed_file=os.environ.get('IMAGE_POOL_MANAGED_MANIFESTS')
        MANAGED=json.loads(Path(managed_file).read_text()) if managed_file else {}
        ROWS,REFERENCES,SOURCE_STATUS=build_pool(CONFIG,RUN/'data',CRISIS_MANIFEST,MANAGED,cloud=True)
        if not ROWS:raise RuntimeError('no source images acquired; see source_status.json')
        ROWS=group_pool(ROWS,CONFIG['seed'])
    else:
        from PIL import Image
        ROWS=[];REFERENCES=[]
        for i in range(12):
            path=RUN/('fixture_'+str(i)+'.png');Image.new('RGB',(32,32),(i*10,0,0)).save(path)
            ROWS.append({'image_id':'fixture:'+str(i),'image_path':str(path),'image_sha256':file_hash(path),
                         'dataset':'fixture','source_id':str(i),'source_revision':'software-fixture',
                         'source_split':'train','parent_id':str(i),'role':'development','parameter_training_exposed':True})
        ROWS=group_pool(ROWS);SOURCE_STATUS={'fixture':{'status':'complete','selected':12}}
    write_json(RUN/'pool_manifest.json',ROWS)
    write_json(RUN/'references.json',REFERENCES)
    write_json(RUN/'source_status.json',SOURCE_STATUS)
    print('POOL',len(ROWS),'source states',SOURCE_STATUS,flush=True)
''')
md('''## 4. Scene arrays — identical image IDs, explicit 93-class dictionary

Uses the same scene implementation as the scene-only notebook: counts/areas/masks, raw boxes,
status, producer identity, full precision and quantized wire features. This does not infer relationships,
distance or urgency. Cached image inputs live in the cloud pool; no second media download is required.
''')
code('''class FixtureScene:
    signature={'fixture':True,'registry':REGISTRY_SHA256}
    producer_sha256='1'*64
    def sync(self):pass
    def predict(self,image):return []

with stage(RUN,'04_scene_cache'):
    if MODE=='real':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('assignedGPUrequired')
        from ultralytics.utils.downloads import attempt_download_asset
        WEIGHTS=Path(attempt_download_asset(str(RUN/'yolov8s-worldv2.pt')))
        DETECTOR=Detector(WEIGHTS,'cuda:0')
        write_json(RUN/'scene_producer.json',DETECTOR.signature)
        # Exactly10 development-image forward calls before bulk; no model-adjustment sweep.
        from experiments.scene_array_cache.cache import extract
        warm=[r for r in ROWS if r['split']=='train'][:10]
        for r in warm:
            result,_=extract(r['image_path'],r['image_id'],r['image_sha256'],DETECTOR)
            if result['status']!='ok':raise RuntimeError('scene preflightfailed: '+result['failure_reason'])
    else:DETECTOR=FixtureScene();WEIGHTS=None
    SCENE_PROGRESS=cache_all(ROWS,DETECTOR,RUN/'scene',allow_download=False)
    print('SCENE',SCENE_PROGRESS,flush=True)
    del DETECTOR
''')
md('''## 5. SPI student predictions — image input, exact matching encoder and head

The winning head consumes its own trained encoder features. Do not feed CLIP vectors into a SigLIP
head. Labels are outputs of the student, never copied from VLM targets or dataset categories. Preserve
the class order Low/Medium/High/Critical. Decision scores are not calibrated probabilities. Cross-domain
urgency validity remains unestablished, including normal photos, screenshots and surveillance scenes.
''')
code('''class FixtureStudent:
    identity_hash='2'*64
    def predict(self,path):return {'level':'Low','available':True,'reference_type':'software_fixture_not_a_model'}

with stage(RUN,'05_spi_cache'):
    if SCENE_ONLY:
        SPI_FILE=RUN/'spi_pending.jsonl'
        with SPI_FILE.open('w',encoding='utf8') as stream:
            for r in ROWS:
                stream.write(json.dumps({'image_id':r['image_id'],'image_sha256':r['image_sha256'],
                    'spi':{'level':None,'available':False,'status':'pending','reason':'student_training_not_complete'}})+'\\n')
    else:
        if MODE=='real':SPI=Student(BUNDLE,'cuda:0')
        else:SPI=FixtureStudent()
        SPI_FILE=cache_student(ROWS,SPI,RUN/'spi')
        del SPI
    import gc;gc.collect()
    if MODE=='real':torch.cuda.empty_cache()
''')
md('''## 6. Identity-checked scheduler handoff

Runtime JSONL contains scene measurements, student SPI, source IDs/hashes, grouping and observed
frame identities. References stay separate. No fixed relevance, novelty, context centroids, synthetic
timestamps or locations are manufactured. Parent/video and exact-copy groups remain in one role.
FloodNet and any exact duplicate components stay transfer-only, excluded from scheduler fitting.
Near-duplicate and natural cross-view correspondence are not established by this cache.
''')
code('''with stage(RUN,'06_handoff'):
    RESULT=join_runtime(ROWS,RUN/'scene/scene_records.jsonl',SPI_FILE,RUN/'handoff')
    print('PARAMETER COVERAGE',RESULT,flush=True)
    write_json(RUN/'handoff/reference_records.json',REFERENCES)
''')
md('''## 7. CPU/GPU costs, memory and partial neural FLOPs

CPU and GPU use the same ten development images with ten warmups and30/100 calls respectively.
Keep setup/model loading, scene extraction and SPI prediction scopes separate. Do not sum percentile
latencies into a combined p95. Cached values are not fresh inference measurements. No VLM labeling,
training or simulator-dependent relevance/KB operations occur in this notebook.
''')
code('''with stage(RUN,'07_costs'):
    if MODE=='real':
        profile(ROWS,WEIGHTS,RUN/'scene',('cpu','cuda:0'))
        if not SCENE_ONLY:profile_spi(ROWS,BUNDLE,RUN/'spi_profiles',('cpu','cuda:0'))
    else:
        write_json(RUN/'profiles_fixture.json',{'status':'not_measured','reason':'software fixtures are not model performance evidence'})
''')
md('''## 8. Coverage diagnostics and persisted acceptance

Source availability and extraction completeness are independent. Missing managed datasets remain
missing even if all acquired images were cached successfully. New source distributions are not human
priority ground truth. The plots show producer behavior, not classification accuracy.
''')
code('''with stage(RUN,'08_results'):
    scene_summary=report(RUN/'scene');print(scene_summary,flush=True)
    import matplotlib.pyplot as plt
    records=[json.loads(s) for s in Path(SPI_FILE).read_text().splitlines()]
    counts={level:sum(r['spi'].get('level')==level and r['spi'].get('available') for r in records)
            for level in ('Low','Medium','High','Critical')}
    (RUN/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(list(counts),list(counts.values()))
    ax.set_ylabel('Predicted images');ax.set_title('Student SPI outputs — not verified urgency labels')
    fig.tight_layout();fig.savefig(RUN/'figures/spi_distribution.png',dpi=140);plt.show()
    complete_sources=all(r['status']=='complete' for r in SOURCE_STATUS.values())
    write_json(RUN/'ACCEPTANCE.json',{'mode':MODE,'acquired_images':len(ROWS),
       'parameters_complete':RESULT['all_complete'],'all_requested_sources_available':complete_sources,
       'status':('scene_only_spi_pending' if SCENE_ONLY else 'complete_for_acquired_sources') if (RESULT['all_complete'] or SCENE_ONLY) else 'partial_parameters',
       'scene_only':SCENE_ONLY,'spi_pending':SCENE_ONLY,
       'missing_sources':[k for k,v in SOURCE_STATUS.items() if v['status']!='complete'],
       'scene_accuracy':'unmeasured','SPI_domain_accuracy':'unmeasured','native_category_is_priority':False})
    files=[{'path':str(p.relative_to(RUN)),'sha256':file_hash(p),'bytes':p.stat().st_size}
           for p in RUN.rglob('*') if p.is_file() and 'images' not in p.relative_to(RUN).parts
           and not p.name.endswith(('.partial','.sqlite-journal')) and p.suffix not in ('.pt','.joblib','.safetensors')]
    write_json(RUN/'artifact_manifest.json',files)
    print('HANDOFF',RUN/'handoff/image_parameters.jsonl',flush=True)
    scene_rows=[json.loads(s) for s in (RUN/'scene/scene_records.jsonl').read_text().splitlines()]
    if SCENE_ONLY:
        if len(scene_rows)!=len(ROWS) or any(r['status']!='ok' for r in scene_rows):raise RuntimeError('scene extraction incomplete')
    elif not RESULT['all_complete']:raise RuntimeError('someparameteroutputsfailed: inspectcachedfailurestates')
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={
 'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},
 'language_info':{'name':'python'},'source_commit':commit})
nb.validate(notebook);target=HERE/'Image_Pool_Caching.ipynb';nb.write(notebook,target)
assert target.stat().st_size<1000000
write={'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,
       'source_hashes':snap,'commit':commit,'status':'prepared_not_submitted'}
(HERE/'source_manifest.json').write_text(json.dumps(write,indent=2),encoding='utf8')
print(target,'cells',len(cells),'bytes',target.stat().st_size)
