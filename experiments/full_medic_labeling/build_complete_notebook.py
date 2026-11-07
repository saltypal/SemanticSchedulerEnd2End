"""Build readable versioned Kaggle notebook, with saved labels embedded as JSON."""
from pathlib import Path
import hashlib,json,nbformat
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/full_medic_labeling'
stage=BASE/'runs/crisismmd1000_v4_complete';stage.mkdir(parents=True,exist_ok=True)
n=nbformat.read(BASE/'runs/crisismmd1000_v3/notebook.ipynb',4)
labels_path=BASE/'runs/monitor_v3/retrieved/experiments/full_medic_labeling/runs/main_20261029T010916Z/labels.jsonl'
labels=list(map(json.loads,labels_path.read_text().splitlines()))
assert len(labels)==1000
# Full raw records remain in version3 outputs; retain needed resume data without repeated80-class vocabularies.
labels=[{k:v for k,v in r.items() if k in ('item_id','source_split','image_sha256','level','available','parameter_status','clip_seconds','yolo_seconds')} for r in labels]
n.cells[0].source="# CrisisMMD1000: labels, four encoder banks, balanced ML and neural comparison\nGPU1: offline teacher; GPU0: sequential frozen encoders and YOLO. Matching completed labels are reused. Runtime relevance/KB/metadata are skipped. Four-class agreement is against pseudo-labels, not independently verified urgency."
n.cells[1].source=n.cells[1].source.replace("'ultralytics','pyarrow'","'ultralytics','pyarrow','scikit-learn==1.7.2','joblib','psutil','threadpoolctl','floppy-tracker==0.1.0'")
# Persist existing labels before expensive model loads; no costly teacher replay.
n.cells.insert(6,nbformat.v4.new_code_cell("print('RESTORE VERIFIED LABELS START',flush=True)\nSAVED_LABELS = "+repr(labels)+"\nassert len(SAVED_LABELS)==1000\nprint('RESTORE VERIFIED LABELS END',len(SAVED_LABELS),flush=True)"))
n.cells[7].source=n.cells[7].source.replace('path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,allow_patterns=', 'path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,allow_patterns=')
n.cells[7].source=n.cells[7].source.replace("path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])\nteacher=BulkTeacher(path,1)","teacher=None\nprint('VLM labeling already complete; GPU1 teacher load skipped on verified resume',flush=True)")
# Seed SQLite with existing results. New encoder banks use consistent padded preprocessing,
# so do not reuse the old native-preprocessing CLIP vectors.
s=n.cells[9].source
needle="started=time.monotonic();deadline=started+21600;new=0;stop=False"
s=s.replace(needle,"for saved in SAVED_LABELS:\n    with db: db.execute('INSERT OR IGNORE INTO labels VALUES(?,?,?,?,?)',(saved['item_id'],saved['source_split'],saved['image_sha256'],json.dumps(saved),None))\n"+needle)
n.cells[9].source=s
# Readable source files, not Base64 or an opaque archive.
n.cells.append(nbformat.v4.new_markdown_cell('## Shared source snapshot\nEach following cell writes a readable production module. Notebooks orchestrate the shared implementations.'))
needed=['experiments/spi_classifier_repair/'+f+'.py' for f in ('medic_campaign','head_matrix','medic_training_setup','resume_store')]
needed+=['experiments/full_medic_labeling/'+f+'.py' for f in ('balancing','crisis_study','costs')]
needed+=['parameters/'+f+'.py' for f in ('__init__','spi','artifacts','contracts','spi_labels','data')]
files=[ROOT/p for p in needed]
# Compact cohort retains identity/split/provenance; original tweet/annotation records remain in v3.
cohort=json.loads((BASE/'runs/crisismmd1000_selection/selected_1000.json').read_text())
cohort=[{k:v for k,v in r.items() if k in ('image_id','tweet_id','image_path','split','event_name')} for r in cohort]
n.cells[8].source='SELECTED = '+repr(cohort)+'\nprint("FROZEN COHORT",len(SELECTED),flush=True)'
parents=sorted({str(p.relative_to(ROOT).parent).replace('\\','/') for p in files})
n.cells.append(nbformat.v4.new_code_cell("print('SOURCE SNAPSHOT START',flush=True)\n"+"\n".join("Path("+repr(p)+").mkdir(parents=True,exist_ok=True)" for p in parents)))
for file in files:
    relative=file.relative_to(ROOT).as_posix()
    source=file.read_text(encoding='utf8') if relative!='parameters/__init__.py' else '# Standalone experiment: direct shared submodule imports; historical compatibility installation is not required.\n'
    n.cells.append(nbformat.v4.new_code_cell('%%writefile '+relative+'\n'+source))
n.cells.append(nbformat.v4.new_code_cell("""print('ROUND 3 ALL ENCODERS START',flush=True)
import gc
# Release the initial CLIP/detector before sequential bank extraction.
del clip,processor,yolo
if teacher is not None: del teacher
gc.collect();torch.cuda.empty_cache()
from experiments.full_medic_labeling.crisis_study import prepare,run
rows=prepare(SELECTED,RUN/'labels.jsonl',RUN/'comparison')
print('TEACHER PREFILL/DECODE COST SAMPLE START',flush=True)
from experiments.full_medic_labeling.costs import teacher_profile
try:
    teacher_path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
    teacher=BulkTeacher(teacher_path,1)
    teacher_profile(teacher,rows,RUN/'comparison/teacher_costs',SYSTEM_PROMPT)
    del teacher
except Exception as error:
    (RUN/'teacher_cost_error.json').write_text(json.dumps({'error':str(error),'status':'failed, not zero FLOPs'}))
    import traceback;traceback.print_exc()
gc.collect();torch.cuda.empty_cache()
print('ROUND 4/5 ALL CLASSIFIERS START',flush=True)
outcomes=run(rows,RUN/'comparison')
print('ROUND 4/5 COMPLETE',[(r['balance'],r['status']) for r in outcomes],flush=True)
"""))
for c in n.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
nbformat.validate(n);nbformat.write(n,stage/'notebook.ipynb')
meta=json.loads((BASE/'runs/crisismmd1000_v3/kernel-metadata.json').read_text());meta['title']='CrisisMMD1000 SPI Labeling';meta['code_file']='notebook.ipynb'
(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(stage, len(n.cells), (stage/'notebook.ipynb').stat().st_size)
