"""Same Kaggle identity,4000selected images, checkpointed label sessions then complete comparison."""
import json,hashlib,nbformat
from pathlib import Path
BASE=Path(__file__).resolve().parent
# First build the tested source snapshot; this does not execute inference.
exec(compile((BASE/'build_complete_notebook.py').read_text(),str(BASE/'build_complete_notebook.py'),'exec'))
source=BASE/'runs/crisismmd1000_v4_complete'
stage=BASE/'runs/crisismmd4000';stage.mkdir(exist_ok=True)
n=nbformat.read(source/'notebook.ipynb',4)
cohort=json.loads((BASE/'runs/crisismmd4000_selection/selected_1000.json').read_text())
cohort=[{k:v for k,v in r.items() if k in ('image_id','tweet_id','image_path','split','event_name')} for r in cohort]
old_ids={r['image_id'] for r in json.loads((BASE/'runs/crisismmd1000_selection/selected_1000.json').read_text())}
assert len(cohort)==4000 and old_ids<={r['image_id'] for r in cohort}
n.cells[0].source='# CrisisMMD4000: resumable labels, four frozen encoders, all classifier heads and cost study\nExisting1000labels are development-exposed and reused. New3000labels use identical image-only prompt. No guarantee of Medium coverage. Training runs only after all4000records are saved. Same Kaggle identity, successive versions for bounded sessions.'
n.cells[8].source='COHORT_ROWS = '+repr([(r['image_id'],r['tweet_id'],r['image_path'],r['split']) for r in cohort])+"\nSELECTED=[dict(zip(('image_id','tweet_id','image_path','split'),r)) for r in COHORT_ROWS]\nprint('FROZEN COHORT',len(SELECTED),flush=True)"
n.cells[7].source=n.cells[7].source.replace("teacher=None\nprint('VLM labeling already complete; GPU1 teacher load skipped on verified resume',flush=True)","path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])\nteacher=BulkTeacher(path,1)")
s=n.cells[9].source
s=s.replace("'selected_manifest_sha256':'09f7ca3c6c4485ce13f9ae81ed6ff67bfd155c296bed2848e3ef1cf6ba7d68ee'", "'selected_manifest_sha256':hashlib.sha256(json.dumps(SELECTED,sort_keys=True).encode()).hexdigest()")
s=s.replace("'target_rows':1000", "'target_rows':len(SELECTED)").replace("count==1000", "count==len(SELECTED)").replace("'/1000'", "'/4000'")
s=s.replace('deadline=started+21600','deadline=started+19000')
s=s.replace("        try:\n            with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()", """        try:
            raw=None
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url+('?download=true&retry='+str(attempt) if attempt else ''),timeout=90) as response:raw=response.read()
                    break
                except Exception:
                    if attempt==4:raise
                    time.sleep(min(16,2**attempt))""")
s += "\nprogress=json.loads((RUN/'progress.json').read_text())\nif not progress['coverage_complete']:\n    print('LABEL SESSION CHECKPOINTED: attach this version output to next version; training deferred',progress,flush=True)\n    raise SystemExit('Checkpoint complete; more labeling remains')\n"
n.cells[9].source=s
# Only the canonical4000snapshot can restore its label DB; old1000targets are seeded separately.
for c in n.cells:
    if c.cell_type=='code' and c.source.startswith('%%writefile experiments/full_medic_labeling/crisis_study.py'):
        c.source=c.source.replace("    pool=[i for i,r in enumerate(rows) if r['split']!='test']", "    exposed="+repr(old_ids)+"\n    for r in rows:\n        if r['image_id'] in exposed and r['split']=='test':r['split']='train'\n    # Prior exposed test results cannot form a fresh final evaluation.\n    exposed_groups={r['group'] for r in rows if r['image_id'] in exposed}\n    for r in rows:\n        if r['group'] in exposed_groups and r['split']=='test':r['split']='train'\n    pool=[i for i,r in enumerate(rows) if r['split']!='test']")
n.cells[6].source='SEED_ROWS = '+repr([(r['item_id'],r['source_split'],r['image_sha256'],r.get('level'),r.get('available')) for r in labels])+"\nSAVED_LABELS=[dict(zip(('item_id','source_split','image_sha256','level','available'),r)) for r in SEED_ROWS]\nprint('RESTORE VERIFIED LABELS',len(SAVED_LABELS),flush=True)"
for c in n.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
nbformat.validate(n);nbformat.write(n,stage/'notebook.ipynb')
meta=json.loads((source/'kernel-metadata.json').read_text());(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('4000 notebook ready',stage,(stage/'notebook.ipynb').stat().st_size)
