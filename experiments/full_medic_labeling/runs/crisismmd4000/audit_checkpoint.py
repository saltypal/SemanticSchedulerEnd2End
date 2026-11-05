import json,hashlib,sqlite3
from pathlib import Path
from collections import Counter
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
out=Path(__file__).parent/'checkpoint_audit';out.mkdir(exist_ok=True)
a=KaggleApi();a.authenticate()
with a.build_kaggle_client() as client:
 for name in ('label_cache.sqlite','labels.jsonl','progress.json','identity.json'):
  q=ApiDownloadKernelOutputRequest();q.owner_slug='satyapaladugu';q.kernel_slug='crisismmd1000-spi-labeling';q.version_number=7;q.file_path='experiments/full_medic_labeling/runs/main_20261029T174009Z/'+name
  r=client.kernels.kernels_api_client.download_kernel_output(q);r.raise_for_status()
  path=out/name
  with path.open('wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
  r.close();print('DOWNLOADED',name,path.stat().st_size,flush=True)
db=sqlite3.connect(str(out/'label_cache.sqlite'));integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
rows=[json.loads(v) for v, in db.execute('SELECT result FROM labels')]
counts=Counter(r.get('level') if r.get('available') is True else 'Unavailable' for r in rows)
seed=set()
for row in map(json.loads,Path('experiments/full_medic_labeling/runs/monitor_v3/retrieved/experiments/full_medic_labeling/runs/main_20261029T010916Z/labels.jsonl').read_text().splitlines()):seed.add(row['item_id'])
new=Counter()
for i,v in db.execute('SELECT id,result FROM labels'):
 r=json.loads(v)
 if i not in seed:new[r.get('level') if r.get('available') is True else 'Unavailable']+=1
report={'integrity':integrity,'rows':len(rows),'counts':dict(counts),'new_labels_counts':dict(new),'progress':json.loads((out/'progress.json').read_text()),'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()}}
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
