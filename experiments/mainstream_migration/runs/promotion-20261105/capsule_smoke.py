from pathlib import Path
import json,base64,re,io,zipfile,tempfile,subprocess,sys
root=Path.cwd();records=[]
for profile in ['wsl','windows']:
 p=next((root/'Notebooks/SchedulerSuite/paired'/profile).glob('*.ipynb'));book=json.loads(p.read_text());code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code');blob=base64.b64decode(re.search(r"base64.b64decode\('([^']+)'\)",code)[1])
 with tempfile.TemporaryDirectory(prefix='scheduler-promotion-') as tmp:
  target=Path(tmp)
  with zipfile.ZipFile(io.BytesIO(blob)) as z:z.extractall(target)
  for command in ['Scheduler/training/stream_suite/ue_rate_study.py','Scheduler/training/stream_suite/train_real_suite.py','experiments/stream_scheduler/ue_rate_study.py']:
   result=subprocess.run([sys.executable,str(target/command),'--help'],cwd='/tmp',capture_output=True,text=True,timeout=60)
   records.append({'profile':profile,'command':command,'exit':result.returncode,'stderr':result.stderr[-1000:]});assert result.returncode==0,records[-1]
(root/'experiments/mainstream_migration/runs/promotion-20261105/capsule_smoke.json').write_text(json.dumps(records,indent=2));print('Fresh extracted capsule CLI checks passed',len(records))
