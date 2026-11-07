import json,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/image_pool/runs/wsl_scene_only_20261030')
script=root/'monitor_v3.py';out=(root/'monitor_v3_stdout.log').open('ab',buffering=0);err=(root/'monitor_v3_stderr.log').open('ab',buffering=0)
p=subprocess.Popen(['/home/Satya/.venvs/semantic-kaggle/bin/python',str(script)],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,cwd=str(root))
receipt={'pid':p.pid,'script':str(script),'stdout':str(root/'monitor_v3_stdout.log'),'stderr':str(root/'monitor_v3_stderr.log'),'started_utc':datetime.now(timezone.utc).isoformat(),'kernel':'satyapaladuguplus/semantic-image-pool-scene-cache-20261030/3','source_sha256':'43AC33D6C0A4501C9256F838518D4C865470842029665B20BA3AF8DCBA0613F5','poll_seconds':60,'monitor_cap_seconds':24600}
(root/'monitor_v3_process.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
