import json,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/image_pool/runs/wsl_scene_only_20261030')
script=root/'monitor_v4.py';out=(root/'monitor_v4_stdout.log').open('ab',buffering=0);err=(root/'monitor_v4_stderr.log').open('ab',buffering=0)
p=subprocess.Popen(['/home/Satya/.venvs/semantic-kaggle/bin/python',str(script)],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,cwd=str(root))
receipt={'pid':p.pid,'script':str(script),'stdout':str(root/'monitor_v4_stdout.log'),'stderr':str(root/'monitor_v4_stderr.log'),'started_utc':datetime.now(timezone.utc).isoformat(),'kernel':'satyapaladuguplus/semantic-image-pool-scene-cache-20261030/4','source_sha256':'DD51805844DC2B4B67D613F790795F6D8AEE202BB01C8FC0F1C4C15D553DADC2','poll_seconds':60,'monitor_cap_seconds':24600}
(root/'monitor_v4_process.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
