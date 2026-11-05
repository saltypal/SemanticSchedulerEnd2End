import hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(r'D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\experiments\image_pool\runs\wsl_scene_only_20261030')
slug='satyapaladuguplus/semantic-image-pool-scene-cache-20261030/1'
cli='/home/Satya/.venvs/semantic-kaggle/bin/kaggle'
r=subprocess.run([cli,'kernels','logs',slug],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
text=r.stdout+'\n'+r.stderr
text=re.sub(r'(?i)bearer\s+[A-Za-z0-9._~+/-]+=*','Bearer <redacted>',text)
text=re.sub(r'(?i)(api[_ -]?key|token|password|secret|authorization)(\s*[:=]\s*)(["\']?)[^\s,\'"}]+',r'\1\2<redacted>',text)
out=root/'terminal_logs';out.mkdir(parents=True,exist_ok=True)
p=out/'kaggle_v1.log';p.write_text(text,encoding='utf-8')
receipt={'kernel':slug,'collected_utc':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'log_path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest().upper(),'secret_pattern_redaction_applied':True}
(root/'terminal_logs'/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
lines=text.splitlines()
interesting=[i for i,line in enumerate(lines) if any(k in line.lower() for k in ('traceback','exception','error','papermillexecutionerror','execution failed'))]
if interesting:
 start=max(0,interesting[-1]-18); end=min(len(lines),interesting[-1]+22); excerpt='\n'.join(lines[start:end])
else:excerpt='\n'.join(lines[-30:])
print(json.dumps({'receipt':receipt,'line_count':len(lines),'excerpt':excerpt[-7000:]},indent=2))
