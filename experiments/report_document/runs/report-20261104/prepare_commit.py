import subprocess,os,re,json
from pathlib import Path
root=Path.cwd()
def git(*args):return subprocess.run(['git','-c','core.longpaths=true',*args],capture_output=True,check=True)
names=[os.fsdecode(x) for x in git('ls-files','--cached','--others','--exclude-standard','-z').stdout.split(b'\0') if x]
excluded=[];safe=[]
pattern=re.compile(rb'(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-proj-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|X-Amz-Signature=[a-fA-F0-9]{40,}|[?&]GoogleAccessId=[^\s&]+.*?[&]Signature=)')
for name in dict.fromkeys(names):
 p=root/name
 if not p.exists():safe.append(name);continue
 if not p.is_file():continue
 size=p.stat().st_size
 reason=None
 if size>40*1024**2:reason='Large generated artifact above 40 MiB; preserved locally'
 elif p.name.lower() in ['.env','kaggle.json','credentials.json','id_rsa','id_ed25519']:reason='Credential filename'
 elif p.suffix.lower() not in ['.png','.jpg','.jpeg','.pdf','.zip','.tar','.npz','.pt','.ckpt','.docx','.pptx','.bin','.sqlite','.db']:
  if pattern.search(p.read_bytes()):reason='Potential credential or signed access URL; preserved locally without printing contents'
 if reason:excluded.append({'path':name,'bytes':size,'reason':reason})
 else:safe.append(name)
out=root/'experiments/github_checkpoint/runs/checkpoint-20261105';out.mkdir(parents=True,exist_ok=True)
(out/'excluded_artifacts.json').write_text(json.dumps(excluded,indent=2))
(out/'README.md').write_text('# GitHub checkpoint\n\nSource, notebooks, reports and eligible results are committed. Large generated files and potential credential-bearing records remain untouched locally. See excluded_artifacts.json for exact paths and reasons. No model retraining or final test evaluation occurred during this checkpoint.\n')
ignore=root/'.gitignore'
with ignore.open('a',encoding='utf-8') as f:
 f.write('\n# Local large artifacts and credential-bearing records excluded from GitHub checkpoint\n')
 for x in excluded:f.write('/'+x['path'].replace('[','\\[').replace(']','\\]')+'\n')
paths=list(dict.fromkeys(safe+['.gitignore',str((out/'excluded_artifacts.json').relative_to(root)).replace('\\','/'),str((out/'README.md').relative_to(root)).replace('\\','/')]))
spec=out/'stage_paths.txt';spec.write_bytes(b'\0'.join(os.fsencode(p) for p in paths)+b'\0')
print(json.dumps({'eligible_paths':len(paths),'excluded':len(excluded),'excluded_bytes':sum(x['bytes'] for x in excluded)}))
result=git('--literal-pathspecs','add','-A','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
spec.unlink()
print('Staging completed')
