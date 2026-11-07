"""Restore2540checkpoint records; label remaining1460 with four-class thinking rubric."""
import ast,json,sqlite3,hashlib,nbformat
from pathlib import Path
BASE=Path(__file__).resolve().parent
source=BASE/'runs/crisismmd4000'
stage=BASE/'runs/crisismmd4000_thinking';stage.mkdir(exist_ok=True)
n=nbformat.read(source/'notebook.ipynb',4)
checkpoint=source/'checkpoint_audit/label_cache.sqlite'
db=sqlite3.connect(str(checkpoint));assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
seed=[]
for i,split,sha,result in db.execute('select id,split,sha,result from labels order by id'):
 r=json.loads(result);seed.append((i,split,sha,r.get('level'),r.get('available')))
assert len(seed)==2540
n.cells[6].source='SEED_ROWS = '+repr(seed)+"\nSAVED_LABELS=[dict(zip(('item_id','source_split','image_sha256','level','available'),r)) for r in SEED_ROWS]\nprint('RESTORED CHECKPOINT RECORDS',len(SAVED_LABELS),flush=True)"
# Reconstruct the exact original cohort dictionaries, preserving its identity hash.
namespace={};exec(n.cells[8].source,namespace);cohort=namespace['SELECTED']
compact=[(r['image_id'],r['image_path'],r['split']) for r in cohort]
n.cells[8].source='COHORT_ROWS = '+repr(compact)+"\nSELECTED=[{'image_id':i,'tweet_id':int(i.rsplit('_',1)[0]),'image_path':p,'split':s} for i,p,s in COHORT_ROWS]\nprint('COHORT',len(SELECTED),flush=True)"
# Install rubric functions before the inline teacher class is defined.
rubric=(BASE/'spi_rubric_v3.py').read_text().replace('from experiments.spi_classifier_repair.evidence_priority import parse_evidence','')
n.cells[3].source+='\n'+rubric
old=n.cells[5].source
mod=ast.parse((BASE/'teacher.py').read_text());klass=next(x for x in mod.body if isinstance(x,ast.ClassDef) and x.name=='BulkTeacher')
code=ast.get_source_segment((BASE/'teacher.py').read_text(),klass)
old_ast=ast.parse(old);oldclass=next(x for x in old_ast.body if isinstance(x,ast.ClassDef))
oldsegment=ast.get_source_segment(old,oldclass)
n.cells[5].source=old.replace(oldsegment,code).replace("SYSTEM_PROMPT = BASE_PROMPT + '\nBe concise: use at most two observations, each under20 words. Return JSON only.'",'SYSTEM_PROMPT = PROMPT')
# Ensure actual hash is for the actual system prompt, not the previous rubric.
n.cells[5].source+="\nSYSTEM_PROMPT=PROMPT\nPROMPT_SHA256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()\n"
n.cells[7].source=n.cells[7].source.replace('teacher=BulkTeacher(path,1)','teacher=BulkTeacher(path,1,enable_thinking=True,system_prompt=SYSTEM_PROMPT,parser=parse_four_class)')
s=n.cells[9].source
s=s.replace("'tokens':192", "'tokens':2048,'enable_thinking':True,'label_protocol':'mixed_v1_and_v3_thinking'")
s=s.replace("json.dumps(saved),None", "json.dumps({**saved,'label_protocol':'v1_checkpoint','label_prompt_sha256':'fd732eafba52ddc09baaff9c6d4f5f968fc24ec9ff273efd8980ffc005dab035'}),None")
s=s.replace('result=teacher.review(image)',"result=teacher.review(image)\n            result.update(label_protocol='v3_thinking',label_prompt_sha256=PROMPT_SHA256)")
s=s.replace('deadline=started+19000','deadline=started+19000')
n.cells[9].source=s
# Final teacher profiling uses the same new adapter configuration.
n.cells[24].source=n.cells[24].source.replace('teacher=BulkTeacher(teacher_path,1)','teacher=BulkTeacher(teacher_path,1,enable_thinking=True,system_prompt=SYSTEM_PROMPT,parser=parse_four_class)')
n.cells[0].source='# CrisisMMD4000 continuation:2540previous records +1460new thinking-enabled four-class labels\nMixed prompt versions explicitly recorded. Additional labeling may require checkpoint continuation. Four encoders, all48pairs, balancing, tests and CPU/GPU cost profiling run after4000records. Failure is not silently converted to Low.'
for c in n.cells:
 if c.cell_type=='code':
  ast.parse(c.source.split('\n',1)[1] if c.source.startswith('%%writefile') else c.source)
  c.outputs=[];c.execution_count=None
nbformat.validate(n);nbformat.write(n,stage/'notebook.ipynb')
assert (stage/'notebook.ipynb').stat().st_size<1000000,'Kaggle1MBsource limit'
meta=json.loads((source/'kernel-metadata.json').read_text());(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('SOURCE READY',stage,(stage/'notebook.ipynb').stat().st_size)
