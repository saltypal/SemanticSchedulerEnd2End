import ast,json,nbformat
from pathlib import Path
BASE=Path(__file__).resolve().parent
source=BASE/'runs/crisismmd4000_thinking'
stage=BASE/'runs/crisismmd4000_qwen25';stage.mkdir(exist_ok=True)
n=nbformat.read(source/'notebook.ipynb',4)
# Retain2540old labels; retry all1460remaining, including failed thinking attempts.
text=(BASE/'qwen25_teacher.py').read_text();klass=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef));code=ast.get_source_segment(text,klass)
n.cells[5].source="MODEL_ID='Qwen/Qwen2.5-VL-7B-Instruct'\nMODEL_REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'\nSYSTEM_PROMPT=PROMPT\nPROMPT_SHA256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()\n"+code
n.cells[7].source=n.cells[7].source.replace('enable_thinking=True','enable_thinking=False')
n.cells[9].source=n.cells[9].source.replace("'tokens':2048,'enable_thinking':True", "'tokens':384,'enable_thinking':False").replace("mixed_v1_and_v3_thinking","mixed_v1_and_v3_qwen25").replace("label_protocol='v3_thinking'", "label_protocol='v3_qwen25_no_thinking'")
n.cells[24].source=n.cells[24].source.replace('enable_thinking=True','enable_thinking=False')
n.cells[0].source='# CrisisMMD4000 continuation: Qwen2.5-VL7B, four-class rubric, no thinking\nRestore2540priorrecords; retryremaining1460includingthinkingfailures. GPU1teacher/GPU0imageproducers. Fulltrainingandprofilingafterlabelcoverage.'
# Small real-image gate prevents repeating the previous bulk failure.
n.cells[9].source=n.cells[9].source.replace("            result.update(label_protocol='v3_qwen25_no_thinking',label_prompt_sha256=PROMPT_SHA256)","            result.update(label_protocol='v3_qwen25_no_thinking',label_prompt_sha256=PROMPT_SHA256)")
n.cells[9].source=n.cells[9].source.replace("        new+=1", "        new+=1\n        if new<=10 and not result.get('available'):\n            print('TEACHER SAMPLE FAILURE',item,result.get('reason'),flush=True)\n        if new==10:\n            sample=[json.loads(v) for v, in db.execute(\"SELECT result FROM labels\") if json.loads(v).get('label_protocol')=='v3_qwen25_no_thinking']\n            good=sum(r.get('available') is True for r in sample)\n            print('TEACHER TEN IMAGE GATE',good,'/',len(sample),flush=True)\n            if good<8:save_outputs();raise RuntimeError('teacher validity gate failed; preserve sample instead of bulk failure')")
for c in n.cells:
 if c.cell_type=='code':ast.parse(c.source.split('\n',1)[1] if c.source.startswith('%%writefile') else c.source)
nbformat.write(n,stage/'notebook.ipynb');assert (stage/'notebook.ipynb').stat().st_size<1000000
(stage/'kernel-metadata.json').write_text((source/'kernel-metadata.json').read_text())
print(stage,(stage/'notebook.ipynb').stat().st_size)
