import base64,ast,hashlib,json,pathlib,nbformat
p=pathlib.Path('experiments/spi_four_class/runs/run20261028_spi_grounding_v1/retrieved_version4/experiments/spi_four_class/runs/run20261028_spi_grounding_v1/cell_evidence/executed.ipynb')
nb=nbformat.read(p,as_version=4); nbformat.validate(nb)
cells=[]; payloads=[]
for i,c in enumerate(nb.cells):
 outs=[]
 for o in c.get('outputs',[]):
  data=o.get('data',{})
  txt=o.get('text','')
  if 'text/plain' in data: txt += data['text/plain']
  if o.get('output_type')=='error': txt += '\n'.join(o.get('traceback',[]))
  outs.append({'type':o.get('output_type'),'text':str(txt)[:1800],'ename':o.get('ename'),'evalue':o.get('evalue')})
 cells.append({'index':i,'cell_type':c.cell_type,'execution_count':c.get('execution_count'),'output_count':len(outs),'outputs':outs})
 if c.cell_type=='code' and c.get('source'):
  tree=ast.parse(c.source)
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
    try: payloads.append(base64.b64decode(ast.literal_eval(node.args[0])))
    except Exception: pass
expected='5f385fbefde265f390d1520026e583d3cfbb37a2a4c1cb4f51d93da92ff9a3d1'
result={'nbformat_valid':True,'executed_notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cell_count':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'cell_executions':cells,'embedded_source_sha256s':[hashlib.sha256(x).hexdigest() for x in payloads],'embedded_source_matches_approved':any(hashlib.sha256(x).hexdigest()==expected for x in payloads)}
out=p.parent.parent.parent/'cell_capture_summary.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
