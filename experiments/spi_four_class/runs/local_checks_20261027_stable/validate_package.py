from pathlib import Path
import hashlib,json,zipfile,nbformat,sys
repo=Path.cwd(); stage=repo/'experiments/spi_four_class/runs/20261027_spi_evidence_v3a/submission_final'
package=json.loads((stage/'PACKAGE.json').read_text(encoding='utf-8-sig'))
km=json.loads((stage/'kernel-metadata.json').read_text(encoding='utf-8-sig'))
nb=nbformat.read(stage/'notebook.ipynb',as_version=4); nbformat.validate(nb)
compiled=[]
for i,c in enumerate(nb.cells):
 if c.cell_type=='code': compile(c.source,f'<notebook-cell-{i}>','exec'); compiled.append(i)
zip_path=stage/'source.zip'; zip_sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
report={'package':package,'kernel_metadata':km,'notebook_schema':'valid','code_cells_compiled':compiled,'source_zip_sha256':zip_sha,'archive_members':0,'manifest_source_files':0,'archive_manifest_mismatches':[],'canonical_source_mismatches':[],'canonical_missing':[],'required_source_presence':{},'runtime_receipt_mentions':[],'project_compat_present':False}
with zipfile.ZipFile(zip_path) as z:
 bad=z.testzip(); names=z.namelist(); report['archive_members']=len(names); report['archive_crc']='ok' if bad is None else f'bad:{bad}'
 manifest=json.loads(z.read('SOURCE_MANIFEST.json')); files=manifest['files']; report['manifest_source_files']=len(files); report['source_manifest_tag']=manifest.get('tag'); report['source_manifest_commit']=manifest.get('commit')
 for p,expected in files.items():
  data=z.read(p); got=hashlib.sha256(data).hexdigest()
  if got!=expected: report['archive_manifest_mismatches'].append({'path':p,'manifest':expected,'archive':got})
  local=repo/p
  if local.exists():
   lh=hashlib.sha256(local.read_bytes()).hexdigest()
   if lh!=got: report['canonical_source_mismatches'].append({'path':p,'archive':got,'canonical':lh})
  else: report['canonical_missing'].append(p)
 req=['project_compat.py','parameters/spi_protocol.py','parameters/spi_pipeline.py','parameters/spi_teachers.py','parameters/spi_model_selection.py','parameters/spi_final_campaign.py']
 report['required_source_presence']={p:p in names for p in req}; report['project_compat_present']='project_compat.py' in names
report['notebook_path']=str((stage/'notebook.ipynb').relative_to(repo));
for i,c in enumerate(nb.cells):
 if c.cell_type=='code' and 'runtime.json' in c.source:
  lines=[x.strip() for x in c.source.splitlines() if 'runtime.json' in x]
  report['runtime_receipt_mentions'].append({'cell':i,'lines':lines})
report['kernel_identity_exact']=km.get('id')=='satyapaladugu/betterscheduler-spi-20261027finalc' and package.get('kernel')==km.get('id')
report['input_identity_exact']=km.get('dataset_sources')==['satyapaladugu/betterscheduler-spiv2-inputs-20261027b'] and package.get('input')==km['dataset_sources'][0]
report['server_seconds']=package.get('server_seconds');report['gpu_hour_cap']=package.get('aggregate_gpu_hour_cap')
Path('experiments/spi_four_class/runs/local_checks_20261027_stable/package_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['source_zip_sha256','archive_members','manifest_source_files','archive_crc','notebook_schema','code_cells_compiled','archive_manifest_mismatches','canonical_source_mismatches','canonical_missing','required_source_presence','runtime_receipt_mentions','kernel_identity_exact','input_identity_exact','server_seconds','gpu_hour_cap']},indent=2))
