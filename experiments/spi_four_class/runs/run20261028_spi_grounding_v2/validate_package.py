import ast,base64,hashlib,json,zipfile,collections
from pathlib import Path
import nbformat
root=Path.cwd(); stage=root/'experiments/spi_four_class/runs/run20261028_spi_grounding_v2/submission'
nbpath=stage/'notebook.ipynb'; zpath=stage/'source.zip'
checks={}; pkg=json.loads((stage/'PACKAGE.json').read_text(encoding='utf-8-sig'))
meta=json.loads((stage/'kernel-metadata.json').read_text(encoding='utf-8-sig'))
pilot=json.loads((stage/'pilot_manifest.json').read_text(encoding='utf-8-sig'))
archive_bytes=zpath.read_bytes(); source_sha=hashlib.sha256(archive_bytes).hexdigest()
checks['package_sha_match']=pkg['source_sha256']==source_sha
checks['package_identity']=pkg['kernel']=='satyapaladugu/betterscheduler-spi-20261027finalc' and pkg['run_id']=='run20261028_spi_grounding_v2'
checks['resource_caps']=pkg['server_seconds']==3600 and pkg['aggregate_gpu_hour_cap']==2.0 and pkg['diagnostic'] is True
checks['metadata_identity']=meta['id']=='satyapaladugu/betterscheduler-spi-20261027finalc' and meta['id_no']==141089219 and meta['title']=='BetterScheduler SPI 20261027finalc'
checks['metadata_input']=meta['dataset_sources']==['satyapaladugu/betterscheduler-spiv2-inputs-20261027b'] and meta['enable_gpu'] is True and meta['machine_shape']=='NvidiaTeslaT4' and not meta['kernel_sources']
manifest=pilot['source_manifest']['files']
source_results={}
with zipfile.ZipFile(zpath) as z:
    names=set(z.namelist()); bad=[]; local_mismatch=[]
    for rel,expected in manifest.items():
        if rel not in names: bad.append({'path':rel,'issue':'missing archive member'});continue
        actual=hashlib.sha256(z.read(rel)).hexdigest()
        if actual!=expected:bad.append({'path':rel,'issue':'archive differs from source manifest'})
        if rel not in ('Evaluation/__init__.py','Scheduler/__init__.py','Scheduler/training/__init__.py') and (root/rel).is_file():
            local=hashlib.sha256((root/rel).read_bytes()).hexdigest()
            if local!=expected:local_mismatch.append({'path':rel,'expected':expected,'local':local})
    extras={'experiments/spi_four_class/visual_reliability.py','experiments/spi_four_class/grounding_panel.json'}
    checks['manifest_matches_zip']=not bad
    checks['canonical_files_match']=not local_mismatch
    checks['diagnostic_sources_in_archive']=extras.issubset(names)
    checks['archive_member_count']=len(names)
    checks['source_manifest_count']=len(manifest)
    checks['intentional_stubs']={n:hashlib.sha256(z.read(n)).hexdigest() for n in ('Evaluation/__init__.py','Scheduler/__init__.py','Scheduler/training/__init__.py')}
    diagnostic_source=z.read('experiments/spi_four_class/visual_reliability.py').decode('utf-8')
nb=nbformat.read(nbpath,as_version=4); nbformat.validate(nb)
codes=[c for c in nb.cells if c.cell_type=='code']
for c in codes: compile(c.source,'notebook-cell','exec')
first=ast.parse(codes[0].source); assignment=next(n for n in first.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CELL_SOURCES' for t in n.targets))
embedded_sources=ast.literal_eval(assignment.value)
checks['nbformat_valid']=True; checks['code_cell_count']=len(codes); checks['all_code_cells_compile']=True
checks['embedded_source_count']=len(embedded_sources)
rest=chr(10).join(codes[0].source.splitlines()[assignment.end_lineno:])
checks['embedded_code_cells_match']=len(embedded_sources)==len(codes) and rest==embedded_sources[0].rstrip('\n') and all(codes[i].source==embedded_sources[i] for i in range(1,len(codes)))
setup_tree=ast.parse(embedded_sources[0]); payloads=[]
for node in ast.walk(setup_tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
        try: payloads.append(base64.b64decode(ast.literal_eval(node.args[0])))
        except Exception: pass
checks['notebook_embeds_exact_source_zip']=archive_bytes in payloads
exec_tree=ast.parse(embedded_sources[2]); checks['diagnostic_exec_imported']=any(isinstance(n,ast.ImportFrom) and n.module=='experiments.spi_four_class.visual_reliability' for n in ast.walk(exec_tree))
export=embedded_sources[3]; checks['cell3_recovery_after_cell_close']=export.index('with recorder.cell(3):')<export.index("write_json(run/'retrieval_checksums.json'") and export.index('write_json(run/\'retrieval_checksums.json\'')>export.index('print(\'Prepare recovery export')
checks['cell3_excludes_self_checksum']=all(x in export for x in ("'run_recovery.zip'","'retrieval_checksums.json'"))
checks['no_student_training_or_promotion']=all(v in diagnostic_source for v in ('training_allowed','runtime_promotion_allowed')) and not any(v in diagnostic_source for v in ('candidate_release','fit_transform(','.fit('))
checks['teacher_loop_sequential_two']=all(v in diagnostic_source for v in ('TEACHERS[:2]','reviewer.close()','reviewer = None'))
checks['text_modality_flag_forwarded']='text_only_diagnostic=text_only' in diagnostic_source
checks['model_controls_and_grounding']=('for control in CONTROLS' in diagnostic_source and 'for row in baseline' in diagnostic_source)
checks['notebook_mentions_thinking_off']= 'thinking OFF' in nb.cells[0].source
panel=json.loads((root/'experiments/spi_four_class/grounding_panel.json').read_text(encoding='utf-8-sig'))
counts=collections.Counter(r['level'] for r in panel['rows'])
checks['panel_support_24_distinct_groups']=len(panel['rows'])==24 and len({r['group'] for r in panel['rows']})==24 and counts=={'Critical':8,'Medium':5,'High':3,'Low':8}
checks['panel_no_test_rows']=all(r['split']!='test' for r in panel['rows'])
result={'source_sha256':source_sha,'expected_run_id':pkg['run_id'],'kernel_metadata':meta,'package':pkg,'manifest_entries':len(manifest),'archive_members':len(names),'archive_manifest_failures':bad,'local_source_mismatches':local_mismatch,'panel_counts':dict(counts),'checks':checks,'all_checks_pass':all(v is True for v in checks.values() if isinstance(v,bool))}
(root/'experiments/spi_four_class/runs/run20261028_spi_grounding_v2/package_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('kernel_metadata','package','archive_manifest_failures','local_source_mismatches')},indent=2))

