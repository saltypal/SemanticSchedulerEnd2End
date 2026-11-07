"""Rebuild training notebook from live shared modules and preserve older stages."""
from pathlib import Path
import base64,hashlib,io,json,re,time,zipfile
import nbformat

def build():
    root=Path(__file__).resolve().parents[2];exp=root/'experiments/spi_classifier_repair'
    source=exp/'notebooks/notebook9_medic_spi_training.ipynb'
    nb=nbformat.read(source,as_version=4)
    paths=[root/'project_compat.py']+list((root/'parameters').glob('*.py'))+list((root/'experiments/dataset_detector_grid').glob('*.py'))+list(exp.glob('*.py'))
    paths += [root/'parameters/config/final_labeling_teacher.json',root/'parameters/results/spi_inputs_v2_20261027b/panel.json',
              root/'parameters/config/ue_task_profiles.json',root/'parameters/config/medic_native_schema_v18.json']
    stream=io.BytesIO();manifest={}
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            value=path.read_bytes();name=path.relative_to(root).as_posix();archive.writestr(name,value);manifest[name]=hashlib.sha256(value).hexdigest()
    blob=stream.getvalue();sha=hashlib.sha256(blob).hexdigest()
    nb.cells[1].source=re.sub(r'blob=base64.b64decode\([^\n]+\)',lambda _:f'blob=base64.b64decode({base64.b64encode(blob).decode()!r})',nb.cells[1].source)
    nb.cells[1].source=re.sub(r'assert hashlib.sha256\(blob\).hexdigest\(\)==[^\n]+',lambda _:f'assert hashlib.sha256(blob).hexdigest()=={sha!r}',nb.cells[1].source)
    nb.cells[1].source=nb.cells[1].source.replace("'bitsandbytes'","'bitsandbytes==0.49.2'").replace("'accelerate'","'accelerate==1.12.0'")
    if "'huggingface-hub==1.5.0'" not in nb.cells[1].source:
        nb.cells[1].source=nb.cells[1].source.replace("'transformers==5.5.0'","'transformers==5.5.0','huggingface-hub==1.5.0','sentencepiece==0.2.2'")
    nb.cells[1].source=nb.cells[1].source.replace("'nbformat'","'nbformat==5.10.4'")
    nb.cells[2].source='''print('EXACT MEDIC COHORT LOAD START',flush=True)
from parameters.artifacts import write_json
from experiments.spi_classifier_repair.resume_store import file_hash
found=[]
for path in Path('/kaggle/input').glob('**/input_manifest.json'):
 try:
  manifest=json.loads(path.read_text())
  if manifest.get('schema')=='medic300_v18':found.append((path,manifest))
 except (ValueError,OSError):continue
if len(found)!=1:raise RuntimeError('Attach exactly one verified MEDIC300 snapshot; no automatic reacquisition or resampling')
manifest_path,manifest=found[0];input_root=manifest_path.parent
records_path=input_root/'records.json'
expected='1b0feb7ec30c3d9fb033d9df8456daf37e56b5941f35e40aeb684d02e98845ae'
if file_hash(records_path)!=expected or manifest.get('original_records_sha256')!=expected:raise RuntimeError('MEDIC cohort manifest differs')
rows=json.loads(records_path.read_text());schema=json.loads((PROJECT/'parameters/config/medic_native_schema_v18.json').read_text())['features']
for row in rows:
 matches=list(input_root.glob('**/'+row['image_sha256']+'.img'))
 if len(matches)!=1:raise RuntimeError('exact upstream image missing')
 if file_hash(matches[0])!=row['image_sha256']:raise RuntimeError('MEDIC image hash mismatch')
 row['image_path']=str(matches[0]);row['prior_detector_study_exposure']=True
 row['native_features']=schema
if len(rows)!=300 or any(r['source']!='medic' for r in rows):raise RuntimeError('MEDIC300 cohort mismatch')
write_json(RUN/'data/records.json',rows)
write_json(RUN/'input_receipt.json',{'manifest':manifest,'verified_images':300,'records_sha256':expected,'source_sha256':hashlib.sha256(blob).hexdigest()})
# Restore only an explicitly identified compatible checkpoint archive, if attached.
for index_path in Path('/kaggle/input').glob('**/resume_index.json'):
 resume=json.loads(index_path.read_text())
 if resume.get('schema')!='medic_spi_checkpoints_v1' or resume.get('records_sha256')!=expected:continue
 archive=index_path.parent/resume['archive']
 if file_hash(archive)!=resume['sha256']:raise RuntimeError('resume archive checksum mismatch')
 with zipfile.ZipFile(archive) as saved:
  for item in saved.infolist():
   target=(RUN/item.filename).resolve()
   if not target.is_relative_to(RUN.resolve()) or not item.filename.startswith('campaign/'):raise RuntimeError('unsafe resume archive')
  saved.extractall(RUN)
 print('RESTORED CHECKPOINTS',archive,flush=True)
print('EXACT MEDIC COHORT LOAD END',len(rows),flush=True)
'''
    nb.cells[3].source='''print('SPI LABEL/TRAIN/PROFILE START',flush=True)
from experiments.spi_classifier_repair.medic_campaign import run
started=time.monotonic()
try:
 run(RUN/'data/records.json',PROJECT/'parameters/results/spi_inputs_v2_20261027b/panel.json',PROJECT/'parameters/config/final_labeling_teacher.json',RUN/'campaign')
except BaseException:
 (RUN/'failure.txt').write_text(traceback.format_exc());raise
finally:
 print('CAMPAIGN ELAPSED',time.monotonic()-started,flush=True)
 # Checkpoints persist incrementally; package them even on a stage exception.
 print('RECOVERY PACKAGING START',flush=True)
 recovered=[]
 with zipfile.ZipFile('/kaggle/working/spi_recovery.zip','w',zipfile.ZIP_STORED) as archive:
  for file in RUN.rglob('*'):
   if file.is_file():
    name=file.relative_to(RUN).as_posix();archive.write(file,name);recovered.append({'path':name,'bytes':file.stat().st_size,'sha256':file_hash(file)})
 (Path('/kaggle/working')/'spi_recovery_manifest.json').write_text(json.dumps({'run':str(RUN),'files':recovered},indent=2))
 print('RECOVERY PACKAGING END',len(recovered),flush=True)
'''
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'training_notebook','exec')
    stage=exp/'runs'/('medic_training_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));stage.mkdir(parents=True,exist_ok=False)
    nbformat.write(nb,source);nbformat.write(nb,stage/'notebook.ipynb')
    meta=json.loads((exp/'runs/medic_training_submission/kernel-metadata.json').read_text())
    meta['dataset_sources']=['satyapaladugu/semanticscheduler-medic300-v18']
    # The existing identity is also upstream: do not invent a self-dependency.
    # Ops must attach a verified persistent input snapshot before submission.
    (stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (stage/'source.zip').write_bytes(blob);(stage/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    (stage/'VALIDATION.json').write_text(json.dumps({'source_sha256':sha,'nbformat':True,'code_compiled':True,
        'kernel':nb.metadata.kernelspec.name,'required_before_launch':'verified exact MEDIC grid input attachment and Kaggle authentication',
        'remote_submitted':False},indent=2))
    print(stage)
    return stage

if __name__=='__main__':build()
