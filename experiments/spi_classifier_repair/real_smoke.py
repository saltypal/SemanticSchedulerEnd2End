from pathlib import Path
import json,zipfile,time,hashlib
from parameters.spi import LEVELS
from experiments.spi_classifier_repair.cnn_compare import run
root=Path('.').resolve();base=root/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'
labels=json.loads((base/'spi_training/labels.json').read_text());panel=json.loads((root/'parameters/results/spi_inputs_v2_20261027b/panel.json').read_text());lookup={r['image_id']:r for r in panel};chosen=[]
for split,n in [('train',4),('validation',2)]:
 for level in LEVELS:chosen.extend([r for r in labels if r['split']==split and r['level']==level][:n])
out=root/'experiments/spi_classifier_repair/runs'/('real_image_smoke_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True);records=[]
with zipfile.ZipFile(root/'parameters/results/spi_inputs_v2_20261027b/images.zip') as z:
 for label in chosen:
  row=lookup[label['image_id']];data=z.read(row['image_path']);assert hashlib.sha256(data).hexdigest()==row['image_sha256']
  path=out/(row['image_id']+'.png');path.write_bytes(data);records.append({**row,'image_path':str(path)})
print('REAL24IMAGE CPU SOFTWARE PREFLIGHT, not quality evaluation',flush=True)
run(records,chosen,out/'smoke',device='cpu',seconds=300,max_epochs=1)
print('PREFLIGHT_OUTPUT',out,flush=True)
