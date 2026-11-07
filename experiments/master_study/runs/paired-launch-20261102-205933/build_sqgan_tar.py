from pathlib import Path
import json,tarfile
base=Path('/home/Satya/paired_sqgan_bundle_20261102/SQ-GAN')
run='20261028T130100Z-eea4f3aa'
out=Path('/tmp/scheduler_sqgan_input_20261102')
out.mkdir(parents=True,exist_ok=True)
with tarfile.open(out/'SQ-GAN.tar','w') as archive:
    archive.add(base/run,arcname=run,recursive=True)
(out/'dataset-metadata.json').write_text(json.dumps({'title':'Semantic Scheduler SQ-GAN Bundle 20261102','id':'satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102','licenses':[{'name':'other'}]},indent=2))
print('archive_bytes',(out/'SQ-GAN.tar').stat().st_size,flush=True)
