"""MEDIC-only 300-parent detector grid. No SPI training or full downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def acquire_medic(output,budget):
    """Download only individual images; retain native labels without remapping."""
    import hashlib,io,urllib.parse
    import numpy as np
    from PIL import Image
    from experiments.dataset_detector_grid.download_test import save
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];failures=[];seen=set();deadline=time.monotonic()+600
    base={'dataset':'QCRI/MEDIC','config':'default','split':'train'}
    meta=json.loads(budget.get('https://datasets-server.huggingface.co/size?'+urllib.parse.urlencode(base)))
    total=next(s['num_rows'] for s in meta['size']['splits'] if s['split']=='train')
    pages=np.random.default_rng(17).permutation(list(range(0,total,100)))
    for offset in pages:
        if len(rows)==300 or time.monotonic()>deadline:break
        payload=json.loads(budget.get('https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({**base,'offset':int(offset),'length':100}),limit=8_000_000))
        save(output/'native_features.json',payload.get('features',[]))
        for item in payload.get('rows',[]):
            if len(rows)==300 or time.monotonic()>deadline:break
            try:
                native=item['row'];raw=budget.get(native['image']['src']);identity=hashlib.sha256(raw).hexdigest()
                if identity in seen:continue
                with Image.open(io.BytesIO(raw)) as image:image.verify()
                path=output/(identity+'.img');path.write_bytes(raw);seen.add(identity)
                rows.append({'image_id':identity,'image_sha256':identity,'image_path':str(path.resolve()),'source':'medic',
                    'source_row':item['row_idx'],'native_labels':{k:v for k,v in native.items() if k!='image'},
                    'native_features':payload.get('features',[]),
                    'split':'development','group':identity,'event_grouping_verified':False})
            except Exception as error:failures.append({'row':item.get('row_idx'),'error':str(error)})
        save(output/'acquisition_progress.json',{'downloaded':len(rows),'bytes':budget.downloaded,'failures':failures})
        print('MEDIC individual images',len(rows),'/300',flush=True)
    return rows,failures,{'dataset':'QCRI/MEDIC','split':'train','sampling':'seed17 shuffled100-row pages','download_bytes':budget.downloaded}

def run(output):
    import torch
    from huggingface_hub import HfApi,snapshot_download
    from parameters.data import DownloadBudget
    from experiments.dataset_detector_grid.download_test import save
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if torch.cuda.device_count()!=2:raise RuntimeError('two GPUs required for explicit per-model ownership')
    print('MEDIC individual-image acquisition START cap300, image budget1GB',flush=True)
    rows,failures,source=acquire_medic(out/'data/images',budget=DownloadBudget(1_000_000_000))
    unique={}
    for row in rows:unique.setdefault(row['image_sha256'] if 'image_sha256' in row else row['image_id'],row)
    rows=list(unique.values())
    save(out/'data/records.json',rows);save(out/'data/acquisition.json',{'source':source,'failures':failures,'actual_unique_images':len(rows)})
    print('MEDIC acquisition END',len(rows),flush=True)
    if len(rows)!=300:raise RuntimeError('300 unique MEDIC images not available; no silent source substitution')
    api=HfApi();repo='IDEA-Research/grounding-dino-tiny';revision=api.model_info(repo).sha
    dino=snapshot_download(repo,revision=revision)
    from ultralytics import YOLOWorld
    weights=out/'models';weights.mkdir()
    # Ultralytics obtains its published model asset in this dedicated directory.
    before=Path.cwd()
    try:
        os.chdir(weights);YOLOWorld('yolov8s-worldv2.pt')
    finally:os.chdir(before)
    save(out/'models/identity.json',{'dino_repo':repo,'dino_revision':revision,'yolo_asset':'yolov8s-worldv2.pt'})
    runner=Path(__file__).with_name('detector_grid.py');procs=[]
    try:
        for model,path,device in [('yolo_world',weights/'yolov8s-worldv2.pt','cuda:0'),('grounding_dino',dino,'cuda:1')]:
            log=(out/f'{model}.log').open('w',encoding='utf8')
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            command=[sys.executable,str(runner),'--model',model,'--checkpoint',str(path),'--records',str(out/'data/records.json'),
                '--output',str(out/model),'--device',device,'--seconds','4800']
            procs.append((model,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env),log))
        deadline=time.monotonic()+4900
        while any(p.poll() is None for _,p,_ in procs):
            save(out/'progress.json',{n:{'returncode':p.poll(),'settings_saved':len(list((out/n).glob('setting_*.json')))} for n,p,_ in procs})
            print('GRID progress',json.loads((out/'progress.json').read_text()),flush=True)
            if time.monotonic()>deadline:raise TimeoutError('grid subprocess wall budget')
            time.sleep(30)
    finally:
        for _,p,log in procs:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
            log.close()
        save(out/'execution.json',{n:{'returncode':p.poll(),'status':json.loads((out/n/'status.json').read_text()) if (out/n/'status.json').exists() else 'no terminal receipt'} for n,p,_ in procs})
    print('GRID END inspect per-model receipts; execution is not detection accuracy',flush=True)
