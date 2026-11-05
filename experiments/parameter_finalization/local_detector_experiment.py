"""Bounded real-image detector comparison. Each model runs in a timed process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from parameters.artifacts import write_json


def worker(model,image_path,output,flip=False):
    from PIL import Image,ImageDraw,ImageOps
    from parameters.spatial_provider import make_spatial_provider
    checkpoint=ROOT/'models/priority_classifiers/grounding_dino' if model=='grounding_dino' else ROOT/'models/yolov8s-worldv2.pt'
    provider=make_spatial_provider(checkpoint,detector=model,device='cpu')
    provider.vocabulary=('person','car','building','bridge')
    started=time.perf_counter()
    provider.load()
    loaded=time.perf_counter()
    with Image.open(image_path) as im:image=im.convert('RGB')
    original=image.copy()
    if flip:image=ImageOps.mirror(image)
    result=provider.measure(image)
    ended=time.perf_counter()
    rows=result.audit.get('detections',[])
    if flip:
        rows=[{**r,'box':[image.width-r['box'][2],r['box'][1],image.width-r['box'][0],r['box'][3]]} for r in rows]
    overlay=original.copy();draw=ImageDraw.Draw(overlay)
    for row in rows:
        draw.rectangle(row['box'],outline='red',width=2)
        draw.text((row['box'][0],row['box'][1]),row['label'],fill='yellow')
    overlay.save(Path(output).with_suffix('.png'))
    from parameters.scene_measurements import measure_detections
    views={}
    for name,targets in {'infrastructure':{'building','bridge'},'people_vehicles':{'person','car'}}.items():
        views[name]=measure_detections([r for r in rows if r['label'] in targets],image.width,image.height).values() if result.status=='ok' else None
    write_json(output,{'model':model,'status':result.status,'reason':result.reason,
        'mission_views':views,'horizontal_flip':flip,'boxes_original_coordinates':rows,
        'values':result.values(),'audit':result.audit,'load_seconds':loaded-started,
        'first_inference_seconds':ended-loaded,'warmed_latency':False,
        'image_sha256':hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        'scope':'CPU detector output, not verified accuracy or urgency'})


def run(output,timeout=120,models=('yolo_world','grounding_dino')):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    panel=json.loads((ROOT/'experiments/spi_four_class/grounding_panel.json').read_text())['rows']
    all_rows=json.loads((ROOT/'parameters/results/spi_inputs_v2_20261027b/panel.json').read_text())
    chosen=panel[:2]+[r for r in all_rows if r['source']=='ladi' and r['image_id'] not in {x['image_id'] for x in panel[:2]}][:1]
    chosen += [r for r in all_rows if r['source']=='medic'][:3]
    with zipfile.ZipFile(ROOT/'parameters/results/spi_inputs_v2_20261027b/images.zip') as z:
        for i,row in enumerate(chosen):
            data=z.read(row['image_path'])
            if hashlib.sha256(data).hexdigest()!=row['image_sha256']:raise ValueError('image hash mismatch')
            (output/f'image_{i}.png').write_bytes(data)
    write_json(output/'manifest.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'images':chosen,'vocabulary':['person','car','building','bridge'],'python':sys.executable,
        'timeout_seconds_per_process':timeout,'scope':'six exposed images; no detection ground truth',
        'environment_threads':1})
    results=[]
    for model in models:
        for i in list(range(len(chosen)))+[1]:
            flip=(len([r for r in results if r['model']==model])==len(chosen))
            row=chosen[i]
            target=output/f'{model}_{i}{"_flip" if flip else ""}.json'
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',model,'--image',str(output/f'image_{i}.png'),'--output',str(target)]
            if flip:command.append('--flip')
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            start=time.perf_counter()
            with target.with_suffix('.log').open('w',encoding='utf-8') as log:
                try:
                    process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                    status='completed' if process.returncode==0 and target.exists() else 'failed'
                    result=json.loads(target.read_text()) if target.exists() else {}
                except subprocess.TimeoutExpired:
                    status='timeout';result={}
            results.append({'model':model,'image_id':row['image_id'],'process_status':status,
                            'process_seconds':time.perf_counter()-start,**result})
            write_json(output/'results.json',results)
            print(model,i+1,'/',len(chosen),status,result.get('values'),flush=True)
            if status in ('failed','timeout') or result.get('status')=='unavailable':
                print('Stop this candidate after first infrastructure failure; inspect log.',flush=True)
                break
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--image');p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=int,default=120);p.add_argument('--models',nargs='+',choices=('yolo_world','grounding_dino'),default=['yolo_world','grounding_dino'])
    p.add_argument('--flip',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker,a.image,a.output,a.flip)
    else:print(run(a.output,a.timeout,a.models))
