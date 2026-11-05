"""Bounded official-image preparation and frozen upstream extraction on Kaggle."""
from pathlib import Path
import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import numpy as np
from PIL import Image
from Scheduler.scene_policy import canonical
from Scheduler.training.scene_corpus import load_frozen_upstream,split_groups


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(canonical(value),encoding='utf-8');temporary.replace(p)


def request_bytes(url,maximum=20_000_000):
    """Three bounded attempts for idempotent public-data requests."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SemanticScheduler-research'}),timeout=45) as response:
                return response.read(maximum+1)
        except (urllib.error.URLError,TimeoutError) as error:
            retryable=not isinstance(error,urllib.error.HTTPError) or error.code in (429,500,502,503,504)
            if not retryable or attempt==2:raise
            delay=(1,3)[attempt]
            print('Transient data request',type(error).__name__,'retry',attempt+1,'of2 in',delay,'seconds',flush=True)
            time.sleep(delay)


def get_json(url):return json.loads(request_bytes(url))


def prepare_data(output,exclusions=(),maximum_bytes=2_000_000_000,*,target_counts=(80,80,80),source_splits=('test',)):
    if len(target_counts)!=3 or any(type(n) is not int or n<=0 for n in target_counts):
        raise ValueError('three positive source-stratum targets required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir()
    excluded=set(exclusions);records=[];fps=[];counts=[0,0,0];downloaded=0;errors=[]
    dataset='MITLL/LADI-v2-dataset'
    meta=get_json('https://huggingface.co/api/datasets/'+dataset);revision=meta['sha']
    # Source-train exposure belongs to the frozen producer, not the scheduler.
    if not source_splits or any(split not in ('test','train') for split in source_splits):
        raise ValueError('official train/test source splits required')
    for source_split in source_splits:
        phase_target=[min((n+1)//2,40) for n in target_counts] if len(source_splits)>1 and source_split=='test' else list(target_counts)
        for offset in range(0,8000 if source_split=='train' else 1100,100):
            query=urllib.parse.urlencode({'dataset':dataset,'config':'default','split':source_split,'offset':offset,'length':100})
            page=get_json('https://datasets-server.huggingface.co/rows?'+query)
            for entry in page.get('rows',[]):
                row=entry['row'];presence=row.get('buildings_any');damage=row.get('buildings_affected_or_greater')
                if type(presence) is not bool or type(damage) is not bool or (damage and not presence):continue
                stratum=2 if damage else (1 if presence else 0)
                if counts[stratum]>=phase_target[stratum]:continue
                try:
                    url=row['image']['src']
                    if urllib.parse.urlparse(url).hostname!='datasets-server.huggingface.co' or '/'+revision+'/' not in url:
                        raise ValueError('unverified source revision/host')
                    raw=request_bytes(url,min(20_000_000,maximum_bytes-downloaded))
                    downloaded+=len(raw)
                    if downloaded>maximum_bytes or len(raw)>20_000_000:raise RuntimeError('source download byte cap')
                    identity=hashlib.sha256(raw).hexdigest()
                    if identity in excluded:continue
                    from io import BytesIO
                    with Image.open(BytesIO(raw)) as source:
                        image=source.convert('L').resize((9,8));a=np.asarray(image)
                        bits=(a[:,1:]>a[:,:-1]).flatten();fp=sum(int(v)<<i for i,v in enumerate(bits));std=float(np.asarray(source.convert('L')).std())
                    if any(identity==old or (min(std,st)>10 and (fp^f).bit_count()<=2) for old,f,st in fps):continue
                    path=output/'images'/(identity+'.jpg');path.write_bytes(raw)
                    fps.append((identity,fp,std));excluded.add(identity)
                    records.append({'image_id':identity,'image_path':'images/'+path.name,'parent_id':identity,
                        'group':'ladi:'+identity,'stratum':stratum,'damage_positive':damage,'native_labels':{k:v for k,v in row.items() if k!='image'},
                        'source_row':entry['row_idx'],'source_split':source_split,'dataset_revision':revision,
                        'fingerprint':{'dhash':str(fp),'std':std},'event':None})
                    counts[stratum]+=1
                except RuntimeError:raise
                except Exception as e:errors.append({'row':entry['row_idx'],'error':str(e)})
            write_json(output/'progress.json',{'counts':counts,'downloaded_bytes':downloaded,'errors':len(errors)})
            print('Official cohort',counts,'bytes',downloaded,flush=True)
            if all(counts[i]>=phase_target[i] for i in range(3)):break
            if not page.get('rows'):break
    selected=split_groups(records)
    write_json(output/'records.json',selected)
    complete=tuple(counts)==tuple(target_counts)
    write_json(output/'data_receipt.json',{'status':'complete' if complete else 'partial','target_counts':list(target_counts),
        'counts':counts,'downloaded_bytes':downloaded,'revision':revision,
        'errors':errors,'role_counts':{r:sum(x['role']==r for x in selected) for r in ('train','validation','pilot_eval')},
        'source_splits':list(source_splits),'vision_producer_training_exposure':('train' in source_splits),
        'source_by_stratum':{split:[sum(r['source_split']==split and r['stratum']==i for r in selected) for i in range(3)] for split in source_splits},
        'scope':'scheduler-held-out groups; source-train images may be seen by frozen vision producer; event IDs unavailable'})
    if not complete:
        raise RuntimeError('partial cohort: source-stratum targets unmet; preserve observed rows')
    return selected


def extract_data(root,upstream_root,*,device='cuda:0',limit=None,deadline_seconds=1800):
    load_frozen_upstream(upstream_root)
    from huggingface_hub import snapshot_download
    from parameters.pilot_producers import PilotVisionProducer
    from parameters.image_claims import NativeAerialClaimProvider
    from parameters.ue_profiles import load_ue_profiles
    from parameters.model_settings import DetectorSettings
    import torch
    root=Path(root);rows=json.loads((root/'records.json').read_text())
    if limit:rows=rows[:limit]
    start=time.perf_counter()
    clip=snapshot_download('openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    missions,_=load_ue_profiles();native=NativeAerialClaimProvider(device=device)
    producer=PilotVisionProducer(clip,{'building_damage':missions['building_damage']},device=device,
        image_claim_provider=native,enable_detector=True,detector_settings=DetectorSettings(confidence=.05,iou=.5,image_size=640))
    existing={}
    path=root/'features.jsonl'
    if path.exists():existing={r['image_id']:r for r in map(json.loads,path.read_text().splitlines())}
    try:
        producer.load()
        native.load()
        expected_yolo='9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792'
        if producer.provenance.get('weights_sha256')!=expected_yolo:
            raise ValueError('detector checkpoint differs from frozen reference')
        write_json(root/'producer_receipt.json',producer.provenance)
        measured=[]
        for index,row in enumerate(rows):
            if row['image_id'] in existing:continue
            if time.perf_counter()-start>deadline_seconds:raise TimeoutError('bounded extraction deadline')
            if hashlib.sha256((root/row['image_path']).read_bytes()).hexdigest()!=row['image_id']:
                raise ValueError('source image bytes differ from cohort manifest')
            image_start=time.perf_counter()
            with Image.open(root/row['image_path']) as image:
                evidence=producer.extract_batch([image.convert('RGB')],[missions['building_damage']])[0]
            entry={'image_id':row['image_id'],'evidence':evidence.to_dict()}
            with path.open('a',encoding='utf-8') as f:f.write(canonical(entry)+'\n')
            measured.append(time.perf_counter()-image_start)
            if len(measured) in (1,10):
                projected=(len(rows)-index-1)*float(np.mean(measured))
                write_json(root/('preflight_'+str(len(measured))+'.json'),{'images':len(measured),
                    'seconds':measured,'projected_remaining_seconds':projected,'elapsed':time.perf_counter()-start})
                # Model loading and the first CUDA warmup are not steady-state
                # per-image costs. One-image output is an integration gate;
                # use the ten-image measurement for workload projection.
                if len(measured)==10 and time.perf_counter()-start+projected>deadline_seconds:
                    raise TimeoutError('measured preflight projects beyond extraction cap; preserve partial cache')
            if index%10==0:print('Extraction',index+1,'/',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
    finally:
        producer.close()
        write_json(root/'extraction_receipt.json',{'seconds':time.perf_counter()-start,'device':device,
            'torch':torch.__version__,'cuda':torch.cuda.is_available(),'frozen_upstream':str(upstream_root)})


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--stage',choices=['data','extract'],required=True);p.add_argument('--exclusions');p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--target-counts',type=int,nargs=3,default=(80,80,80))
    args=p.parse_args()
    if args.stage=='data':prepare_data(args.output,json.loads(Path(args.exclusions).read_text()) if args.exclusions else (),target_counts=tuple(args.target_counts))
    else:extract_data(args.output,args.upstream,deadline_seconds=args.seconds)

if __name__=='__main__':main()
