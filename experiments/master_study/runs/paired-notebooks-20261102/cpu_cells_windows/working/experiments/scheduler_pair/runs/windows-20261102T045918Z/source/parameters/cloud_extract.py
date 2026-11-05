"""Actual bounded dual-GPU extraction on Kaggle, with explicit cache provenance."""
from pathlib import Path
import json
import time
import numpy as np
from .artifacts import Run,write_json
from .contracts import MISSIONS,digest
from .cloud_data import load_portable_records
from .vision import FrozenVisionProvider,MODEL_IDS


def extract(manifest,output,gpu_hours=8.,batch_size=16,resume=None):
    import torch
    from PIL import Image
    from huggingface_hub import HfApi,snapshot_download
    devices=torch.cuda.device_count()
    if devices<2:raise RuntimeError('two assigned GPUs required before extraction')
    run=Run(output,{'stage':'extract','input_manifest':str(manifest),'seconds':int(gpu_hours*3600/devices),
                    'aggregate_gpu_hours_cap':gpu_hours,'assigned_gpus':devices,'batch_size':batch_size})
    records=load_portable_records(manifest);by_id={r['image_id']:r for r in records};records=list(by_id.values())
    identity={'images':sorted(by_id),'preprocessing':'aspect-preserving-pad224-v1',
              'missions':{key:value.version for key,value in MISSIONS.items()},
              'backbones':['clip','siglip'],'batch_size':batch_size}
    write_json(run.path/'data/extraction_identity.json',{'definition':identity,'sha256':digest(identity)})
    hardware={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_names':[torch.cuda.get_device_name(i) for i in range(devices)],
              'gpu_count':devices,'data_parallel_devices':[0,1]}
    write_json(run.path/'hardware.json',hardware)
    profiles={}
    resume_root=Path(resume) if resume else None
    if resume_root:
        previous_identity=json.loads((resume_root/'data/extraction_identity.json').read_text())
        if previous_identity.get('sha256')!=digest(identity):
            raise ValueError('resume dataset/configuration/preprocessing identity differs')
        previous=json.loads((resume_root/'manifest.json').read_text())
        if previous.get('source_hashes')!=run.manifest.get('source_hashes'):
            raise ValueError('resume source snapshot differs')
    for backbone in ('clip','siglip'):
        with run.stage(backbone+'_model_preflight'):
            info=HfApi().model_info(MODEL_IDS[backbone]);names={s.rfilename for s in info.siblings}
            weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
            model_path=snapshot_download(MODEL_IDS[backbone],revision=info.sha,
                allow_patterns=[weight,'*.json','*.txt','*.model','merges.txt','vocab.json'])
            provider=FrozenVisionProvider(backbone,model_path,device='cuda:0',data_parallel=True,batch_size=batch_size,local_files_only=True)
            provider.load();provider.provenance['model_revision']=info.sha
            pictures=[]
            for row in records[:4]:
                with Image.open(row['image_path']) as image:pictures.append(image.convert('RGB'))
            proof=provider.verify_parallel(pictures,MISSIONS['building_damage'])
            write_json(run.path/'metrics'/(backbone+'_parallel_preflight.json'),proof)
            print('Verified real DataParallel:',backbone,proof,flush=True)
        with run.stage(backbone+'_feature_cache'):
            started=time.monotonic();count=0;seen=set();destination=run.path/'features'/(backbone+'.jsonl')
            try:
                with destination.open('x',encoding='utf-8') as handle:
                    if resume_root and (resume_root/'features'/(backbone+'.jsonl')).exists():
                        with (resume_root/'features'/(backbone+'.jsonl')).open() as previous:
                            for line in previous:
                                cached=json.loads(line);identity=cached['image_id']
                                if identity not in by_id or identity in seen or cached['evidence']['provenance'].get('weights_sha256')!=provider.provenance['weights_sha256']:
                                    raise ValueError('incompatible/duplicate cached image')
                                vector=np.asarray(cached['evidence']['embedding'],float)
                                if vector.ndim!=1 or not np.isfinite(vector).all():raise ValueError('invalid cached embedding')
                                handle.write(line);seen.add(identity);count+=1
                        write_json(run.path/'features'/(backbone+'_resume.json'),{'origin':str(resume_root),'validated_rows':count})
                    pending=[r for r in records if r['image_id'] not in seen]
                    begin=0
                    while begin<len(pending):
                        run.budget();chunk=pending[begin:begin+provider.batch_size];images=[]
                        for row in chunk:
                            with Image.open(row['image_path']) as image:images.append(image.convert('RGB'))
                        try:values=provider.extract_batch(images,[MISSIONS['building_damage']]*len(images))
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if provider.batch_size<=4:raise
                            provider.batch_size//=2
                            print('OOM: reducing batch size to',provider.batch_size,flush=True);continue
                        for row,evidence in zip(chunk,values):
                            if row['image_id'] in seen:raise RuntimeError('duplicate image ownership')
                            seen.add(row['image_id']);handle.write(json.dumps({'image_id':row['image_id'],'evidence':evidence.to_dict()},allow_nan=False)+'\n')
                        handle.flush();count+=len(values);begin+=len(values)
                        if count%128<provider.batch_size or count==len(records):
                            status={'images':count,'seconds':time.monotonic()-started,'batch_size':provider.batch_size,
                                    'gpu_hours_allocated':devices*(time.monotonic()-run.started)/3600}
                            write_json(run.path/'features'/(backbone+'_progress.json'),status)
                            print(backbone,status,flush=True)
                ownership={str(d):provider.device_ownership.count(d) for d in (0,1)}
                if pending and not all(ownership.values()):raise RuntimeError('both GPUs did not process image rows')
                profiles[backbone]={'records':count,'seconds':time.monotonic()-started,'weights':provider.provenance,
                                    'model_revision':info.sha,'ownership':ownership,'parallel_proof':proof}
            finally:provider.close()
        write_json(run.path/'metrics/producer_profiles.json',profiles)
    write_json(run.path/'status.json',{'status':'complete','unique_images':len(records),'seconds':time.monotonic()-run.started,
               'gpu_hours_allocated':devices*(time.monotonic()-run.started)/3600,'actual_data_parallel':True})
    return run.path
