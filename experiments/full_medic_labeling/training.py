"""Full-size entry point for the existing42-candidate classifier comparison."""
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
from PIL import Image,ImageOps

from experiments.spi_classifier_repair.resume_store import atomic_json,file_hash,fingerprint
from experiments.spi_classifier_repair.medic_campaign import compare,load_encoder,encode
from experiments.full_medic_labeling.campaign import SPLIT_COUNTS,LEVELS,inventory,download_shard


def require_complete(output):
    output=Path(output)
    manifest=json.loads((output/'checkpoint_manifest.json').read_text())
    if file_hash(output/'full_medic_checkpoint.sqlite')!=manifest['sha256']:
        raise ValueError('saved label checkpoint checksum mismatch')
    db=sqlite3.connect(f"file:{(output/'full_medic_checkpoint.sqlite').resolve().as_posix()}?mode=ro",uri=True)
    counts=dict(db.execute('SELECT split,COUNT(*) FROM items GROUP BY split'))
    if counts!=SPLIT_COUNTS:
        db.close();raise ValueError('training blocked: full71198-row labeling pass is incomplete')
    return db,manifest


def training_records(db):
    # Resolve cross-split exact copies before fitting, retaining the most reserved role.
    grouped={}
    priority={'train':0,'dev':1,'test':2}
    for sha,source,result in db.execute('SELECT i.sha,i.source,p.result FROM items i JOIN predictions p ON i.sha=p.sha ORDER BY i.id'):
        item=json.loads(source);prediction=json.loads(result)
        if sha not in grouped:
            grouped[sha]={'item':item,'prediction':prediction,'source_split':item['source_split']}
        elif priority[item['source_split']]>priority[grouped[sha]['source_split']]:
            grouped[sha]['source_split']=item['source_split']
    rows=[];excluded=[]
    for sha,value in sorted(grouped.items()):
        pred=value['prediction'];role=value['source_split']
        if role=='test' or pred.get('available') is not True or pred.get('level') not in LEVELS:
            excluded.append({'image_sha256':sha,'reason':'reserved_native_test' if role=='test' else 'unavailable_teacher_label'});continue
        rows.append({'image_id':sha,'image_sha256':sha,'source':'medic','group':'exact_sha:'+sha,
                     'split':'validation' if role=='dev' else 'train','source_item':value['item'],
                     'level':pred['level'],'available':True,'event_grouping_verified':False})
    for role in ('train','validation'):
        if {r['level'] for r in rows if r['split']==role}!=set(LEVELS):
            raise ValueError('training blocked: four-class support absent in '+role)
    return rows,excluded


class BatchBank:
    """Lazy row views; only a minibatch is converted tofloat32 for the head."""
    def __init__(self,array,identity,indices=None):
        self.array=array;self.indices=np.arange(len(array)) if indices is None else np.asarray(indices)
        self.shape=(len(self.indices),)+array.shape[1:];self.content_identity=identity
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        value=self.array[self.indices[index]]
        return np.asarray(value,dtype=bool if self.array.dtype==bool else np.float32)
    def subset(self,indices):return BatchBank(self.array,self.content_identity,self.indices[indices])


class FullFeatureLoader:
    def __init__(self,output):
        self.output=Path(output);self.previous=None;self.live=[]
    def __call__(self,rows,name,out):
        import pyarrow.parquet as pq
        import torch
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        # Previous encoder's heads are committed before the next encoder arrives.
        for array in self.live:
            if hasattr(array,'_mmap'):array._mmap.close()
        self.live=[]
        if self.previous:
            for file in self.previous.glob('tokens_*.npy'):file.unlink()
        self.previous=out
        ids=[r['image_id'] for r in rows]
        signature=fingerprint({'rows':[(r['image_id'],r['split']) for r in rows],
                              'encoder':name,'source':file_hash(__file__)})
        marker=out/'feature_progress.json'
        state=json.loads(marker.read_text()) if marker.exists() else {'signature':signature,'shards':[]}
        if state['signature']!=signature:raise ValueError('feature cache source/config mismatch')
        core,projection,processor,identity=load_encoder(name,'cuda:0')
        atomic_json(out/'identity.json',{**identity,'encoder_name':name})
        have_tokens=True
        # Probe shape using a real retained image in the source walk below.
        vectors=None;tokens=None;masks=None
        scratch=Path('/kaggle/temp/full_medic_training_shards');scratch.mkdir(parents=True,exist_ok=True)
        needed={}
        for index,row in enumerate(rows):
            item=row['source_item'];needed.setdefault(item['source_shard'],{})[item['shard_row']]=index
        shapes=state.get('shapes')
        if shapes:
            paths=[out/'vectors.npy']+([out/'tokens_data.npy',out/'tokens_mask.npy'] if have_tokens else [])
            if all(p.exists() for p in paths):
                vectors=np.load(paths[0],mmap_mode='r+')
                if have_tokens:tokens=np.load(paths[1],mmap_mode='r+');masks=np.load(paths[2],mmap_mode='r+')
            else:state['shards']=[]
        for shard in inventory()['shards']:
            targets=needed.get(shard['path'],{})
            if not targets or shard['path'] in state['shards']:continue
            path=download_shard(shard,scratch/Path(shard['path']).name)
            positions=[];pictures=[];sizes=[];row_number=0
            def flush():
                nonlocal vectors,tokens,masks
                if not pictures:return
                kw={'do_resize':False}
                if hasattr(processor,'do_center_crop'):kw['do_center_crop']=False
                pixels=processor(images=pictures,return_tensors='pt',**kw)['pixel_values'].to('cuda:0')
                with torch.inference_mode():v,t=encode(core,projection,pixels,name)
                values=v.cpu().numpy();patches=t.cpu().numpy()
                if not np.isfinite(values).all() or not np.isfinite(patches).all():raise ValueError('nonfinite frozen features')
                if vectors is None:
                    dimensions={'vectors':[len(rows),values.shape[-1]],'tokens':[len(rows),*patches.shape[1:]]}
                    bytes_needed=len(rows)*values.shape[-1]*4
                    if have_tokens:bytes_needed+=int(np.prod(dimensions['tokens']))*2+len(rows)*patches.shape[1]
                    if shutil.disk_usage(out).free<bytes_needed+3_000_000_000:
                        raise RuntimeError('full token bank exceeds available disk; no silent subsampling')
                    vectors=np.lib.format.open_memmap(out/'vectors.npy',mode='w+',dtype='float32',shape=tuple(dimensions['vectors']))
                    if have_tokens:
                        tokens=np.lib.format.open_memmap(out/'tokens_data.npy',mode='w+',dtype='float16',shape=tuple(dimensions['tokens']))
                        masks=np.lib.format.open_memmap(out/'tokens_mask.npy',mode='w+',dtype=bool,shape=(len(rows),patches.shape[1]))
                    state['shapes']=dimensions
                vectors[positions]=values
                if have_tokens:
                    tokens[positions]=patches
                    side=int(np.sqrt(patches.shape[1]))
                    if side*side!=patches.shape[1]:raise ValueError('nonsquare token grid')
                    yy,xx=np.meshgrid((np.arange(side)+.5)*224/side,(np.arange(side)+.5)*224/side,indexing='ij')
                    for index,(w,h) in zip(positions,sizes):
                        scale=min(224/w,224/h);width,height=w*scale,h*scale
                        mask=((xx>=(224-width)/2)&(xx<(224+width)/2)&(yy>=(224-height)/2)&(yy<(224+height)/2)).ravel()
                        if not mask.any():raise ValueError('no valid image tokens')
                        masks[index]=mask
                print('FEATURE BATCH',name,'rows',positions[0],'-',positions[-1],flush=True)
                positions.clear();pictures.clear();sizes.clear()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for source in batch.to_pylist():
                    index=targets.get(row_number);row_number+=1
                    if index is None:continue
                    raw=source['image']['bytes']
                    if hashlib.sha256(raw).hexdigest()!=rows[index]['image_sha256']:raise ValueError('training source image mismatch')
                    with Image.open(io.BytesIO(raw)) as image:
                        image=ImageOps.exif_transpose(image).convert('RGB');sizes.append(image.size)
                        pictures.append(ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127)))
                    positions.append(index)
                    if len(pictures)>=16:flush()
            flush()
            vectors.flush()
            if have_tokens:tokens.flush();masks.flush()
            state['shards'].append(shard['path']);atomic_json(marker,state)
            path.unlink();print('FEATURE SHARD COMPLETE',name,shard['path'],flush=True)
        del core,projection;gc.collect();torch.cuda.empty_cache()
        if vectors is None:raise ValueError('no feature rows')
        self.live=[vectors]+([tokens,masks] if have_tokens else [])
        if have_tokens:
            content={'signature':signature,'tokens_sha256':file_hash(out/'tokens_data.npy'),
                     'mask_sha256':file_hash(out/'tokens_mask.npy'),'storage':'float16 tokens; float32 head batches'}
            atomic_json(out/'token_identity.json',content)
            return vectors,BatchBank(tokens,content),BatchBank(masks,content)
        return vectors,None,None


def train_all(label_output,output):
    db,manifest=require_complete(label_output)
    try:rows,excluded=training_records(db)
    finally:db.close()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'records.json',rows);atomic_json(output/'excluded.json',excluded)
    atomic_json(output/'study.json',{'label_checkpoint_sha256':manifest['sha256'],'required_candidates':48,
                'token_storage':'float16 disk banks, expanded tofloat32 by minibatch',
                'grouping':'exact content groups; natural near-view/event independence unverified',
                'native_test':'excluded from fitting, selection and scoring',
                'reference':'VLM pseudo-label agreement, not verified urgency','runtime_promotion_allowed':False})
    print('ALL CLASSIFIERS START',len(rows),'usable unique train/dev images;48 unique planned combinations',flush=True)
    return compare(rows,rows,output,feature_loader=FullFeatureLoader(output),require_noncollapsed_selection=True,
                   token_encoders=('clip_b32','siglip2_b16','dino','convnextv2_tiny'))
