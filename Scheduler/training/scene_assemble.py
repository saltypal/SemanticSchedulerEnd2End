"""Assemble existing official test images and a pinned train shard, without HTTP."""
from pathlib import Path
from io import BytesIO
import hashlib,json
import numpy as np
from PIL import Image
from Scheduler.training.scene_arrow_source import annotated_rows,REVISION
from Scheduler.training.scene_corpus import split_groups
from Scheduler.training.scene_prepare import write_json


def fingerprint(raw):
    with Image.open(BytesIO(raw)) as image:
        a=np.asarray(image.convert('L').resize((9,8)))
        bits=(a[:,1:]>a[:,:-1]).flatten()
        return sum(int(v)<<i for i,v in enumerate(bits)),float(np.asarray(image.convert('L')).std())


def assemble(test_root,shards,output,*,excluded=(),historical_fingerprints=()):
    test_root=Path(test_root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'images').mkdir();seen=set(excluded);fps=list(historical_fingerprints)
    selected=[];counts={'test':[0,0,0],'train':[0,0,0]};rejected=[]
    def add(row,raw,split):
        identity=hashlib.sha256(raw).hexdigest();stratum=row['stratum']
        if counts[split][stratum]>=40:return
        if identity in seen:rejected.append({'image_id':identity,'reason':'exact_or_exposed'});return
        fp,std=fingerprint(raw)
        if any(min(std,old_std)>10 and (fp^old_fp).bit_count()<=2 for old_fp,old_std in fps):
            rejected.append({'image_id':identity,'reason':'near_duplicate'});return
        seen.add(identity);fps.append((fp,std));counts[split][stratum]+=1
        image_path='images/'+identity+'.jpg';(output/image_path).write_bytes(raw)
        selected.append({'image_id':identity,'image_path':image_path,'parent_id':identity,
            'group':'ladi:'+identity,'stratum':stratum,'damage_positive':row['damage_positive'],
            'native_labels':row['native_labels'],'source_split':split,'dataset_revision':REVISION,
            'source_row':row.get('source_row'),'event':None,'fingerprint':{'dhash':str(fp),'std':std}})
    for row in json.loads((test_root/'records.json').read_text()):
        if row['dataset_revision']!=REVISION:raise ValueError('source revision mismatch')
        raw=(test_root/row['image_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['image_id']:raise ValueError('test image checksum differs')
        add(row,raw,'test')
    receipts=[]
    for shard in map(Path,shards):
        receipt=json.loads(shard.with_suffix('.receipt.json').read_text())
        if not receipt['complete'] or receipt['revision']!=REVISION:raise ValueError('unverified train shard')
        sha=hashlib.sha256()
        with shard.open('rb') as handle:
            for block in iter(lambda:handle.read(4_194_304),b''):sha.update(block)
        if sha.hexdigest()!=receipt['sha256']:raise ValueError('train shard checksum differs')
        receipts.append(receipt)
        for row in annotated_rows(shard):
            add(row,row['image_bytes'],'train')
            if counts['train']==[40,40,40]:break
    rows=split_groups(selected);write_json(output/'records.json',rows)
    complete=counts=={'test':[40,40,40],'train':[40,40,40]}
    write_json(output/'data_receipt.json',{'status':'complete' if complete else 'partial','counts':counts,
        'role_counts':{role:sum(r['role']==role for r in rows) for role in ('train','validation','pilot_eval')},
        'vision_producer_training_exposure':True,'revision':REVISION,'shards':receipts,
        'exclusions':rejected,'scope':'scheduler held-out; frozen vision producer may have source-train exposure'})
    if not complete:raise ValueError('insufficient unexposed distinct images per source/stratum; partial cohort preserved')
    return rows


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root',required=True)
    parser.add_argument('--shards',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--historical-inputs',required=True)
    args=parser.parse_args()
    history=json.loads(Path(args.historical_inputs).read_text())
    print('START cohort assembly; published labels and exact bytes',flush=True)
    rows=assemble(args.test_root,args.shards,args.output,excluded=history['excluded_image_ids'],
        historical_fingerprints=[(int(r['dhash']),r['std']) for r in history['fingerprints']])
    print('END cohort assembly',len(rows),'images',args.output,flush=True)


if __name__=='__main__':main()
