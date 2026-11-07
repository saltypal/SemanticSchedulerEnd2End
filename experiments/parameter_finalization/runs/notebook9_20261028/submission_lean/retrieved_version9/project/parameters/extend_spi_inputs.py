"""Extend the exposed SPI development cohort using already-cached LADI images.

No network access. New images inherit existing groups when near-duplicates;
ambiguous bridges are excluded. All inputs remain development-exposed.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

from .artifacts import write_json
from .contracts import MissionSpec
from .extractor import embedding_signature
from .minimal_vision import MinimalVisionProvider, prepare_image, INPUT_POLICY


def dhash(image):
    a=np.asarray(image.convert('L').resize((9,8)),dtype=float)
    return sum(int(v)<<k for k,v in enumerate((a[:,1:]>a[:,:-1]).ravel())),float(a.std())


def build(old_root, csv_path, checkpoint, output, maximum=120):
    old_root,csv_path,output=map(Path,(old_root,csv_path,output))
    if not 0<maximum<=120:raise ValueError('additional image cap must be1..120')
    if output.exists():raise FileExistsError('immutable output already exists')
    old_rows=json.loads((old_root/'panel.json').read_text(encoding='utf-8'))
    old_manifest=json.loads((old_root/'input_manifest.json').read_text(encoding='utf-8'))
    for name,expected in old_manifest['files'].items():
        if hashlib.sha256((old_root/name).read_bytes()).hexdigest()!=expected:raise ValueError('input checksum mismatch: '+name)
    with np.load(old_root/'features.npz',allow_pickle=False) as loaded:
        old_features=np.asarray(loaded['embeddings'],np.float32)
        if loaded['image_ids'].tolist()!=[r['image_id'] for r in old_rows]:raise ValueError('old feature order mismatch')
        signature=str(loaded['feature_signature'].item())
    rows=[dict(r) for r in old_rows]; entries=[]; seen_raw={r['image_id'] for r in rows};seen_png={r['image_sha256'] for r in rows}
    with zipfile.ZipFile(old_root/'images.zip') as archive:
        for row in rows:
            data=archive.read(row['image_path'])
            if hashlib.sha256(data).hexdigest()!=row['image_sha256']:raise ValueError('old image mismatch')
            with Image.open(io.BytesIO(data)) as image:h,std=dhash(image)
            entries.append((h,std,row['group'],row['split']))
    with csv_path.open(encoding='utf-8-sig',newline='') as handle: raw=list(csv.DictReader(handle))
    candidates=[]
    for row in raw:
        sha=row['source_sha256']
        if sha in seen_raw:continue
        native=json.loads(row['native_labels_json'])
        # LADI's affected-or-greater includes the minor-or-greater subset;
        # subtracting affected removes every genuinely minor-or-greater sample.
        stratum=('minor_or_greater' if native.get('buildings_minor_or_greater')
                 else 'affected' if native.get('buildings_affected_or_greater') or native.get('roads_damage') else 'other')
        path=csv_path.parent/row['image_path']
        if path.is_file():candidates.append((stratum,sha,path,row))
    lists={key:sorted([r for r in candidates if r[0]==key],key=lambda r:r[1]) for key in ('minor_or_greater','affected','other')}
    ordered=[]
    while any(lists.values()):
        for name in ('minor_or_greater','affected','other'):
            if lists[name]:ordered.append(lists[name].pop(0))
    selected=[];excluded=[]
    for stratum,raw_sha,path,source_row in ordered:
        if len(selected)>=maximum:break
        if hashlib.sha256(path.read_bytes()).hexdigest()!=raw_sha:
            excluded.append({'image_id':raw_sha,'reason':'cached source content mismatch'});continue
        with Image.open(path) as source:picture=prepare_image(source)
        stream=io.BytesIO();picture.save(stream,format='PNG');data=stream.getvalue();sha=hashlib.sha256(data).hexdigest()
        if sha in seen_png:continue
        h,std=dhash(picture)
        matches={(group,split) for previous,previous_std,group,split in entries
                 if min(std,previous_std)>10 and (h^previous).bit_count()<=2}
        if len(matches)>1:
            excluded.append({'image_id':raw_sha,'reason':'near-duplicate links multiple existing groups'});continue
        if matches:group,split=next(iter(matches))
        else:
            group='duplicate_component:'+raw_sha
            split='validation' if int(raw_sha[:8],16)%5==0 else 'train'
        row={'image_id':raw_sha,'image_sha256':sha,'image_path':'images/'+sha+'.png','group':group,
             'split':split,'source':'ladi','development_exposed':True}
        selected.append((row,data,stratum));rows.append(row);entries.append((h,std,group,split));seen_png.add(sha)
    if not selected:raise RuntimeError('no additional eligible cached LADI images')
    output.mkdir(parents=True)
    mission=MissionSpec('spi_reference','Review visible images.',(('visible','an image'),))
    provider=MinimalVisionProvider(checkpoint,{mission.mission_id:mission},device='cpu')
    embeddings=[]
    try:
        with zipfile.ZipFile(old_root/'images.zip') as archive:
            with Image.open(io.BytesIO(archive.read(old_rows[0]['image_path']))) as image:
                old_evidence=provider.extract_batch([image],[mission])[0]
        if embedding_signature(old_evidence)!=signature:raise ValueError('existing CLIP signature changed')
        difference=float(np.max(np.abs(np.asarray(old_evidence.embedding)-old_features[0])))
        if difference>1e-4:raise ValueError('existing CLIP feature mismatch')
        for start in range(0,len(selected),8):
            pictures=[]
            for _,data,_ in selected[start:start+8]:
                with Image.open(io.BytesIO(data)) as image:pictures.append(image.copy())
            evidence=provider.extract_batch(pictures,[mission]*len(pictures))
            if any(embedding_signature(e)!=signature for e in evidence):raise ValueError('mixed CLIP signatures')
            embeddings.extend(e.embedding for e in evidence)
            print('Additional cached image features',min(start+8,len(selected)),'/',len(selected),flush=True)
    finally:provider.close()
    with zipfile.ZipFile(output/'images.zip','w',zipfile.ZIP_STORED) as destination:
        with zipfile.ZipFile(old_root/'images.zip') as source:
            for row in old_rows:destination.writestr(row['image_path'],source.read(row['image_path']))
        for row,data,_ in selected:destination.writestr(row['image_path'],data)
    np.savez_compressed(output/'features.npz',image_ids=np.array([r['image_id'] for r in rows]),
        image_sha256=np.array([r['image_sha256'] for r in rows]),
        embeddings=np.concatenate((old_features,np.asarray(embeddings,np.float32))),feature_signature=np.array(signature))
    write_json(output/'panel.json',rows)
    write_json(output/'selection_audit.json',{'old':len(old_rows),'added':len(selected),'strata':
        {s:sum(item[2]==s for item in selected) for s in ('minor_or_greater','affected','other')},
        'excluded':excluded,'all_development_exposed':True,'event_grouping_verified':False,
        'new_image_source':'locally cached LADI subset; native labels used only for sampling',
        'old_feature_max_abs_difference':difference,'source_csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest()})
    write_json(output/'input_manifest.json',{'schema':'spi_inputs_v1','images':len(rows),'input_policy':INPUT_POLICY,
        'development_exposed':True,'native_labels_included':False,'feature_signature':signature,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}})
    print('Prepared bounded input',len(rows),'images at',output,flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-root',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--maximum',type=int,default=120)
    a=p.parse_args();build(a.old_root,a.csv,a.checkpoint,a.output,a.maximum)
