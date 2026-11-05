"""Bounded official-source adapters. Labels are evaluator/training-only records."""
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import urllib.request
import urllib.parse


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class DownloadBudget:
    def __init__(self, maximum_bytes):
        self.maximum_bytes,self.downloaded=maximum_bytes,0

    def get(self,url,limit=30_000_000):
        request=urllib.request.Request(url,headers={'User-Agent':'parameters-research/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            chunks=[]
            while True:
                block=response.read(min(1024*1024,limit+1))
                if not block:break
                self.downloaded+=len(block);limit-=len(block)
                if self.downloaded>self.maximum_bytes or limit<0:
                    raise RuntimeError('declared download budget exceeded')
                chunks.append(block)
        return b''.join(chunks)

    def download_to(self,url,path,callback=lambda:None):
        """Stream an official shard without retaining a shard-sized byte buffer."""
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        request=urllib.request.Request(url,headers={'User-Agent':'parameters-research/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response,path.open('xb') as handle:
            transferred=0
            while True:
                callback();block=response.read(1024*1024)
                if not block:break
                self.downloaded+=len(block);transferred+=len(block)
                if self.downloaded>self.maximum_bytes:raise RuntimeError('declared download budget exceeded')
                handle.write(block)
                if transferred%(64*1024*1024)==0:print('Shard bytes received:',transferred,flush=True)
        return transferred


def ladi_records(manifest,limit=100):
    from Evaluation.decoder_datasets import ladi_manifest_records
    native,excluded=ladi_manifest_records(manifest,limit=limit)
    output=[]
    for row in native:
        # The earlier adapter names its normalized tasks; do not use prototype priorities.
        labels={'building_damage':row['labels']['building_damage_binary'],
                'road_damage':None,'affected_people':None,
                'structural_flooding':row['labels']['structural_flooding_transfer']}
        output.append(dict(image_id=row['image_id'],image_path=row['image_path'],source='ladi',
                           parent_id=row['parent_id'],group=row['group'],event=None,labels=labels,
                           label_definition='native LADI observation claims; not urgency',development_exposed=True))
    # Road labels are available in the source CSV even though the historical adapter is building-focused.
    with Path(manifest).open(encoding='utf-8-sig',newline='') as handle:
        original={row['source_sha256']:json.loads(row['native_labels_json']) for row in csv.DictReader(handle)}
    for row in output:
        value=original[row['image_id']].get('roads_damage')
        row['labels']['road_damage']=int(value) if value in (True,False,0,1) and value is not None else None
    return output,excluded


def balanced_ladi_records(manifest,limit=100):
    records,excluded=ladi_records(manifest,limit=100000)
    selected={}
    # Deterministic label-stratified development sampling; not a fresh final split.
    for claim in ('building_damage','road_damage'):
        for value in (0,1):
            for row in sorted((r for r in records if r['labels'][claim]==value),key=lambda r:r['image_id'])[:max(1,limit//4)]:
                selected[row['image_id']]=row
    for row in sorted(records,key=lambda r:r['image_id']):
        if len(selected)>=limit:break
        selected.setdefault(row['image_id'],row)
    return list(selected.values())[:limit],excluded


def medic_records(output,limit=80,budget=None):
    """Sample API records and image URLs; never download a full parquet shard."""
    from PIL import Image
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    budget=budget or DownloadBudget(1_000_000_000)
    result,failures=[],[]
    labels_map={'disaster_types':['earthquake','fire','flood','hurricane','landslide','not_disaster','other_disaster'],
                'humanitarian':['affected_people','infrastructure','not_humanitarian','rescue_effort'],
                'informative':['informative','not_informative'],
                'damage_severity':['little_or_none','mild','severe']}
    api='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':'QCRI/MEDIC','config':'default','split':'train','offset':0,'length':100})
    began=time.monotonic()
    for offset in (0,12000,24000,36000,48000):
        if len(result)>=limit or time.monotonic()-began>=300:break
        page_start=len(result)
        page_quota=max(1,(limit+4)//5)
        page_url=api.replace('offset=0','offset='+str(offset))
        payload=json.loads(budget.get(page_url,limit=5_000_000))
        rows=payload.get('rows',[])
        if not rows:break
        for wrapper in rows:
            if len(result)>=limit or len(result)-page_start>=page_quota:break
            row=wrapper['row'];index=wrapper.get('row_idx')
            if any(r.get('source_row')==index for r in result) or any(f.get('source_row')==index for f in failures):continue
            try:
                image_field=row['image'];url=image_field['src']
                raw=budget.get(url);image=Image.open(io.BytesIO(raw));image.verify()
                identity=hashlib.sha256(raw).hexdigest();path=output/(identity+'.img')
                if not path.exists():path.write_bytes(raw)
                native={}
                for key,names in labels_map.items():
                    value=row.get(key)
                    if type(value) is int and 0<=value<len(names):native[key]=names[value]
                    elif isinstance(value,str) and value in names:native[key]=value
                    else:native[key]=None
                claims={'building_damage':None,'road_damage':None,
                        'affected_people':None if native['humanitarian'] is None else int(native['humanitarian']=='affected_people'),
                        'structural_flooding':None}
                result.append(dict(image_id=identity,image_path=str(path.resolve()),source='medic',source_row=index,
                                   parent_id=identity,group='medic:'+identity,event=None,
                                   unverified_source_event=row.get('event_name'),labels=claims,native_labels=native,
                                   label_definition='native humanitarian classification; not individual distress',development_exposed=True))
            except Exception as error:
                failures.append(dict(source_row=index,error=type(error).__name__,message=str(error)))
        if not rows or len(rows)<100:break
    return result,failures,{'downloaded_bytes':budget.downloaded,'source':'QCRI/MEDIC datasets-server train rows','source_api':api,'event_groups_verified':False}


def add_duplicate_groups(records):
    """Conservative exact/dHash components; no inferred natural asset identity."""
    from PIL import Image
    import numpy as np
    parent=list(range(len(records)));hashes=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(records):
        with Image.open(row['image_path']) as image:
            a=np.asarray(image.convert('L').resize((9,8)),dtype=float)
        bits=a[:,1:]>a[:,:-1];h=sum(int(v)<<k for k,v in enumerate(bits.ravel()))
        hashes.append((h,float(a.std())))
        for j,other in enumerate(records[:i]):
            same=(row['parent_id']==other['parent_id'] or row.get('group')==other.get('group'))
            near=min(hashes[i][1],hashes[j][1])>10 and (h^hashes[j][0]).bit_count()<=2
            if same or near:parent[find(i)]=find(j)
    for i,row in enumerate(records):
        row['group']='duplicate_component:'+records[find(i)]['image_id']
    return records
