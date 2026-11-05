"""Cloud-only bounded data preparation using official row/image services."""
from pathlib import Path
import hashlib
import io
import json
import time
import urllib.parse
import numpy as np
from .artifacts import Run,write_json
from .data import DownloadBudget,add_duplicate_groups


def fetch_source(dataset,source,maximum,output,budget,callback=lambda:None):
    from PIL import Image
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    images=output/'images';images.mkdir(exist_ok=True)
    existing=output/'records.json';records=json.loads(existing.read_text()) if existing.exists() else []
    seen={r['source_row'] for r in records};failures=[]
    size_query=urllib.parse.urlencode({'dataset':dataset,'config':'default','split':'train'})
    size=json.loads(budget.get('https://datasets-server.huggingface.co/size?'+size_query,limit=2_000_000))
    total=next(s['num_rows'] for s in size['size']['splits'] if s['split']=='train')
    offsets=np.random.default_rng(17).permutation(np.arange(0,total,100)).tolist()
    write_json(output/'sampling.json',{'method':'uniform shuffled100-row pages','seed':17,'train_rows':total,'page_order':offsets})
    for offset in offsets:
        callback()
        if len(records)>=maximum:break
        query=urllib.parse.urlencode({'dataset':dataset,'config':'default','split':'train','offset':offset,'length':100})
        page=json.loads(budget.get('https://datasets-server.huggingface.co/rows?'+query,limit=8_000_000))
        rows=page.get('rows',[])
        if not rows:break
        for wrapper in rows:
            if len(records)>=maximum:break
            index=wrapper['row_idx']
            if index in seen:continue
            row=wrapper['row']
            try:
                raw=budget.get(row['image']['src']);picture=Image.open(io.BytesIO(raw));picture.verify()
                identity=hashlib.sha256(raw).hexdigest();path=images/(identity+'.img')
                if not path.exists():path.write_bytes(raw)
                labels={'building_damage':None,'road_damage':None,'affected_people':None,'structural_flooding':None}
                native={k:v for k,v in row.items() if k!='image'}
                if source=='ladi':
                    for name,key in (('building_damage','buildings_affected_or_greater'),('road_damage','roads_damage'),('structural_flooding','flooding_structures')):
                        value=row.get(key)
                        if type(value) is bool:labels[name]=int(value)
                elif source=='medic':
                    value=row.get('humanitarian')
                    if type(value) is int and 0<=value<=3:labels['affected_people']=int(value==0)
                    elif isinstance(value,str) and value in ('affected_injured_or_dead_people','infrastructure_and_utility_damage','not_humanitarian','rescue_volunteering_or_donation_effort'):
                        labels['affected_people']=int(value=='affected_injured_or_dead_people')
                records.append({'image_id':identity,'image_path':str(path.relative_to(output)),'source':source,
                    'source_row':index,'upstream_dataset':dataset,'upstream_split':'train',
                    'parent_id':identity,'group':source+':'+identity,'event':None,'labels':labels,
                    'native_labels':native,'label_definition':'published observation classification, not urgency',
                    'development_exposed':False})
                seen.add(index)
                if len(records)==20:
                    known={name:sum(r['labels'][name] is not None for r in records) for name in labels}
                    if not any(known.values()):raise RuntimeError('20-record annotation preflight has no supported labels')
                    write_json(output/'annotation_preflight.json',{'records':20,'supported_label_counts':known,
                              'image_bytes_verified':True,'event_ids_verified':False})
                    print(source,'20-record annotation/image preflight passed:',known,flush=True)
            except RuntimeError:raise
            except Exception as error:failures.append({'source_row':index,'error':type(error).__name__,'message':str(error)})
        write_json(existing,records)
        write_json(output/'progress.json',{'records':len(records),'offset':offset,'downloaded_bytes':budget.downloaded,'failures':len(failures)})
        print(source,'records',len(records),'downloaded bytes',budget.downloaded,flush=True)
        if len(rows)<100:break
    write_json(output/'failures.json',failures)
    return records


def fetch_ladi_shards(maximum,output,budget,callback=lambda:None):
    """Sample24 of40 pinned official Arrow shards within the20GB cap."""
    from huggingface_hub import HfApi,hf_hub_url
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from PIL import Image
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    images=output/'images';images.mkdir(exist_ok=True)
    info=HfApi().dataset_info('MITLL/LADI-v2-dataset',files_metadata=True)
    files=sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('data/train/data-') and s.rfilename.endswith('.arrow'))
    if not files:raise RuntimeError('official pinned train Arrow shards not found')
    files=[files[int(i)] for i in np.random.default_rng(17).permutation(len(files))[:24]]
    write_json(output/'upstream.json',{'dataset':'MITLL/LADI-v2-dataset','revision':info.sha,'files':files,
                                     'sampling':'seed17 random24 train shards; uniform rows within selected shard'})
    records=[];row_base=0;rng=np.random.default_rng(17)
    for file_index,name in enumerate(files):
        callback();path=output/('shard_'+str(file_index)+'.arrow')
        url=hf_hub_url('MITLL/LADI-v2-dataset',name,repo_type='dataset',revision=info.sha)
        budget.download_to(url,path,callback)
        with pa.memory_map(str(path),'r') as mapped:
            reader=ipc.open_stream(mapped);count=sum(batch.num_rows for batch in reader)
        quota=min(count,(maximum-len(records)+len(files)-file_index-1)//(len(files)-file_index))
        selected=set(rng.choice(count,quota,replace=False).tolist());index=0
        mapped=pa.memory_map(str(path),'r');reader=ipc.open_stream(mapped)
        for batch in reader:
            callback()
            for row in batch.to_pylist():
                current=index;index+=1
                if current not in selected:continue
                raw=row['image']['bytes']
                if raw is None:raise ValueError('official parquet image has no embedded bytes')
                Image.open(io.BytesIO(raw)).verify()
                identity=hashlib.sha256(raw).hexdigest();destination=images/(identity+'.img')
                if not destination.exists():destination.write_bytes(raw)
                native={k:v for k,v in row.items() if k!='image'}
                labels={'building_damage':None,'road_damage':None,'affected_people':None,'structural_flooding':None}
                for mission,key in (('building_damage','buildings_affected_or_greater'),('road_damage','roads_damage'),('structural_flooding','flooding_structures')):
                    if type(row.get(key)) is bool:labels[mission]=int(row[key])
                records.append({'image_id':identity,'image_path':str(destination.relative_to(output)),
                    'source':'ladi','source_row':None,'source_shard':name,'source_shard_row':current,'upstream_dataset':'MITLL/LADI-v2-dataset',
                    'upstream_revision':info.sha,'upstream_split':'train','parent_id':identity,'group':'ladi:'+identity,
                    'event':None,'labels':labels,'native_labels':native,'development_exposed':False,
                    'label_definition':'published image-level building affected/road damage; not urgency'})
                if len(records)==20:
                    audit={key:sum(r['labels'][key] is not None for r in records) for key in labels}
                    if not all(audit[key]==20 for key in ('building_damage','road_damage','structural_flooding')):
                        raise RuntimeError('official LADI label schema failed20-record preflight')
                    write_json(output/'annotation_preflight.json',{'records':20,'known_labels':audit,'revision':info.sha})
                if len(records)%100==0:
                    write_json(output/'records.json',records);print('LADI parents:',len(records),flush=True)
        row_base+=count
        write_json(output/'records.json',records)
        reader.close();mapped.close();path.unlink() # Only this explicitly created disposable shard; images/results persist.
    return records


def prepare(output,maximum=10000,download_bytes=20_000_000_000,seconds=7200,exposure_ledger=None):
    if not 1<=maximum<=10000 or download_bytes>20_000_000_000:raise ValueError('cloud data budget exceeds approval')
    if exposure_ledger is None:raise ValueError('known-exposure ledger is required before cloud preparation')
    run=Run(output,{'stage':'prepare','maximum_parents':maximum,'download_bytes':download_bytes,'seconds':seconds})
    budget=DownloadBudget(download_bytes);all_records=[];availability={}
    ladi_count=min(4800,int(maximum*.48))
    for dataset,source,count in [('MITLL/LADI-v2-dataset','ladi',ladi_count),('QCRI/MEDIC','medic',maximum-ladi_count)]:
        with run.stage(source+'_official_sample'):
            try:
                records=(fetch_ladi_shards(count,run.path/'data'/source,budget,run.budget) if source=='ladi'
                         else fetch_source(dataset,source,count,run.path/'data'/source,budget,run.budget))
                for row in records:row['image_path']=str((run.path/'data'/source/row['image_path']).relative_to(run.path))
                all_records+=records;availability[source]={'available':True,'records':len(records)}
            except Exception as error:
                partial=run.path/'data'/source/'records.json'
                recovered=json.loads(partial.read_text()) if partial.exists() else []
                for row in recovered:
                    row['image_path']=str((run.path/'data'/source/row['image_path']).relative_to(run.path))
                all_records+=recovered
                availability[source]={'available':bool(recovered),'complete':False,
                                      'records':len(recovered),'reason':str(error)}
                print('Source failed:',source,str(error),flush=True)
    # Preserve conflicting annotations as unknown; exact files are one parent.
    unique={};conflicts=[]
    for row in all_records:
        identity=row['image_id']
        if identity not in unique:unique[identity]=row;continue
        existing=unique[identity]
        for mission,value in row['labels'].items():
            previous=existing['labels'].get(mission)
            if previous is not None and value is not None and previous!=value:
                existing['labels'][mission]=None
                conflicts.append({'image_id':identity,'mission':mission,'values':[previous,value]})
            elif previous is None and value is not None and not any(
                    c['image_id']==identity and c['mission']==mission for c in conflicts):
                existing['labels'][mission]=value
    all_records=list(unique.values());write_json(run.path/'data/conflicting_annotations.json',conflicts)
    # Duplicate grouping uses actual pixels, but final paths stay portable between Kaggle stages.
    for row in all_records:row['image_path']=str(run.path/row['image_path'])
    all_records=add_duplicate_groups(all_records)
    from .exposure import apply_ledger
    ledger=json.loads(Path(exposure_ledger).read_text())
    write_json(run.path/'data/exposure_audit.json',apply_ledger(all_records,ledger))
    write_json(run.path/'data/exposure_ledger.json',ledger)
    for row in all_records:row['image_path']=str(Path(row['image_path']).relative_to(run.path))
    write_json(run.path/'data/records.json',all_records)
    availability.update({'incidents1m':{'available':False,'reason':'official annotations require access request'},
                         'xbd':{'available':False,'reason':'official image/annotation access not supplied'},
                         'rescuenet':{'available':False,'reason':'native paired image-mask manifest not supplied'}})
    write_json(run.path/'data/source_availability.json',availability)
    write_json(run.path/'status.json',{'status':'complete' if all_records else 'failed','parents':len(all_records),
               'downloaded_bytes':budget.downloaded,'event_grouping_verified':False,'source_availability':availability})
    if not all_records:raise RuntimeError('no official images acquired')
    return run.path


def load_portable_records(manifest):
    manifest=Path(manifest).resolve();records=json.loads(manifest.read_text())
    # manifest is <run>/data/records.json; relative paths start at the run root.
    root=manifest.parent.parent
    for row in records:
        path=Path(row['image_path'])
        if not path.is_absolute():path=root/path
        path=path.resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('unsafe/missing portable data path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_id']:
            raise ValueError('portable image checksum mismatch')
        row['image_path']=str(path)
    return records
