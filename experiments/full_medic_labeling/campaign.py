"""Pinned MEDIC shard stream and durable labels. Classifiers run only after completion."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
from PIL import Image

from experiments.spi_classifier_repair.resume_store import atomic_json,file_hash,fingerprint
from experiments.full_medic_labeling.teacher import SYSTEM_PROMPT,PROMPT_SHA256,MODEL_ID,MODEL_REVISION

DATA_REVISION='bc2e6f3c99f511ce89055cad5a661c69e43ec6de'
LEVELS=['Low','Medium','High','Critical']
SPLIT_COUNTS={'train':49353,'dev':6157,'test':15688}


class LabelStore:
    """Single writer; backup is a consistent resume artifact, never a live DB copy."""
    def __init__(self,path,identity):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);'
          'CREATE TABLE IF NOT EXISTS predictions(sha TEXT PRIMARY KEY,result TEXT NOT NULL,embedding BLOB);'
          'CREATE TABLE IF NOT EXISTS items(id TEXT PRIMARY KEY,split TEXT,shard TEXT,row_index INTEGER,sha TEXT,source TEXT);')
        old=self.db.execute("SELECT v FROM meta WHERE k='identity'").fetchone()
        if old and json.loads(old[0]) != identity:
            self.db.close();raise ValueError('incompatible label checkpoint identity')
        self.db.execute("INSERT OR IGNORE INTO meta VALUES('identity',?)",(json.dumps(identity,sort_keys=True),))
        self.db.commit()

    def has_item(self,identity):
        return self.db.execute('SELECT 1 FROM items WHERE id=?',(identity,)).fetchone() is not None

    def prediction(self,sha):
        row=self.db.execute('SELECT result,embedding FROM predictions WHERE sha=?',(sha,)).fetchone()
        return None if row is None else (json.loads(row[0]),row[1])

    def add(self,item,result,embedding=None):
        vector=None if embedding is None else np.asarray(embedding,dtype=np.float32).tobytes()
        if vector is not None and (len(vector)!=512*4 or not np.isfinite(embedding).all()):
            raise ValueError('invalid CLIP embedding')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO predictions VALUES(?,?,?)',
                            (item['image_sha256'],json.dumps(result,allow_nan=False),vector))
            self.db.execute('INSERT OR IGNORE INTO items VALUES(?,?,?,?,?,?)',
              (item['item_id'],item['source_split'],item['source_shard'],item['shard_row'],
               item['image_sha256'],json.dumps(item,allow_nan=False)))

    def export(self,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        counts=Counter();available=0;attempts=0
        partial=output/'labels.jsonl.partial';csv_partial=output/'labels.csv.partial'
        with partial.open('w',encoding='utf8') as target,csv_partial.open('w',encoding='utf8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['item_id','source_split','image_sha256','level','available','status'])
            writer.writeheader()
            for source,result in self.db.execute('SELECT i.source,p.result FROM items i JOIN predictions p ON i.sha=p.sha ORDER BY i.id'):
                row={**json.loads(source),**json.loads(result)}
                target.write(json.dumps(row,allow_nan=False)+'\n')
                writer.writerow({k:row.get(k) for k in writer.fieldnames})
                attempts+=1;counts[row.get('level') or 'unavailable']+=1;available+=row.get('available') is True
        partial.replace(output/'labels.jsonl');csv_partial.replace(output/'labels.csv')
        unique=self.db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        progress={'attempted_rows':attempts,'available_label_rows':available,'unique_image_hashes':unique,
                  'source_split_counts':dict(self.db.execute('SELECT split,COUNT(*) FROM items GROUP BY split').fetchall()),
                  'class_counts':dict(counts),'target_rows':sum(SPLIT_COUNTS.values()),
                  'coverage_complete':attempts==sum(SPLIT_COUNTS.values()),
                  'all_labels_available':available==sum(SPLIT_COUNTS.values()),
                  'independent_urgency_validated':False}
        atomic_json(output/'progress.json',progress)
        temporary=output/'full_medic_checkpoint.partial'
        dest=sqlite3.connect(temporary)
        self.db.backup(dest);dest.close();temporary.replace(output/'full_medic_checkpoint.sqlite')
        atomic_json(output/'checkpoint_manifest.json',{'sha256':file_hash(output/'full_medic_checkpoint.sqlite'),
            'bytes':(output/'full_medic_checkpoint.sqlite').stat().st_size,
            'identity':json.loads(self.db.execute("SELECT v FROM meta WHERE k='identity'").fetchone()[0]),
            'progress':progress})
        return progress


def download_shard(item,path):
    import requests
    path=Path(path);temporary=path.with_suffix('.partial')
    if path.exists() and file_hash(path)==item['sha256']:
        return path
    if shutil.disk_usage(path.parent).free < item['bytes']+3_000_000_000:
        raise RuntimeError('insufficient shard space plus3GB reserve')
    url=f"https://huggingface.co/datasets/QCRI/MEDIC/resolve/{DATA_REVISION}/{item['path']}"
    digest=hashlib.sha256();size=0
    with requests.get(url,stream=True,timeout=(30,120)) as response,temporary.open('wb') as target:
        response.raise_for_status()
        for block in response.iter_content(4*1024*1024):
            if block:target.write(block);digest.update(block);size+=len(block)
    if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
        raise ValueError('parquet source checksum mismatch')
    temporary.replace(path);return path


def inventory():
    from huggingface_hub import HfApi
    info=HfApi().dataset_info('QCRI/MEDIC',revision=DATA_REVISION,files_metadata=True)
    shards=[]
    for item in info.siblings:
        if item.rfilename.endswith('.parquet'):
            split=item.rfilename.split('/')[-1].split('-')[0]
            shards.append({'path':item.rfilename,'split':split,'bytes':item.size,'sha256':item.lfs.sha256})
    shards.sort(key=lambda r:({'train':0,'dev':1,'test':2}[r['split']],r['path']))
    if len(shards)!=26:raise ValueError('unexpected pinned MEDIC shard inventory')
    return {'repo':'QCRI/MEDIC','revision':DATA_REVISION,'splits':SPLIT_COUNTS,'shards':shards}


def run(output,teachers,config,parameter_producers=None,profiles=None):
    """Each worker exclusively owns one teacher/GPU. Only the main thread writes."""
    import pyarrow.parquet as pq
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={'data_revision':DATA_REVISION,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
              'system_prompt_sha256':PROMPT_SHA256,'settings':{'max_new_tokens':192,'max_pixels':262144,'thinking':False},
              'implementation_sha256':file_hash(__file__),'teacher_source_sha256':file_hash(Path(__file__).with_name('teacher.py')),
              'producer_identity':config['producer_identity']}
    store=LabelStore(output/'work.sqlite',identity)
    atomic_json(output/'identity.json',identity);(output/'SYSTEM_PROMPT.txt').write_text(SYSTEM_PROMPT,encoding='utf8')
    inv=inventory();atomic_json(output/'dataset_inventory.json',inv)
    scratch=Path('/kaggle/temp/full_medic_shards');scratch.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();deadline=started+config['wall_seconds'];processed=0

    def work(slot,raw):
        image=Image.open(io.BytesIO(raw));image.load()
        result=teachers[slot].review(image)
        embedding=None
        if parameter_producers is not None:
            try:
                parameters,embedding=parameter_producers[slot].extract(image)
                result['parameters']=parameters
            except Exception as error:
                result['parameters']={'status':'failed','reason':str(error)}
        return result,embedding

    futures={};reason='all_rows_visited'
    def finish(slot):
        if slot not in futures:return
        future,item=futures.pop(slot)
        try:result,embedding=future.result()
        except Exception as e:result,embedding={'available':False,'level':None,'status':'failed','reason':str(e)},None
        store.add(item,result,embedding)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            stop=False
            for shard in inv['shards']:
                if stop:break
                print('SHARD START',shard['path'],flush=True)
                path=download_shard(shard,scratch/Path(shard['path']).name)
                row_number=0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                    for row in batch.to_pylist():
                        item_id=shard['path']+':'+str(row_number);row_number+=1
                        if store.has_item(item_id):continue
                        if time.perf_counter()>=deadline or processed>=config['max_new_rows']:
                            reason='bounded_session_limit';stop=True;break
                        image=row.pop('image');raw=image.get('bytes')
                        if not raw:raise ValueError('embedded source image bytes missing')
                        sha=hashlib.sha256(raw).hexdigest()
                        item={'item_id':item_id,'source_split':shard['split'],'source_shard':shard['path'],
                              'shard_row':row_number-1,'image_sha256':sha,'source_metadata':row}
                        for pending_slot,(_,pending_item) in list(futures.items()):
                            if pending_item['image_sha256']==sha:finish(pending_slot)
                        cached=store.prediction(sha)
                        if cached is not None:
                            # Reuse one teacher decision for an exact duplicate, preserving every source row.
                            store.add(item,cached[0],None if cached[1] is None else np.frombuffer(cached[1],dtype=np.float32))
                        else:
                            slot=processed%2;finish(slot);futures[slot]=(pool.submit(work,slot,raw),item)
                        processed+=1
                        if processed%100==0:print('LABELED NEW ROWS',processed,'seconds',round(time.perf_counter()-started,1),flush=True)
                        if processed%500==0:
                            for slot in list(futures):finish(slot)
                            print('CHECKPOINT',store.export(output),flush=True)
                    if stop:break
                for slot in list(futures):finish(slot)
                store.export(output)
                # Only delete the verified disposable active shard, never saved results.
                if path.resolve().parent!=scratch.resolve():raise ValueError('unexpected shard cleanup path')
                path.unlink()
                print('SHARD END',shard['path'],flush=True)
    except BaseException:
        reason='execution_error'
        raise
    finally:
        for slot in list(futures):finish(slot)
        progress=store.export(output)
        if profiles is not None:
            from experiments.full_medic_labeling.parameters_cache import export_context_records
            export_context_records(store,output,profiles)
        atomic_json(output/'terminal.json',{'status':('failed' if reason=='execution_error' else
                    'complete' if progress['coverage_complete'] else 'partial'),
                    'reason':reason,'seconds':time.perf_counter()-started,'progress':progress})
        store.db.close()
    return progress
