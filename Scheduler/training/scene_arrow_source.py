"""Revision-pinned official Arrow fallback for the broken HF row service.

This reads one declared shard, never an implicit complete dataset. Raw shard
bytes count toward the same source-download cap as selected images. The caller
must join/filter/group records before assigning scheduler roles.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import pyarrow as pa

DATASET='MITLL/LADI-v2-dataset'
REVISION='5f2dbfe8c466d32edafd1bab847ec5252309acdb'


def download_shard(destination,*,split='train',index=0,maximum_bytes=650_000_000):
    if split!='train' or type(index) is not int or not 0<=index<40:
        raise ValueError('explicit official training shard index 0..39 required')
    destination=Path(destination)
    if destination.exists():raise ValueError('preserve existing shard; new destination required')
    destination.parent.mkdir(parents=True,exist_ok=True)
    source=f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train/data-{index:05d}-of-00040.arrow'
    temporary=destination.with_suffix('.partial');total=0;sha=hashlib.sha256()
    try:
        with urllib.request.urlopen(source,timeout=45) as response,temporary.open('xb') as handle:
            declared=response.headers.get('Content-Length')
            if declared and int(declared)>maximum_bytes:raise ValueError('declared shard exceeds source-byte cap')
            while True:
                block=response.read(min(4_194_304,maximum_bytes-total+1))
                if not block:break
                total+=len(block)
                if total>maximum_bytes:raise ValueError('shard source-byte cap exceeded')
                handle.write(block);sha.update(block)
                if total//33_554_432!=(total-len(block))//33_554_432:
                    print('Official Arrow download bytes',total,flush=True)
        temporary.replace(destination)
    finally:
        receipt={'dataset':DATASET,'revision':REVISION,'split':split,'shard_index':index,
            'downloaded_bytes':total,'sha256':sha.hexdigest(),'complete':destination.exists(),
            'vision_producer_training_exposure':True}
        destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def annotated_rows(path):
    """Yield published labels plus exact image bytes; no model predictions."""
    with pa.memory_map(str(path),'r') as source:
        reader=pa.ipc.open_stream(source)
        for batch in reader:
            for row in batch.to_pylist():
                presence=row.get('buildings_any');damage=row.get('buildings_affected_or_greater')
                if type(presence) is not bool or type(damage) is not bool or (damage and not presence):continue
                image=row.get('image')
                if not isinstance(image,dict) or not isinstance(image.get('bytes'),bytes):
                    raise ValueError('official shard lacks embedded image bytes')
                yield {'image_bytes':image['bytes'],'image_id':hashlib.sha256(image['bytes']).hexdigest(),
                    'stratum':2 if damage else (1 if presence else 0),'damage_positive':damage,
                    'native_labels':{key:value for key,value in row.items() if key!='image'},
                    'source_split':'train','dataset_revision':REVISION}
