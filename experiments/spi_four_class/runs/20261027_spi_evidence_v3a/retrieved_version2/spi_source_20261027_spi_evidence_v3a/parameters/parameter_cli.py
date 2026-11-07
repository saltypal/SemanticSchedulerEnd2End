"""Extract compact parameters and versioned headers for a causal UE image stream."""
import argparse
from dataclasses import asdict
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .artifacts import write_json
from .contracts import MissionSpec
from .extractor import SenderMemory
from .minimal_parameters import FinalParameterExtractor
from .minimal_vision import MinimalVisionProvider
from .scene_measurements import YOLOWorldSceneProvider,UnavailableSceneProvider
from .spi import SPIClassifier


def load_memory(path,source_id,mission_id,mission_version,capacity=32):
    memory=SenderMemory(source_id,mission_id,capacity)
    if path is None or not Path(path).exists():return memory
    path=Path(path)
    if path.stat().st_size>5_000_000:raise ValueError('KB exceeds five megabytes')
    obj=json.loads(path.read_text(encoding='utf-8'))
    if (obj.get('schema')!='better_ue_kb_v1' or obj.get('source_id')!=source_id or
            obj.get('mission_id')!=mission_id or obj.get('mission_version')!=mission_version):
        raise ValueError('KB source/mission/schema mismatch')
    if obj.get('capacity')!=capacity or not isinstance(obj.get('entries'),list) or len(obj['entries'])>capacity:
        raise ValueError('invalid KB capacity or entries')
    version,last=obj.get('version'),obj.get('last_sequence')
    if type(version) is not int or version<0 or type(last) is not int or last < -1 or version<len(obj['entries']):
        raise ValueError('invalid KB lifecycle')
    for item in obj['entries']:
        vector=item.get('embedding');signature=item.get('embedding_signature')
        if vector is not None:
            vector=np.asarray(vector,dtype=float)
            if vector.ndim!=1 or vector.size!=512 or not np.isfinite(vector).all() or np.linalg.norm(vector)<=1e-10:
                raise ValueError('invalid KB embedding')
        if not isinstance(signature,str) or not isinstance(item.get('probabilities'),dict):raise ValueError('invalid KB entry')
        memory.entries.append({'embedding':vector,'embedding_signature':signature,'probabilities':item['probabilities']})
    memory.version=version;memory.last_sequence=last
    return memory


def save_memory(path,memory,mission_version):
    if path is None:return
    entries=[{'embedding':None if row['embedding'] is None else row['embedding'].tolist(),
              'embedding_signature':row['embedding_signature'],'probabilities':row['probabilities']} for row in memory.entries]
    write_json(path,{'schema':'better_ue_kb_v1','source_id':memory.source_id,'mission_id':memory.mission_id,'mission_version':mission_version,
        'capacity':memory.capacity,'version':memory.version,'last_sequence':memory.last_sequence,'entries':entries})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',nargs='+',required=True);p.add_argument('--mission',required=True)
    p.add_argument('--clip-checkpoint',required=True);p.add_argument('--model',required=True)
    p.add_argument('--yolo-checkpoint');p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--ue-id',default='ue01');p.add_argument('--source-id',type=int,default=1)
    p.add_argument('--session-id',type=int,default=1);p.add_argument('--message-base',type=int)
    p.add_argument('--profile-id',type=int,default=1);p.add_argument('--allowed-delay-ms',type=int)
    p.add_argument('--capture-time-ms',type=int);p.add_argument('--timestamp-trusted',action='store_true')
    p.add_argument('--kb');p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('output already exists; choose a new run path')
    mission=MissionSpec('cli_mission',a.mission,(('mission',a.mission),),allowed_delay_ms=a.allowed_delay_ms)
    if a.kb and Path(a.kb).resolve()==Path(a.output).resolve():raise ValueError('output and KB paths must differ')
    memory=load_memory(a.kb,a.ue_id,mission.mission_id,mission.version)
    provider=MinimalVisionProvider(a.clip_checkpoint,{mission.mission_id:mission},device=a.device)
    scene=YOLOWorldSceneProvider(a.yolo_checkpoint,device=a.device) if a.yolo_checkpoint else UnavailableSceneProvider('YOLO checkpoint not supplied')
    try:
        classifier=SPIClassifier.load(a.model)
        images=[]
        for path in a.images:
            with Image.open(path) as image:images.append(image.copy())
        now=int(time.time()*1000);first_sequence=memory.last_sequence+1
        message_base=first_sequence+1 if a.message_base is None else a.message_base
        metadata=[{'source_id':a.ue_id,'source_numeric_id':a.source_id,'session_id':a.session_id,
                   'message_id':message_base+i,'profile_id':a.profile_id,'sequence':first_sequence+i,
                   'capture_time_ms':a.capture_time_ms,'timestamp_trusted':a.timestamp_trusted,'now_ms':now}
                  for i in range(len(images))]
        extractor=FinalParameterExtractor(provider,classifier,scene)
        pairs=extractor.transmit_batch(images,[mission]*len(images),metadata,[memory]*len(images))
        rows=[]
        for path,(bundle,wire) in zip(a.images,pairs):
            rows.append({'image':str(Path(path).resolve()),'parameters':bundle['compact'],
                'header_bytes':len(wire),'header_hex':wire.hex(),
                'audit':{'spi':bundle['spi'],'scene':asdict(bundle['scene']),'relevance':bundle['relevance'],
                         'novelty':bundle['novelty'],'capture':bundle['capture'],
                         'mission':{'id':mission.mission_id,'version':mission.version,'paragraph':mission.paragraph},
                         'feature_signature':bundle['feature_signature']}})
        # Persist output first. The KB update is committed only after all
        # observations were serialized, and must be persisted atomically.
        result={'schema':'better_parameter_cli_v2','status':'pending_kb_commit','records':rows,'kb_version':memory.version}
        write_json(a.output,result)
        save_memory(a.kb,memory,mission.version)
        result['status']='complete';write_json(a.output,result)
        print(json.dumps({'status':'complete','images':len(rows),'output':str(Path(a.output).resolve()),
            'kb_version':memory.version,'header_bytes':len(pairs[0][1]) if pairs else 0},indent=2))
    finally:provider.close()

if __name__=='__main__':main()
