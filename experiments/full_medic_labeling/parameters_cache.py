"""Research image cache; contextual stream replay is explicit and deterministic."""
from dataclasses import asdict
import time
import numpy as np

from parameters.contracts import MissionSpec
from parameters.minimal_vision import MinimalVisionProvider
from parameters.scene_measurements import YOLOWorldSceneProvider


class ImageParameterProducer:
    def __init__(self,clip_path,yolo_path,profiles,device):
        self.missions={r['mission_id']:MissionSpec(r['mission_id'],r['paragraph'],
                       tuple(tuple(c) for c in r['conditions'])) for r in profiles['missions']}
        self.visual=MinimalVisionProvider(clip_path,list(self.missions.values()),device)
        self.detector=YOLOWorldSceneProvider(yolo_path,device=device)
        self.device=device
        self.visual.provider.load();self.detector.load()
        self.text={k:self.visual.context_embedding(m) for k,m in self.missions.items()}

    def extract(self,image):
        import torch
        def clock():
            if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)
            return time.perf_counter()
        start=clock();mission=next(iter(self.missions.values()))
        evidence=self.visual.extract_batch([image],[mission])[0]
        vector=np.asarray(evidence.embedding,dtype=np.float32)
        clip_end=clock();scene=self.detector.measure(image);scene_end=clock()
        result={'status':'ok' if scene.status=='ok' else 'partial',
                'scene':asdict(scene),
                'mission_relevance':{k:float(vector@v) for k,v in self.text.items()},
                'relevance_interpretation':'raw CLIP cosine, not probability',
                'capture_metadata':{'capture_time_ms':None,'timestamp_trusted':False,
                                    'reason':'no verified capture timestamp supplied'},
                'timing_seconds':{'clip_image':clip_end-start,'scene':scene_end-clip_end,
                                  'total':scene_end-start},
                'producer':evidence.provenance,
                'flops':None,'flops_reason':'per-image FLOPs not instrumented; latency is measured separately',
                'spi_source':'offline VLM teacher, not a ready student classifier',
                'runtime_promotion_allowed':False}
        return result,vector


def export_context_records(store,output,profiles):
    """Query-before-commit over a declared engineering stream, not capture order."""
    import json
    from pathlib import Path
    output=Path(output)
    temporary=output/'context_parameters.jsonl.partial'
    histories={};sequence={}
    with temporary.open('w',encoding='utf8') as target:
        for source,result,embedding in store.db.execute(
            'SELECT i.source,p.result,p.embedding FROM items i JOIN predictions p ON i.sha=p.sha ORDER BY i.id'):
            item=json.loads(source);prediction=json.loads(result)
            if embedding is None:continue
            vector=np.frombuffer(embedding,dtype=np.float32)
            for ue in profiles['ues']:
                key=(ue['ue_id'],ue['mission_id'],item['source_split'])
                history=histories.setdefault(key,[]);n=sequence.get(key,0)
                novelty=None if not history else float(np.clip((1-max(float(vector@v) for v in history))/2,0,1))
                record={'item_id':item['item_id'],'image_sha256':item['image_sha256'],
                        'ue_id':key[0],'mission_id':key[1],'source_split':key[2],
                        'sequence':n,'prior_history_size':len(history),'novelty':novelty,
                        'novelty_available':bool(history),'cold_start':not bool(history),
                        'context_relevance':prediction.get('parameters',{}).get('mission_relevance',{}).get(key[1]),
                        'stream_order':'lexicographic source shard and row ID; engineering replay, not capture chronology',
                        'history_capacity':128,'method':'query_before_commit; (1-max_same_UE_cosine)/2'}
                target.write(json.dumps(record,allow_nan=False)+'\n')
                history.append(vector.copy());del history[:-128];sequence[key]=n+1
    temporary.replace(output/'context_parameters.jsonl')
