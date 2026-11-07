"""Ready-model extraction and native-label parameter diagnostics, no policies."""
import copy,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from parameters.artifacts import write_json
from parameters.ue_profiles import load_ue_profiles
from parameters.extractor import SenderMemory
from parameters.minimal_parameters import FinalParameterExtractor,ExtendedParameterHeader
from parameters.minimal_vision import MinimalVisionProvider
from parameters.scene_measurements import YOLOWorldSceneProvider,SceneMeasurements
from parameters.pilot_parameters import context_similarity_key
from experiments.spi_classifier_repair.medic_deployment import SPIImagePredictor
from experiments.spi_classifier_repair.resume_store import ResumeStore,file_hash,implementation_identity

def cache_ready(rows,winner,identity,receipt,clip_path,yolo_path,profiles_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if receipt.get('runtime_promotion_allowed') is not True:
        write_json(out/'status.json',{'status':'not_started_readiness_failed','checks':receipt.get('checks'),
            'final_test_gate_passed':receipt.get('final_test_gate_passed'),'records':0})
        return None
    missions,bindings=load_ue_profiles(profiles_path);store=ResumeStore(out/'image_checkpoints')
    model=SPIImagePredictor(winner,identity,'cuda:0',receipt['temperature']);model.authorize(receipt)
    visual=MinimalVisionProvider(clip_path,list(missions.values()),device='cuda:0')
    detector=YOLOWorldSceneProvider(yolo_path,device='cuda:0');memories={};records=[];timings=[];headers=bytearray()
    class CachedVisual:
        evidence=None
        def extract_batch(self,images,tasks):
            values=[]
            for task in tasks:
                evidence=copy.deepcopy(self.evidence)
                evidence.similarities={context_similarity_key(task):float(evidence.embedding@visual.context_embedding(task))}
                values.append(evidence)
            return values
    class CachedScene:
        scene=None
        def measure(self,image):return self.scene
    class CachedSPI:
        prediction=None
        def predict_image(self,image):return self.prediction
    cv,cs,cp=CachedVisual(),CachedScene(),CachedSPI()
    extractor=FinalParameterExtractor(cv,scene_provider=cs,spi_image_provider=cp)
    def clock():torch.cuda.synchronize(0);return time.perf_counter()
    try:
        for sequence,row in enumerate(rows):
            spec={'image_sha256':row['image_sha256'],'model':receipt['model_sha256'],'temperature':receipt['temperature'],
                'identity':identity,'implementation':implementation_identity(cache_ready,MinimalVisionProvider,YOLOWorldSceneProvider)}
            cached=store.read('image_parameters',spec)
            if cached is None:
                begin=clock()
                with Image.open(row['image_path']) as image:
                    loaded=clock();evidence=visual.extract_batch([image],[next(iter(missions.values()))])[0];clip_end=clock()
                    prediction=model.predict_image(image);spi_end=clock();scene=detector.measure(image);scene_end=clock()
                if scene.status!='ok':raise RuntimeError('final scene extraction unavailable: '+str(scene.reason))
                directory=store.directory('image_parameters',spec)
                write_json(directory/'value.json',{'spi':prediction,'scene':asdict(scene),'evidence':evidence.to_dict(),
                    'timing':{'load_seconds':loaded-begin,'clip_seconds':clip_end-loaded,'spi_seconds':spi_end-clip_end,
                              'scene_seconds':scene_end-spi_end,'image_extraction_seconds':scene_end-begin}})
                value=json.loads((directory/'value.json').read_text());cached=store.complete('image_parameters',spec,[directory/'value.json'],value)
            value=cached['value']
            from parameters.extractor import VisualEvidence
            cv.evidence=VisualEvidence.from_dict(value['evidence']);cs.scene=SceneMeasurements(**value['scene']);cp.prediction=value['spi']
            timings.append({'image_id':row['image_id'],**value['timing'],'measurement_scope':'fresh image computation at original cache creation'})
            for source,binding in enumerate(bindings,1):
                mission=missions[binding.mission_id];key=(binding.ue_id,mission.version,row['split'])
                memory=memories.setdefault(key,SenderMemory(binding.ue_id,mission.mission_id,capacity=128))
                before=memory.version
                meta={'source_id':binding.ue_id,'source_numeric_id':source,'session_id':{'train':1,'validation':2,'test':3}[row['split']],
                    'message_id':sequence+1,'profile_id':source,'sequence':sequence,'capture_time_ms':row.get('capture_time_ms'),
                    'timestamp_trusted':row.get('timestamp_trusted') is True}
                begin=time.perf_counter();bundle,payload=extractor.transmit_batch([None],[mission],[meta],[memory])[0]
                elapsed=time.perf_counter()-begin;decoded=ExtendedParameterHeader.from_bytes(payload)
                records.append({'image_id':row['image_id'],'ue_id':binding.ue_id,'mission':mission.mission_id,'split':row['split'],
                    'parameters':bundle['compact'],'header_hex':payload.hex(),'header_bytes':len(payload),'kb_version_before':before,
                    'kb_version_after':memory.version,'context_kb_serialization_seconds':elapsed,
                    'quantization_error':{'relevance':None if bundle['relevance']['cosine'] is None else abs(decoded.core.relevance-bundle['relevance']['cosine']),
                        'novelty':None if bundle['novelty']['value'] is None else abs(decoded.core.novelty-bundle['novelty']['value'])}})
                headers.extend(payload)
            write_json(out/'progress.json',{'images':sequence+1,'headers':len(records),'knowledge_bases':len(memories)})
            print('FINAL PARAMETERS',sequence+1,'/',len(rows),flush=True)
        write_json(out/'runtime_records.json',records);write_json(out/'image_costs.json',timings)
        (out/'headers.bin').write_bytes(headers)
        write_json(out/'status.json',{'status':'complete','images':len(rows),'headers':len(records),'header_bytes_each':35,
            'total_header_bytes':len(headers),'payload_bytes':None,'payload_reason':'codec outside parameter study',
            'history':'separate causal UE/mission/split histories, FIFO128; no future image access'})
        return records
    finally:
        visual.close();del model;torch.cuda.empty_cache()

def parameter_diagnostics(runtime_records,source_records,output):
    """Only exact supported native tasks; no general urgency or novelty truth."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    lookup={r['image_id']:r for r in source_records};results={}
    for mission in ('incident_informativeness','building_damage','road_damage','road_blockage','household_inventory'):
        y=[];scores=[]
        for r in runtime_records or []:
            if r['mission']!=mission:continue
            source=lookup[r['image_id']];native=source.get('native_labels',{});value=native.get('informative')
            schema={f.get('name'):f.get('type',{}) for f in source.get('native_features',[]) if isinstance(f,dict)}
            if isinstance(value,int) and isinstance(schema.get('informative'),dict):
                names=schema['informative'].get('names',[])
                value=names[value] if 0<=value<len(names) else None
            if mission!='incident_informativeness' or value not in ('informative','not_informative'):continue
            score=r['parameters']['relevance']
            if score is None:continue
            y.append(int(value=='informative'));scores.append(score)
        if len(set(y))==2:results[mission]={'n':len(y),'auroc':float(roc_auc_score(y,scores)),'average_precision':float(average_precision_score(y,scores)),
            'scope':'native informativeness proxy, not exhaustive paragraph relevance or urgency truth'}
        else:results[mission]={'status':'unsupported_native_labels_or_insufficient_classes','n':len(y)}
    write_json(output,{'relevance':results,'novelty':'exact-repeat/causal tests only; natural new-fact labels unavailable',
        'scene':'rectangular geometry tests and detector sensitivity; no box ground truth here',
        'scheduling':'not evaluated by this parameter study'})

def feature_diagnostics(rows,clip_features,clip_path,profiles_path,schema_path,output):
    """Development feature study allowed before release; produces no headers/cache."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    missions,bindings=load_ue_profiles(profiles_path);provider=MinimalVisionProvider(clip_path,list(missions.values()),'cuda:0')
    schema=json.loads(Path(schema_path).read_text())['features'];mapping={r['name']:r['type'] for r in schema}
    with np.load(clip_features,allow_pickle=False) as data:features=data['embeddings'];ids=list(map(str,data['image_ids']))
    if ids!=[r['image_id'] for r in rows]:raise ValueError('feature study identity mismatch')
    development=[i for i,r in enumerate(rows) if r['split']!='test'];results={}
    try:
        for name,mission in missions.items():
            context=provider.context_embedding(mission);values=features[development]@context
            result={'n':len(values),'min':float(values.min()),'median':float(np.median(values)),'max':float(values.max())}
            targets=[];scores=[]
            if name=='incident_informativeness':
                for index,value in zip(development,values):
                    native=rows[index].get('native_labels',{}).get('informative');names=mapping['informative']['names']
                    if type(native) is int and 0<=native<len(names):native=names[native]
                    if native in ('informative','not_informative'):targets.append(int(native=='informative'));scores.append(float(value))
                if len(set(targets))==2:result.update(auroc=float(roc_auc_score(targets,scores)),average_precision=float(average_precision_score(targets,scores)),
                    endpoint='native informativeness proxy; not urgency or exact mission-completion truth')
            else:result['native_metric']='unavailable: MEDIC does not establish this mission claim'
            results[name]=result
        novelty={}
        for capacity in (32,128,512,None):
            histories=[[] for _ in bindings];queries=[];repeats=[];begin=time.perf_counter()
            for step,index in enumerate(development):
                ue=step%len(bindings);vector=features[index];history=histories[ue]
                first=1. if not history else (1-float(np.clip(max(vector@old for old in history),-1,1)))/2
                queries.append(first);history.append(vector)
                if capacity:del history[:-capacity]
                # Controlled exact repeat; known identity, not semantic equivalence.
                repeat=(1-float(np.clip(max(vector@old for old in history),-1,1)))/2;repeats.append(repeat)
            novelty[str(capacity or 'all')]={'queries':len(queries),'mean_appearance_distance':float(np.mean(queries)),
                'maximum_exact_repeat_distance':float(max(repeats)),'query_and_update_seconds':time.perf_counter()-begin,
                'embedding_storage_bytes':sum(len(h) for h in histories)*features.shape[1]*4,
                'same_sender_history_only':True,'semantic_new_fact_accuracy':None}
        write_json(output,{'role':'development diagnostics only; test rows excluded; not final parameter cache',
            'relevance':results,'novelty':novelty,'image_feature_policy':'frozen CLIP fullframe pad224 training features',
            'runtime_preprocessing_note':'MinimalVision runtime adds longest-side512 preparation; exact cache is evaluated separately',
            'generalization':'MEDIC-only; other missions/domains unsupported until tested'})
    finally:provider.close()
