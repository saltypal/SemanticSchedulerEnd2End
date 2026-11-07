"""Bounded parameter-only sequence tests and native presence diagnostics."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
from sklearn.metrics import roc_auc_score
from .artifacts import Run,write_json
from .contracts import MissionSpec
from .extractor import SenderMemory,VisualEvidence,embedding_signature
from .minimal_parameters import FinalParameterExtractor,ExtendedParameterHeader
from .minimal_vision import MinimalVisionProvider
from .pilot_parameters import VOCABULARY,canonical_label,context_similarity_key
from .scene_measurements import SceneMeasurements
from .spi import SPIClassifier


def binary_metrics(truth,pred):
    truth=np.asarray(truth,bool);pred=np.asarray(pred,bool)
    tp=int(np.sum(truth&pred));fp=int(np.sum(~truth&pred));fn=int(np.sum(truth&~pred));tn=int(np.sum(~truth&~pred))
    return {'n':len(truth),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}


def execute(input_root,scene_path,model_path,clip_checkpoint,ladi_csv,output,split_manifest=None):
    input_root=Path(input_root);run=Run(output,{'seconds':600,'seed':17,'scope':'parameter extraction, not scheduling',
        'source':'existing exposed development images','native_label_role':'evaluation only'})
    provider=None
    try:
        def sha_file(path):
            digest=hashlib.sha256()
            with Path(path).open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
            return digest.hexdigest()
        paths={'input_manifest':input_root/'input_manifest.json','feature_cache':input_root/'features.npz',
               'panel':input_root/'panel.json','model':model_path,'ladi_csv':ladi_csv,
               'ue_profiles':Path(__file__).parent/'config/ue_parameter_profiles_v2.json'}
        if scene_path:paths['scene_records']=scene_path
        if split_manifest:paths['split_manifest']=split_manifest
        write_json(run.path/'data/input_provenance.json',{name:{'path':str(Path(path).resolve()),'sha256':sha_file(path)}
                                                          for name,path in paths.items()})
        rows=json.loads((input_root/'panel.json').read_text(encoding='utf8'));by_id={r['image_id']:r for r in rows}
        with np.load(input_root/'features.npz',allow_pickle=False) as f:
            ids=list(map(str,f['image_ids'].tolist()));x=np.asarray(f['embeddings'],float);signature=str(f['feature_signature'].item())
        if ids!=[r['image_id'] for r in rows] or not np.isfinite(x).all():raise ValueError('feature identity mismatch')
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        if np.any(norms<=1e-10):raise ValueError('zero cached image embedding')
        x=x/norms;index={identity:i for i,identity in enumerate(ids)}
        profiles=json.loads((Path(__file__).parent/'config/ue_parameter_profiles_v2.json').read_text())
        if profiles['feature_signature']!=signature:raise ValueError('profile feature signature mismatch')
        scene_map={}
        if scene_path:
            for line in Path(scene_path).read_text(encoding='utf8').splitlines():
                row=json.loads(line)
                if row['image_id'] not in by_id or row['image_sha256']!=by_id[row['image_id']]['image_sha256']:
                    raise ValueError('scene/image provenance mismatch')
                if row['image_id'] in scene_map:raise ValueError('duplicate scene record')
                allowed={canonical_label(name) for name in VOCABULARY}
                actual=set(row['scene'].get('audit',{}).get('category_counts',{}))
                if not actual<=allowed:raise ValueError('scene output contains unknown categories')
                scene_map[row['image_id']]=SceneMeasurements(**row['scene'])
        roles={r['image_id']:r['split'] for r in rows}
        if split_manifest:
            assignments=json.loads(Path(split_manifest).read_text(encoding='utf8'))['assignments']
            if {r['image_id'] for r in assignments}!=set(ids):raise ValueError('split manifest mismatch')
            roles={r['image_id']:r['split'] for r in assignments}
        model=SPIClassifier.load(model_path);rng=np.random.default_rng(17)
        usable=[identity for identity in ids if roles[identity]!='test']
        if len(usable)<8:raise ValueError('insufficient development images')
        shared=usable[0];pool=[k for k in usable if k!=shared]
        missions=[]
        for p in profiles['profiles']:
            missions.append(MissionSpec(p['mission_id'],p['paragraph'],((p['mission_id'],p['paragraph']),),
                                        allowed_delay_ms=p['allowed_delay_ms']))
        provider=MinimalVisionProvider(clip_checkpoint,{m.mission_id:m for m in missions},device='cpu')
        extractor=FinalParameterExtractor(provider,model)
        sequences=[];variants=[];variant_missions=[];events=[]
        for profile,mission in zip(profiles['profiles'],missions):
            chosen=[shared]+[pool[int(i)] for i in rng.choice(len(pool),7,replace=False)]
            sequence=[(identity,'original') for identity in chosen]+[(chosen[0],'exact_repeat'),(chosen[1],'exact_repeat'),
                      (chosen[2],'brightness_variant'),(chosen[3],'crop_variant')]
            sequences.append((profile,mission,sequence))
        with zipfile.ZipFile(input_root/'images.zip') as archive:
            def picture(identity):
                data=archive.read(by_id[identity]['image_path'])
                if hashlib.sha256(data).hexdigest()!=by_id[identity]['image_sha256']:raise ValueError('image checksum mismatch')
                with Image.open(io.BytesIO(data)) as image:return image.convert('RGB')
            probe=provider.extract_batch([picture(shared)],[missions[0]])[0]
            if embedding_signature(probe)!=signature or np.max(np.abs(np.asarray(probe.embedding)-x[index[shared]]))>1e-4:
                raise ValueError('CLIP cache/provenance mismatch')
            provenance=dict(probe.provenance)
            for profile,mission,sequence in sequences:
                for identity,kind in sequence:
                    if kind not in ('brightness_variant','crop_variant'):continue
                    image=picture(identity)
                    if kind=='brightness_variant':image=ImageEnhance.Brightness(image).enhance(.8)
                    else:
                        width,height=image.size;image=image.crop((width//20,height//20,width-width//20,height-height//20))
                    variants.append(image);variant_missions.append(mission)
            with run.stage('appearance_variants'):
                variant_evidence=provider.extract_batch(variants,variant_missions)
            if scene_map:
                # Diagnostic overlays show detector predictions, never native truth.
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle
                figures=[]
                for source in ('ladi','medic'):
                    candidates=[r for r in rows if r['source']==source and roles[r['image_id']]!='test' and
                        r['image_id'] in scene_map and scene_map[r['image_id']].status=='ok' and
                        scene_map[r['image_id']].instances]
                    if not candidates:continue
                    row=candidates[0];scene=scene_map[row['image_id']];image=picture(row['image_id'])
                    fig,ax=plt.subplots(figsize=(10,7));ax.imshow(image);ax.axis('off')
                    detections=scene.audit.get('detections',[])
                    for detection in detections[:12]:
                        x1,y1,x2,y2=detection['box']
                        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,linewidth=1.8,edgecolor='yellow'))
                        ax.text(x1,max(0,y1-3),detection['label'],fontsize=8,color='black',
                                bbox={'facecolor':'yellow','alpha':.8,'pad':1})
                    ax.set_title(f'{source}: YOLO predictions, first {min(12,len(detections))} of {len(detections)}; not ground truth')
                    path=run.path/'figures'/f'yolo_example_{source}.png'
                    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
                    figures.append({'source':source,'image_id':row['image_id'],'path':str(path),
                                    'displayed':min(12,len(detections)),'total':len(detections)})
                write_json(run.path/'figures/manifest.json',figures)
        # Produce parameters for the complete cached cohort without exposing
        # annotations to extraction. Keep split/UE histories disjoint so sealed
        # or validation observations cannot enter training histories.
        with run.stage('cached_cohort_extraction'):
            memories={};sequence_numbers={};counts={'train':0,'validation':0,'test':0}
            sessions={'train':1,'validation':2,'test':3}
            with (run.path/'data/cached_parameters.jsonl').open('w',encoding='utf-8') as handle:
                for i,identity in enumerate(ids):
                    role=roles[identity];profile=profiles['profiles'][i%len(profiles['profiles'])]
                    mission=missions[i%len(missions)];ue=profile['ue_id'];key=(role,ue)
                    if key not in memories:
                        memories[key]=SenderMemory(ue,mission.mission_id,capacity=32);sequence_numbers[key]=0
                    sequence_numbers[key]+=1;seq=sequence_numbers[key]
                    text=np.asarray(profile['clip_text_embedding'],float);text/=np.linalg.norm(text)
                    evidence=VisualEvidence(embedding=x[index[identity]],
                        similarities={context_similarity_key(mission):float(x[index[identity]]@text)},provenance=provenance)
                    scene=scene_map.get(identity,SceneMeasurements.unavailable('detector output not retrieved'))
                    meta={'source_id':ue,'source_numeric_id':i%len(missions)+1,'session_id':sessions[role],
                          'message_id':seq,'profile_id':1,'sequence':seq,'timestamp_trusted':False,'now_ms':0}
                    bundle,wire=extractor.serialize_and_commit(evidence,mission,meta,memories[key],scene)
                    output_record={'image_id':identity,'ue_id':ue,'split':role,'parameters':bundle['compact'],
                            'header_hex':wire.hex(),'header_bytes':len(wire),'scene_status':scene.status,
                            'scene_reason':scene.reason,'spi_supported_levels':list(model.levels),
                            'audit':{'spi':bundle['spi'],'relevance':bundle['relevance'],
                                     'novelty':bundle['novelty'],'capture':bundle['capture'],
                                     'scene_categories':scene.audit.get('category_counts',{})}}
                    handle.write(json.dumps(output_record,allow_nan=False)+'\n');handle.flush();counts[role]+=1
                    if (i+1)%50==0:print('Cached cohort parameters',i+1,'/',len(ids),flush=True)
            write_json(run.path/'metrics/cached_cohort.json',{'records':len(ids),'split_counts':counts,
                'kb_isolation':'independent sender memory per UE and split','scene_records':len(scene_map),
                'runtime_fields_exclude':'native labels, split, image ID and dataset source',
                'model_supported_levels':list(model.levels)})
        iterator=iter(variant_evidence);relevance_across_missions=[];repeat_results=[]
        with run.stage('ue_sequences'):
            for n,(profile,mission,sequence) in enumerate(sequences):
                memory=SenderMemory(profile['ue_id'],mission.mission_id,capacity=32)
                text=np.asarray(profile['clip_text_embedding'],float);text/=np.linalg.norm(text)
                relevance_across_missions.append({'ue_id':profile['ue_id'],'mission_id':mission.mission_id,
                    'image_id':shared,'cosine':float(x[index[shared]]@text)})
                for step,(identity,kind) in enumerate(sequence):
                    if kind in ('brightness_variant','crop_variant'):
                        evidence=next(iterator)
                        scene=SceneMeasurements.unavailable('appearance variant has no matching detector output')
                    else:
                        evidence=VisualEvidence(embedding=x[index[identity]],
                            similarities={context_similarity_key(mission):float(x[index[identity]]@text)},
                            provenance=provenance)
                        scene=scene_map.get(identity,SceneMeasurements.unavailable('detector output not retrieved'))
                    metadata={'source_id':profile['ue_id'],'source_numeric_id':n+1,'session_id':1,'message_id':step+1,
                              'profile_id':1,'sequence':step+1,'timestamp_trusted':False,'now_ms':0}
                    bundle,wire=extractor.serialize_and_commit(evidence,mission,metadata,memory,scene)
                    decoded=ExtendedParameterHeader.from_bytes(wire)
                    row={'ue_id':profile['ue_id'],'mission_id':mission.mission_id,'step':step+1,'image_id':identity,
                         'kind':kind,'parameters':bundle['compact'],'novelty_cold_start':bundle['novelty']['cold_start'],
                         'scene_status':scene.status,'scene_reason':scene.reason,'bytes':len(wire),
                         'scene_category_counts':scene.audit.get('category_counts',{}),
                         'quantized':{'spi':decoded.core.scheduling_class,'relevance':decoded.core.relevance,
                                      'novelty':decoded.core.novelty,'scene':decoded.scene.values()}}
                    events.append(row)
                    if kind=='exact_repeat':repeat_results.append({'ue_id':profile['ue_id'],'step':step+1,
                        'novelty':row['parameters']['novelty'],'passed':row['parameters']['novelty'] is not None and abs(row['parameters']['novelty'])<1e-6})
                print('Completed UE',profile['ue_id'],'12 causal observations',flush=True)
        write_json(run.path/'data/sequence_records.json',events)
        write_json(run.path/'metrics/sequence_controls.json',{'ues':len(sequences),'observations':len(events),
            'repeat_controls':repeat_results,'repeat_passed':sum(r['passed'] for r in repeat_results),
            'repeat_total':len(repeat_results),'shared_image_context_scores':relevance_across_missions,
            'capture_times_available':sum(r['parameters']['capture_time_ms'] is not None for r in events),
            'model_supported_levels':list(model.levels),'scene_available':sum(r['scene_status']=='ok' for r in events),
            'variant_scope':'brightness/crop are controlled appearance changes, not verified new facts'})
        collisions=defaultdict(list)
        for row in events:
            q=row['quantized'];s=q['scene'];key=(q['spi'],q['relevance'],q['novelty'],row['novelty_cold_start'],s['instances'],s['categories'],
                                                 s['largest_box_fraction'],s['union_box_coverage'])
            collisions[key].append(row)
        differing=[];collision_example=None
        for group in collisions.values():
            if len({r['image_id'] for r in group})<2:continue
            if collision_example is None:
                a=group[0];b=next(r for r in group if r['image_id']!=a['image_id'])
                collision_example={'a':a['image_id'],'b':b['image_id'],'ue_a':a['ue_id'],'ue_b':b['ue_id'],
                                   'quantized_parameters':a['quantized']}
            for a in group:
                for b in group:
                    if a['image_id']!=b['image_id'] and a['scene_category_counts']!=b['scene_category_counts']:
                        differing.append({'a':a['image_id'],'b':b['image_id'],'ue_a':a['ue_id'],'ue_b':b['ue_id'],
                            'predicted_categories_a':a['scene_category_counts'],'predicted_categories_b':b['scene_category_counts']})
                        break
                if differing:break
        write_json(run.path/'metrics/representation_collisions.json',{'different_image_parameter_collisions':
            sum(len({r['image_id'] for r in group})>1 for group in collisions.values()),
            'example':collision_example,
            'example_different_predicted_category_collision':differing[:1],
            'scope':'observed finite cohort, category differences are detector predictions'})
        same_spi=None
        for a in events:
            for b in events:
                if (a['image_id']!=b['image_id'] and a['parameters']['spi'] is not None and
                    a['parameters']['spi']==b['parameters']['spi'] and
                    abs(a['parameters']['relevance']-b['parameters']['relevance'])>=.08):
                    same_spi={'a':{'image_id':a['image_id'],'ue':a['ue_id'],'parameters':a['parameters']},
                              'b':{'image_id':b['image_id'],'ue':b['ue_id'],'parameters':b['parameters']},
                              'interpretation':'same predicted SPI, different relevance; no priority truth implied'}
                    break
            if same_spi:break
        write_json(run.path/'metrics/parameter_contrasts.json',{'same_image_across_missions':relevance_across_missions,
            'same_image_cold_then_repeat':repeat_results,'same_spi_different_relevance':same_spi,
            'scope':'parameter distinguishability, not ranking accuracy'})
        native={}
        allowed_native={r['image_id'] for r in rows if roles[r['image_id']]!='test'}
        with Path(ladi_csv).open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                if row['source_sha256'] in allowed_native:
                    native[row['source_sha256']]=json.loads(row['native_labels_json'])
        spatial={};context={}
        for field,category in [('buildings_any','building'),('roads_any','road')]:
            eligible=[r for r in rows if r['image_id'] in native and roles[r['image_id']]!='test'
                      and type(native[r['image_id']].get(field)) is bool and r['image_id'] in scene_map
                      and scene_map[r['image_id']].status=='ok']
            truth=[native[r['image_id']][field] for r in eligible]
            preds=[scene_map[r['image_id']].audit.get('category_counts',{}).get(category,0)>0 for r in eligible]
            spatial[field]=binary_metrics(truth,preds)
        mission=profiles['profiles'][0];text=np.asarray(mission['clip_text_embedding'],float);text/=np.linalg.norm(text)
        eligible=[r for r in rows if r['image_id'] in native and roles[r['image_id']]!='test'
                  and type(native[r['image_id']].get('buildings_affected_or_greater')) is bool]
        truth=[int(native[r['image_id']]['buildings_affected_or_greater']) for r in eligible]
        scores=[float(x[index[r['image_id']]]@text) for r in eligible]
        context={'n':len(truth),'positives':sum(truth),'roc_auc':float(roc_auc_score(truth,scores)) if len(set(truth))==2 else None,
            'scope':'building-affected source flag versus broad building/road mission cosine; partial proxy, not calibrated relevance'}
        write_json(run.path/'metrics/native_descriptive.json',{'scene_presence':spatial,'context_proxy':context,
            'scope':'previously exposed LADI image-level annotations; no box/count or SPI urgency truth; excludes sealed groups'})
        telemetry={'backend':'cpu','clip_model_load_seconds':provider.provider.provenance.get('model_load_seconds'),
            'clip_forward_batches':provider.provider.timings,
            'scope':'local CPU process; batch forward timing excludes checkpoint load; no GPU timing inferred'}
        try:
            import psutil
            info=psutil.Process().memory_info()
            telemetry['rss_bytes_at_end']=info.rss
            telemetry['os_peak_working_set_bytes']=getattr(info,'peak_wset',None)
            telemetry['memory_scope']='whole process including Python, checkpoint and caches'
        except ImportError:telemetry['memory_status']='psutil unavailable'
        write_json(run.path/'metrics/runtime.json',telemetry)
        write_json(run.path/'status.json',{'status':'complete','events':len(events),'model_levels':list(model.levels),
            'detector_records':len(scene_map),'claim':'development diagnostics, not validated urgency or scheduler performance'})
        return run.path
    except Exception as error:
        write_json(run.path/'status.json',{'status':'failed','reason':type(error).__name__+': '+str(error)})
        raise
    finally:
        if provider is not None:provider.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--scene')
    p.add_argument('--model',required=True);p.add_argument('--clip-checkpoint',required=True)
    p.add_argument('--ladi-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--split-manifest')
    a=p.parse_args();print(execute(a.input_root,a.scene,a.model,a.clip_checkpoint,a.ladi_csv,a.output,a.split_manifest))
