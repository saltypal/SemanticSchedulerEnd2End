"""One bounded diagnostic campaign: detector, CLIP prompts and reviewer controls."""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
from .artifacts import Run,write_json
from .contracts import canonical,digest
from .annotation_pilot import resolve_image,append_json
from .annotation_review import native_reference,control_cases
from .approach_grid import search_plan
from .claim_repair_pilot import binary_metrics
from .model_settings import DetectorSettings
from .pilot_parameters import VOCABULARY,canonical_label
from .pilot_producers import file_sha256,snapshot_model,CONTRASTS
from .ue_profiles import load_ue_profiles

BALANCED_CONTRASTS={
    'building.affected_or_greater':('an aerial photograph showing disaster-affected buildings','an aerial photograph showing unaffected buildings'),
    'road.damaged':('an aerial photograph showing a physically damaged road','an aerial photograph showing an intact road'),
    'road.blocked':('an aerial photograph showing an obstructed road','an aerial photograph showing an unobstructed road'),
    'incident.informative':('an aerial photograph documenting a disaster incident','an aerial photograph unrelated to a disaster incident')}


def detector_predictions(result):
    rows=[]
    for box,score,label in zip(result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),result.boxes.cls.detach().cpu().tolist()):
        if not np.isfinite(box+[score]).all():raise ValueError('nonfinite detector output')
        rows.append({'box':list(map(float,box)),'score':float(score),'label':canonical_label(result.names[int(label)])})
    return rows


def parity_matches(cached,direct,threshold=.25):
    a=sorted((r['label'],r['score'],*r['box']) for r in cached if r['score']>=threshold)
    b=sorted((r['label'],r['score'],*r['box']) for r in direct)
    return len(a)==len(b) and all(x[0]==y[0] and np.allclose(x[1:],y[1:],rtol=1e-4,atol=1e-3) for x,y in zip(a,b))


def run_detector(records,output,budget):
    from PIL import Image
    import torch,ultralytics
    from ultralytics import YOLOWorld
    model=YOLOWorld('yolov8s-worldv2.pt').to('cuda:0');model.set_classes(list(VOCABULARY))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'model.json',{'checkpoint':str(model.ckpt_path),'sha256':file_sha256(model.ckpt_path),
        'ultralytics':ultralytics.__version__,'vocabulary':list(VOCABULARY),'device':'cuda:0',
        'image_labels_do_not_validate_box_localization':True})
    configs=search_plan()['stages']['detector']['configurations'];results=[];parity=[]
    try:
        for size,iou in ((640,.5),(640,.7),(1024,.5),(1024,.7)):
            settings=DetectorSettings(confidence=.05,image_size=size,iou=iou);cache={};began=time.perf_counter()
            for index,row in enumerate(records):
                budget()
                with Image.open(row['image_path']) as image:picture=image.convert('RGB')
                torch.cuda.synchronize(0);start=time.perf_counter()
                raw=model.predict(source=picture,device='cuda:0',**settings.predict_kwargs())[0]
                torch.cuda.synchronize(0);elapsed=time.perf_counter()-start
                rows=detector_predictions(raw);cache[row['image_id']]=rows
                append_json(output/'raw_detections.jsonl',{'image_id':row['image_id'],'settings':settings.predict_kwargs(),
                    'detections':rows,'seconds':elapsed,'max_det_saturated':len(rows)>=settings.max_detections})
                if index<2:
                    direct=detector_predictions(model.predict(source=picture,device='cuda:0',
                        **replace(settings,confidence=.25).predict_kwargs())[0])
                    check={'image_id':row['image_id'],'size':size,'iou':iou,'matches':parity_matches(rows,direct)}
                    parity.append(check);write_json(output/'parity.json',parity)
                    if not check['matches']:raise RuntimeError('low-confidence replay differs from direct inference; cached grid invalid')
                if (index+1)%5==0:print('YOLO size',size,'IoU',iou,index+1,'/',len(records),flush=True)
            for cfg in [c for c in configs if c['image_size']==size and c['iou']==iou]:
                metrics={}
                for category in ('building','road'):
                    fact=category+'.present';cohort=[r for r in records if fact in native_reference(r)['facts']]
                    truth=[native_reference(r)['facts'][fact] for r in cohort]
                    predicted=[any(d['label']==category and d['score']>=cfg['confidence'] for d in cache[r['image_id']]) for r in cohort]
                    metrics[fact]={'raw_presence_proxy':binary_metrics(truth,predicted),
                        'always_positive':binary_metrics(truth,[True]*len(truth)),
                        'always_negative':binary_metrics(truth,[False]*len(truth))}
                result={'configuration':cfg,'metrics':metrics,'forward_cache_seconds':time.perf_counter()-began,
                    'not_runtime_negative_claims':True,'scope':'image presence proxy, no box ground truth'}
                results.append(result);append_json(output/'grid_rows.jsonl',result)
        write_json(output/'status.json',{'status':'complete','configurations':len(results),'images':len(records),
            'parity_checks':len(parity),'selected_configuration':None})
    finally:
        del model
        import gc
        gc.collect();torch.cuda.empty_cache()
    return results


def run_clip(records,missions,output,budget):
    from PIL import Image
    from .vision import FrozenVisionProvider
    path,revision=snapshot_model('openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    provider=FrozenVisionProvider('clip',path,device='cuda:0',batch_size=4)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);results=[]
    try:
        for policy in ('checkpoint_native','aspect_preserving_pad'):
            provider.preprocessing=policy;provider.load()
            provider.provenance={**provider.provenance,'preprocessing':policy,'revision':revision}
            cache={}
            for start in range(0,len(records),4):
                budget();chunk=records[start:start+4];images=[]
                for row in chunk:
                    with Image.open(row['image_path']) as image:images.append(image.convert('RGB'))
                values=provider.extract_batch(images,[missions['building_damage']]*len(images))
                for row,value in zip(chunk,values):cache[row['image_id']]=value
                print('CLIP',policy,start+len(chunk),'/',len(records),flush=True)
            for cfg in [c for c in search_plan()['stages']['clip_alignment']['configurations'] if c['preprocessing']==policy]:
                contrast_set=CONTRASTS if cfg['condition_prompt_set']=='historical' else BALANCED_CONTRASTS
                contrast_names=list(contrast_set);texts=[t for pair in contrast_set.values() for t in pair]
                features=provider._texts(texts)
                mission_names=list(missions)
                task_text=[missions[name].paragraph if cfg['text_source']=='paragraph' else missions[name].conditions[0][1] for name in mission_names]
                task_features=provider._texts(task_text)
                for row in records:
                    vector=np.asarray(cache[row['image_id']].embedding);cos=vector@features.T;alignment=vector@task_features.T
                    item={'configuration':cfg,'image_id':row['image_id'],
                        'mission_cosines':dict(zip(mission_names,map(float,alignment))),
                        'condition_contrasts':{key:{'positive':float(cos[2*i]),'negative':float(cos[2*i+1]),
                            'margin':float(cos[2*i]-cos[2*i+1])} for i,key in enumerate(contrast_names)},
                        'not_probabilities_or_native_claims':True}
                    append_json(output/'scores.jsonl',item)
                result={'configuration':cfg,'images':len(records)};results.append(result)
            write_json(output/('model_'+policy+'.json'),provider.provenance)
        write_json(output/'status.json',{'status':'complete','configurations':len(results),'rows':len(results)*len(records),
            'selected_configuration':None,'scope':'alignment sensitivity, not validated native-label classifier'})
    finally:provider.close()
    return results


def run(manifest,output,exclusions_file,seconds=3000):
    import torch
    from .qwen_reviewer import QwenAnnotationReviewer
    from .reviewer_grid import run as run_reviews
    if torch.cuda.device_count()!=2:raise RuntimeError('exactly2actual GPUs required')
    run=Run(output,{'stage':'pending_model_grids','seconds':seconds,'plan':search_plan(),
        'devices':{'YOLO_CLIP':'cuda:0','Qwen':'cuda:1'},'not_scheduler_training':True})
    reviewer=None;outcomes={}
    try:
        with run.stage('00_data_and_controls'):
            source=json.loads(Path(manifest).read_text());missions,_=load_ue_profiles()
            excluded=json.loads(Path(exclusions_file).read_text())
            if any(r.get('role') in ('final','locked_final') or r.get('protected_final') for r in source):
                raise ValueError('protected final data not permitted in exploratory grid')
            cases=control_cases(source,missions,parents=8,excluded_groups=excluded['groups'])
            records=sorted((r for r in source if r['source']=='ladi'),key=lambda r:r['image_id'])[:20]
            wanted={r['image_id'] for r in records}|{c['image_id'] for c in cases}
            selected={r['image_id']:resolve_image(r,manifest) for r in source if r['image_id'] in wanted}
            records=[selected[r['image_id']] for r in records]
            for case in cases:case['image_path']=selected[case['image_id']]['image_path']
            write_json(run.path/'data/exclusions.json',excluded)
            write_json(run.path/'data/detector_evaluator_records.json',records)
            write_json(run.path/'data/exposure.json',{'image_ids':sorted(wanted),
                'groups':sorted({r['group'] for r in selected.values()}),'role':'development_diagnostic',
                'data_selection':'fixed first20LADI image hashes; independent of model outputs'})
            write_json(run.path/'hardware.json',{'gpu_names':[torch.cuda.get_device_name(i) for i in (0,1)],
                'torch':torch.__version__,'cuda':torch.version.cuda})
            print('Prepared',len(records),'detector images,',len(cases),'review controls; excluded',len(excluded['groups']),'prior groups',flush=True)
        for stage,operation in (
            ('01_detector_grid',lambda:run_detector(records,run.path/'detector',run.budget)),
            ('02_clip_grid',lambda:run_clip(records,missions,run.path/'clip',run.budget))):
            with run.stage(stage):
                try:
                    operation();outcomes[stage]={'status':'complete'}
                except Exception as error:
                    outcomes[stage]={'status':'failed','error':type(error).__name__,'message':str(error)}
                    print('STAGE FAILED, preserving other independent grids:',outcomes[stage],flush=True)
                write_json(run.path/'stage_outcomes.json',outcomes)
        with run.stage('03_reviewer_grid'):
            run.budget();path,revision=snapshot_model('Qwen/Qwen3-VL-4B-Instruct',revision='ebb281ec70b05090aa6165b016eac8ec08e71b17')
            reviewer=QwenAnnotationReviewer(path,revision)
            remaining=max(1,min(2000,seconds-(time.monotonic()-run.started)-30))
            results=run_reviews(reviewer,cases,missions,run.path/'reviewer',excluded_groups=excluded['groups'],seconds=remaining)
            outcomes['03_reviewer_grid']={'status':'complete','configurations':len(results),
                'passing_configurations':sum(r['metrics']['engineering_gate_passed'] for r in results)}
        write_json(run.path/'status.json',{'status':'complete' if all(o['status']=='complete' for o in outcomes.values()) else 'partial',
            'stages':outcomes,'selected_configuration':None,'seconds':time.monotonic()-run.started,
            'no_scheduler_training':True,'no_new_dataset_download':True})
    except Exception as error:
        write_json(run.path/'status.json',{'status':'failed','error':type(error).__name__,'message':str(error),'stages':outcomes})
        raise
    finally:
        if reviewer is not None:reviewer.close()
    return run.path
