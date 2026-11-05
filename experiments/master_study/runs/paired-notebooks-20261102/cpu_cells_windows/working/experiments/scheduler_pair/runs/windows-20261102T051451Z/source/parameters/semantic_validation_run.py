"""Pixel extraction and model comparison workers for the balanced pilot."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from .artifacts import write_json
from .contracts import canonical
from .official_panel import fetch_page
from .semantic_panel import panel_cases,make_case,score_panel
from .semantic_contract import BUILDING_FACTS,audit_candidate


def materialize_panel(spec,output):
    output=Path(output)
    # Run pre-creates an empty data directory. Accept that owned empty directory,
    # while retaining refusal to overwrite any previous dataset or partial run.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('panel output is not an empty directory: '+str(output))
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir()
    wanted=spec['panel_records'];indices=[r['source_row'] for r in wanted];metadata={}
    for offset in range(min(indices),max(indices)+1,100):
        page=fetch_page(offset,min(100,max(indices)-offset+1))
        metadata.update({r['row_idx']:r['row'] for r in page['rows']})
    records=[];bytes_used=0
    for record in wanted:
        row=metadata[record['source_row']]
        if {k:v for k,v in row.items() if k!='image'}!=record['native_labels']:
            raise ValueError('source annotations changed since panel selection')
        parsed=urlsplit(row['image']['src'])
        if parsed.hostname!='datasets-server.huggingface.co' or '/'+record['dataset_revision']+'/' not in parsed.path:
            raise ValueError('source image revision changed')
        with urlopen(Request(row['image']['src'],headers={'User-Agent':'parameters-research'}),timeout=45) as response:
            image=response.read(5_000_001)
        bytes_used+=len(image)
        if len(image)>5_000_000 or bytes_used>20_000_000:raise RuntimeError('runtime image byte cap')
        if hashlib.sha256(image).hexdigest()!=record['image_id']:raise ValueError('runtime image checksum differs')
        path=output/'images'/(record['image_id']+'.jpg');path.write_bytes(image)
        records.append({**record,'image_path':str(path.resolve())})
        print('VERIFIED IMAGE',len(records),'/',len(wanted),'bytes',bytes_used,flush=True)
    panel_cases(records) # Assert balanced states and usable references before inference.
    write_json(output/'records.json',records)
    return records


def extract(root,spec,device='cuda:0',detector=True):
    from PIL import Image
    from .pilot_producers import PilotVisionProducer,snapshot_model
    from .image_claims import NativeAerialClaimProvider
    from .model_settings import DetectorSettings
    from .annotation_pilot import representation
    from .ue_profiles import load_ue_profiles
    root=Path(root);output=root/'extraction';output.mkdir(parents=True,exist_ok=False)
    records=json.loads((root/'data/records.json').read_text());missions,_=load_ue_profiles();mission=missions['building_damage']
    path,revision=snapshot_model('openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    native=NativeAerialClaimProvider(device=device)
    provider=PilotVisionProducer(path,{'building_damage':mission},device=device,image_claim_provider=native,
        enable_detector=detector,detector_settings=DetectorSettings(confidence=.05,iou=.5,image_size=640),
        clip_preprocessing='checkpoint_native')
    evidence={};started=time.perf_counter()
    try:
        with (output/'features.jsonl').open('w',encoding='utf-8') as handle:
            for row in records:
                if time.perf_counter()-started>180:raise TimeoutError('pixel extraction pilot exceeded 180 seconds')
                # Published labels are not passed to the visual producer.
                with Image.open(row['image_path']) as picture:
                    item=provider.extract_batch([picture.convert('RGB')],[mission])[0]
                evidence[row['image_id']]=item
                handle.write(canonical({'image_id':row['image_id'],'evidence':item.to_dict()})+'\n');handle.flush()
                print('EXTRACT IMAGE',len(evidence),'/',len(records),flush=True)
        write_json(output/'producer.json',{'clip_revision':revision,'clip':provider.global_provider.provenance,
            'detector':provider.provenance,'native_image_claims':native.provenance,
            'timings':provider.timings,'native_timings':native.timings,'device':device})
        return finish_extraction(root,records,evidence,mission,output,started)
    finally:provider.close()


def finish_extraction(root,records,evidence,mission,output,started):
    from .pilot_parameters import PilotParameterExtractor,PilotMemory
    from .annotation_pilot import representation
    extractor=PilotParameterExtractor();memories={};trace=[];actual=[];availability=Counter()
    # Two independent senders, then exact repeats and a third sender's cold start.
    stream=[('UE-A' if i%2==0 else 'UE-B',r) for i,r in enumerate(records)]
    stream.extend([('UE-A',records[0]),('UE-B',records[1]),('UE-C',records[0])])
    for index,(sender,row) in enumerate(stream):
        memory=memories.setdefault(sender,PilotMemory(sender,mission.mission_id))
        item=evidence[row['image_id']];sequence=memory.version
        bundle=extractor.extract_evidence(item,mission,{'source_id':sender,'sequence':sequence,
            'image_token':sender+'-'+str(index)},memory)
        candidate=representation(bundle,mission)
        case=make_case(row,candidate,'actual-'+str(index))
        if index<len(records):actual.append(case)
        wire=canonical(candidate).encode() # Research JSON bytes, not a frozen binary header.
        trace.append({'observation_id':sender+'-'+str(index),'image_id':row['image_id'],'sender':sender,
            'kb_version_before':memory.version,'bundle':bundle.to_dict(),'candidate':candidate,
            'analysis_json_bytes':len(wire),'audit':case['audit']})
        availability.update(k for k,value in bundle.parameters.items() if value.available)
        memory.commit(item,sequence,{'observation_id':sender+'-'+str(index),'image_id':row['image_id'],
                                     'mission_version':mission.version})
    repeats=[trace[-3]['bundle']['parameters']['novelty'],trace[-2]['bundle']['parameters']['novelty']]
    if any(not n['available'] or abs(n['value']['distance'])>1e-6 for n in repeats):
        raise AssertionError('cached exact repeats must have zero appearance novelty')
    if trace[-1]['bundle']['parameters']['novelty']['value']['cold_start']!=1:
        raise AssertionError('new sender must have an independent cold-start KB')
    write_json(output/'trace.json',trace)
    write_json(root/'data/controls_evaluator_only.json',panel_cases(records))
    write_json(root/'data/actual_evaluator_only.json',actual)
    known=[row['audit'] for row in actual]
    compared=[v for audit in known for v in audit['factual']['compared'].values()]
    summary={'status':'complete','images':len(records),'observations':len(trace),'controls':4*len(records),
        'actual_review_cases':len(actual),'field_availability_counts':dict(availability),
        'comparable_native_facts':len(compared),'correct_native_facts':sum(v['correct'] for v in compared),
        'exact_repeat_novelty_checked':True,'independent_sender_cold_start_checked':True,
        'total_seconds':time.perf_counter()-started,
        'limits':{'spatial':'box geometry only; no localization truth',
                 'novelty':'controlled repeats, not natural event or asset identity',
                 'rarity':'causal predicted frequency; not importance',
                 'quality':'two native building facts; other scene semantics not verified',
                 'time_geography':'unavailable without trusted metadata'}}
    write_json(output/'status.json',summary);return summary


def select_review_cohort(root,per_stratum):
    """Budget-only balanced selection before reviewer results; preserve full cases."""
    root=Path(root);records=json.loads((root/'data/records.json').read_text())
    selected=[]
    for kind in ('absent','present_unaffected','affected'):
        pool=sorted((r for r in records if r['stratum']==kind),key=lambda r:r['source_row'])
        if len(pool)<per_stratum:raise ValueError('insufficient review cohort stratum')
        selected.extend(r['image_id'] for r in pool[:per_stratum])
    for name in ('controls','actual'):
        path=root/'data'/(name+'_evaluator_only.json')
        cases=json.loads(path.read_text())
        write_json(root/'data'/('full_'+name+'_evaluator_only.json'),cases)
        write_json(path,[c for c in cases if c['image_id'] in selected])
    result={'parent_images':len(selected),'images_per_stratum':per_stratum,'image_ids':selected,
            'controls':4*len(selected),'actual':len(selected),
            'selection':'first source rows in each native stratum; budget choice before reviewer outcomes',
            'full_extraction_images':len(records)}
    write_json(root/'data/review_cohort.json',result)
    return result


def review_case_order(cases,ordering='controls_first'):
    indices=[next(i for i,c in enumerate(cases) if c['stratum']==kind and c['arm']==arm)
             for kind,arm in (('absent','complete'),('present_unaffected','condition_omitted'),('affected','wrong_condition'))]
    if ordering=='largest_actual_first':
        actual=[i for i,c in enumerate(cases) if c['arm']=='actual_extractor']
        if not actual:raise ValueError('actual-first review requires real extracted representations')
        indices=[max(actual,key=lambda i:len(canonical(cases[i]['candidate'])))]+indices
    elif ordering!='controls_first':raise ValueError('unknown review ordering')
    return indices+[i for i in range(len(cases)) if i not in indices],len(indices)


def review(root,spec,model_key,seconds=1000):
    from .pilot_producers import snapshot_model
    from .qwen_reviewer import QwenAnnotationReviewer,load_review_processor,thinking_template_receipt,pixel_input_receipt,build_review_messages
    from .model_settings import ReviewerSettings
    from .ue_profiles import load_ue_profiles
    from PIL import Image,ImageOps
    from huggingface_hub import snapshot_download
    root=Path(root);output=root/model_key;output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    model=spec['models'][model_key];settings=ReviewerSettings(**spec['review_settings']);missions,_=load_ue_profiles()
    controls=json.loads((root/'data/controls_evaluator_only.json').read_text())
    actual=json.loads((root/'data/actual_evaluator_only.json').read_text());cases=controls+actual
    reviewer=None;reviews=[]
    try:
        import torch,transformers,sys
        cache=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))
        cache.mkdir(parents=True,exist_ok=True)
        resources={'python':sys.executable,'transformers':transformers.__version__,'torch':torch.__version__,
                   'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   'reviewer_free_gpu_bytes':torch.cuda.mem_get_info(1)[0],
                   'free_cache_disk_bytes':shutil.disk_usage(cache).free,'enable_thinking':False}
        write_json(output/'resources.json',resources)
        if transformers.__version__!=model['transformers']:
            raise RuntimeError('reviewer dependency version differs from frozen specification')
        if model_key=='qwen9b' and resources['free_cache_disk_bytes']<spec['limits']['minimum_9b_free_disk_bytes']:
            raise RuntimeError('insufficient free disk for pinned 9B source weights')
        if model_key=='qwen9b' and any(importlib.util.find_spec(k) is not None for k in ('fla','causal_conv1d')):
            raise RuntimeError('unverified optional DeltaNet kernels present; reject before weight download')
        assets=snapshot_download(model['model_id'],revision=model['revision'],
            allow_patterns=['*.json','*.txt','*.model','*.tiktoken','*.jinja'])
        processor=load_review_processor(assets,settings)
        with Image.open(cases[0]['image_path']) as image:picture=ImageOps.contain(image.convert('RGB'),(512,512))
        messages=build_review_messages([{'type':'image','image':picture},{'type':'text','text':'Return one JSON object.'}],settings.prompt_variant)
        template=thinking_template_receipt(processor,messages,model['family'])
        tensors=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=True,return_tensors='pt')
        template.update(pixel_input_receipt(tensors,processor,settings.max_pixels))
        write_json(output/'processor_preflight.json',template)
        path,revision=snapshot_model(model['model_id'],revision=model['revision'])
        if model['family']=='qwen3_5':
            from .qwen35_reviewer import Qwen35AnnotationReviewer
            reviewer=Qwen35AnnotationReviewer(path,revision,settings=settings)
        else:reviewer=QwenAnnotationReviewer(path,revision,settings=settings)
        reviewer.maximum_generation_seconds=60;reviewer.maximum_input_tokens=8192
        reviewer.require_finite_preflight=True
        reviewer.load();write_json(output/'model.json',reviewer.provenance)
        # A complete, genuinely missing, and contradicted case form a technical
        # timing/format preflight. These same calls are reused in the main cohort.
        order,technical_count=review_case_order(cases,spec.get('review_order','controls_first'))
        write_json(output/'review_order.json',{'case_ids':[cases[i]['case_id'] for i in order],
            'technical_cases':technical_count,'ordering':spec.get('review_order','controls_first')})
        state='complete';reason=None
        with (output/'reviews.jsonl').open('w',encoding='utf-8') as handle:
            for position,index in enumerate(order):
                if time.perf_counter()-started>seconds-15:
                    state='partial';reason='declared worker time budget';break
                case=cases[index]
                row=reviewer.review([case['image_path']],case['candidate'],case['reference'],missions['building_damage'],
                    case['rubric'],[],case['case_id'],output/'raw')
                reviews.append(row);handle.write(canonical(row)+'\n');handle.flush()
                write_json(output/'progress.json',{'completed':len(reviews),'planned':len(cases)})
                if (position+1)%3==0:print('REVIEW',model_key,position+1,'/',len(cases),flush=True)
                if row['status']=='model_failed':
                    state='failed';reason='model failure; no automatic retry';break
                if position==technical_count-1:
                    mean=sum(r['wall_seconds'] for r in reviews)/technical_count
                    remaining=seconds-(time.perf_counter()-started)
                    projected=mean*(len(cases)-technical_count)*1.2
                    receipt={'valid':sum(r['status']=='complete' for r in reviews),'mean_case_seconds':mean,
                        'projected_remaining_seconds_with_margin':projected,'remaining_seconds':remaining}
                    write_json(output/'technical_preflight.json',receipt)
                    if receipt['valid']<2 or projected>remaining:
                        state='partial';reason='technical preflight format/runtime gate';break
        metrics={name:score_panel(subset,[r for r in reviews if r['case_id'] in {c['case_id'] for c in subset}])
                 for name,subset in (('controls',controls),('actual',actual))}
        write_json(output/'metrics.json',metrics)
        write_json(output/'status.json',{'status':state,'reason':reason,'completed_reviews':len(reviews),
            'planned_reviews':len(cases),'seconds':time.perf_counter()-started,
            'comparison_scope':'practical model/runtime systems; family and precision both differ'})
    except Exception as error:
        write_json(output/'status.json',{'status':'failed','error':type(error).__name__,'message':str(error),
                                         'completed_reviews':len(reviews),'planned_reviews':len(cases)})
        raise
    finally:
        if reviewer is not None:reviewer.close()
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('extract','review'),required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--model',choices=('qwen4b','qwen9b'));parser.add_argument('--seconds',type=int,default=1000)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--no-detector',action='store_true')
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if args.stage=='extract':extract(args.root,spec,args.device,not args.no_detector)
    else:review(args.root,spec,args.model,args.seconds)
