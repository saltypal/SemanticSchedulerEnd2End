"""End-to-end bounded annotation-assisted parameter pilot; no scheduler training."""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
from .artifacts import Run,write_json
from .contracts import MethodSpecification,digest,canonical
from .extractor import VisualEvidence
from .ue_profiles import load_ue_profiles
from .pilot_parameters import PilotMemory,PilotParameterExtractor,detections,matches_target
from .annotation_review import (native_reference,make_rubric,control_cases,score_controls,
                                direct_fact_metrics)


def append_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as handle:handle.write(canonical(value)+'\n');handle.flush()


def resolve_image(row,manifest):
    """Verify selected inputs only; do not hash the entire14GB upstream again."""
    from .pilot_producers import file_sha256
    root=Path(manifest).resolve().parent.parent;path=Path(row['image_path'])
    if not path.is_absolute():path=root/path
    path=path.resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('unsafe/missing selected image')
    if file_sha256(path)!=row['image_id']:raise ValueError('selected image checksum mismatch')
    return {**row,'image_path':str(path)}


def representation(bundle,mission,view='structured'):
    if view not in ('compact','structured'):raise ValueError('unknown representation view')
    groups={k:{'value':v.value,'available':v.available,'method':v.method,'reason':v.reason}
            for k,v in bundle.parameters.items() if view=='structured' or k!='scene'}
    claims={}
    spatial=bundle.parameters['spatial']
    if spatial.available and spatial.value['target_count']>0 and mission.target_entities==('building',):
        claims['building.present']=True
    if spatial.available and spatial.value['target_count']>0 and mission.target_entities==('road',):
        claims['road.present']=True
    if view=='structured' and bundle.parameters['scene'].available:
        scene=bundle.parameters['scene'].value;presence=scene.get('entity_presence',{})
        for category in ('building','road'):
            if category in presence:claims[category+'.present']=bool(presence[category])
        claims.update(scene.get('condition_predictions',{}))
        claims.update({key:row['value'] for key,row in scene.get('image_claim_predictions',{}).items()})
        targets=('chair','couch','dining table','television','refrigerator')
        if any(presence.get(t) is True for t in targets):claims['household.target_present']=True
        elif all(presence.get(t) is False for t in targets):claims['household.target_present']=False
    from .annotation_review import REQUIRED
    availability={name:{'available':name in claims,
        'reason':None if name in claims else 'not represented by enabled producer evidence in this view'}
        for name in REQUIRED.get(mission.mission_id,())}
    return {'representation':view,'is_ablation':view=='compact',
            'output_role':'diagnostic_information_removal' if view=='compact' else 'full_extractor_output',
            'parameters':groups,'claims':claims,'claim_availability':availability,'temporal':bundle.temporal,
            'definitions':{'relevance':'(CLIP image-task cosine+1)/2;not probability or priority',
              'novelty':'appearance distance and predicted concept change under supplied sender history',
              'rarity':'normalized predicted-concept Bernoulli surprisal over causal sender history',
              'spatial':'normalized target bounding-box extent/count;not physical distance',
              'complexity':'predicted task-count/box-coverage/diversity proxies',
              'claims':'positive detector observations or image-classifier decisions; raw CLIP contrasts stay descriptive evidence'}}


def actual_observations(records,bindings,steps=6):
    """Small declared synthetic series; repeats are controlled,not natural event IDs."""
    ladi=sorted((r for r in records if r['source']=='ladi'),key=lambda r:r['image_id'])
    medic=sorted((r for r in records if r['source']=='medic'),key=lambda r:r['image_id'])
    if len(ladi)<12:raise ValueError('need12 LADI observations for the bounded pilot')
    output=[]
    for position,binding in enumerate(bindings):
        pool=medic if binding.mission_id=='incident_informativeness' and len(medic)>=6 else ladi
        # Same images for the two buildingUEs;their history orders differ.
        start=0 if position<2 else position*2
        chosen=[pool[(start+i)%len(pool)] for i in range(steps)]
        if position==0 and steps>=2:chosen[1]=chosen[0]
        if position==1 and steps>=3:chosen[2]=chosen[0]
        for step,row in enumerate(chosen):
            output.append({'binding':binding,'step':step,'record':row,
                'observation_id':binding.ue_id+'-'+str(step),
                'sequence_kind':'constructed observation order;no real capture-time claim'})
    return output


def run_annotation_pilot(manifest,output,seconds=3000,control_parents=24,steps_per_ue=6,control_exclusions=None):
    import torch
    from PIL import Image
    from .qwen_reviewer import QwenAnnotationReviewer
    from .pilot_producers import PilotVisionProducer,snapshot_model
    if torch.cuda.device_count()!=2:raise RuntimeError('exactly two assigned GPUs required before model work')
    if not 0<seconds<=3600 or control_parents!=24 or steps_per_ue>6:raise ValueError('pilot limits exceed frozen initial configuration')
    run=Run(output,{'stage':'annotation_assisted_parameter_pilot','seconds':seconds,
        'control_parents':control_parents,'steps_per_ue':steps_per_ue,
        'devices':{'producer':'cuda:0','reviewer':'cuda:1'},'not_scheduler_training':True})
    reviewer=None;producer=None
    try:
        with run.stage('00_data_and_reference_preflight'):
            records=json.loads(Path(manifest).read_text());missions,bindings=load_ue_profiles()
            if any(r.get('role') in ('final','locked_final') or r.get('protected_final') for r in records):
                raise ValueError('protected final role cannot be used for exploratory pilot')
            excluded=json.loads(Path(control_exclusions).read_text()) if control_exclusions is not None else {'groups':[]}
            cases=control_cases(records,missions,parents=control_parents,excluded_groups=excluded['groups'])
            write_json(run.path/'data/control_exclusions.json',excluded)
            if {c['reference']['group'] for c in cases}&set(excluded['groups']):
                raise ValueError('new reviewer controls overlap excluded prior groups')
            observations=actual_observations(records,bindings,steps_per_ue)
            wanted={c['image_id'] for c in cases}|{o['record']['image_id'] for o in observations}
            selected={r['image_id']:resolve_image(r,manifest) for r in records if r['image_id'] in wanted}
            for case in cases:case['image_path']=selected[case['image_id']]['image_path']
            for observation in observations:observation['record']=selected[observation['record']['image_id']]
            write_json(run.path/'data/development_exposure.json',{'image_ids':sorted(wanted),
                'groups':sorted({r['group'] for r in selected.values()}),'roles':'development_only',
                'controls_are_annotation_assisted':True})
            write_json(run.path/'data/control_cases_evaluator_only.json',cases)
            write_json(run.path/'hardware.json',{'gpu_names':[torch.cuda.get_device_name(i) for i in (0,1)],
                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_count':2,'data_parallel':False})
            print('Selected',len(selected),'parents;',len(cases),'control cases;',len(observations),'UE observations',flush=True)
        with run.stage('01_reviewer_load_and_vision_preflight'):
            model_path,revision=snapshot_model('Qwen/Qwen3-VL-4B-Instruct')
            reviewer=QwenAnnotationReviewer(model_path,revision);reviewer.load()
            write_json(run.path/'models/reviewer.json',reviewer.provenance)
            # Two source files and nonempty pixel tensors verify technical input delivery;
            # source hashes alone do not prove distinct transformed tensors or grounding.
            preflight=[]
            for case in (cases[0],cases[3]):
                run.budget();result=reviewer.review([case['image_path']],case['candidate'],case['reference'],
                    missions[case['mission_id']],case['rubric'],[],case['case_id']+'-preflight',run.path/'raw/reviewer')
                preflight.append(result)
                print('Reviewer preflight:',result['status'],'seconds',result['wall_seconds'],flush=True)
                if result['status']!='complete':
                    print('Reviewer preflight error:',result.get('error'),result.get('message'),flush=True)
            write_json(run.path/'metrics/reviewer_technical_preflight.json',preflight)
            if any(r['status']!='complete' for r in preflight):raise RuntimeError('reviewer technical/JSON preflight failed')
            if preflight[0]['image_inputs'][0]['sha256']==preflight[1]['image_inputs'][0]['sha256']:
                raise RuntimeError('reviewer preflight did not supply distinct source files')
        with run.stage('02_annotation_assisted_reviewer_controls'):
            order=np.random.default_rng(17).permutation(len(cases));outputs=[]
            for index in order:
                run.budget();case=cases[int(index)]
                result=reviewer.review([case['image_path']],case['candidate'],case['reference'],missions[case['mission_id']],
                    case['rubric'],[],case['case_id'],run.path/'raw/reviewer')
                outputs.append(result);append_json(run.path/'reviews/control_reviews.jsonl',result)
                if len(outputs)%6==0:print('Control reviews',len(outputs),'/',len(cases),flush=True)
                write_json(run.path/'metrics/control_progress.json',{'completed':len(outputs),'total':len(cases)})
            control_metrics=score_controls(cases,outputs);write_json(run.path/'metrics/control_metrics.json',control_metrics)
            print('Reviewer control gate:',control_metrics,flush=True)
            if not control_metrics['engineering_gate_passed']:
                write_json(run.path/'status.json',{'status':'reviewer_gate_failed','producer_loaded':False,
                    'scope':'annotation-assisted auditing controls;not visual recognition benchmark'})
                return run.path
        with run.stage('03_gpu0_parameter_producer'):
            run.budget();clip_path,clip_revision=snapshot_model('openai/clip-vit-base-patch32')
            producer=PilotVisionProducer(clip_path,missions);producer.load()
            write_json(run.path/'models/producer.json',{'clip':producer.global_provider.provenance,
                        'clip_revision':clip_revision,'detector':producer.provenance})
            evidence={}
            for start in range(0,len(selected),8):
                run.budget();chunk=list(selected.values())[start:start+8];images=[]
                for row in chunk:
                    with Image.open(row['image_path']) as image:images.append(image.convert('RGB'))
                values=producer.extract_batch(images)
                for row,value in zip(chunk,values):
                    evidence[row['image_id']]=value
                    append_json(run.path/'features/evidence.jsonl',{'image_id':row['image_id'],'evidence':value.to_dict()})
                print('Visual evidence cached:',len(evidence),'/',len(selected),flush=True)
        with run.stage('04_parameters_histories_and_binary_reviews'):
            spec=MethodSpecification(detector='yolov8s-worldv2',producer_revisions={
                'clip_revision':clip_revision,'yolo_sha256':producer.provenance['weights_sha256']})
            extractor=PilotParameterExtractor(spec);memories={};results=[]
            for observation in observations:
                run.budget();binding=observation['binding'];row=observation['record'];mission=missions[binding.mission_id]
                memory=memories.setdefault((binding.ue_id,mission.version),PilotMemory(binding.ue_id,mission.mission_id,64))
                history=memory.history();obs_id=observation['observation_id'];value=evidence[row['image_id']]
                bundle=extractor.extract_evidence(value,mission,{'image_token':obs_id,'source_id':binding.ue_id},memory)
                append_json(run.path/'parameters/parameters.jsonl',{'observation_id':obs_id,'image_id':row['image_id'],
                    'ue_id':binding.ue_id,'bundle':bundle.to_dict(),'history_digest':digest(history)})
                reference=native_reference(row);rubric=make_rubric(mission,reference)
                append_json(run.path/'references/references.jsonl',{'observation_id':obs_id,'reference':reference,'rubric':rubric})
                views={view:representation(bundle,mission,view) for view in ('compact','structured')}
                # Candidate writes are complete before advancing sender state.
                for view,candidate in views.items():append_json(run.path/'parameters/representations.jsonl',{
                    'observation_id':obs_id,'view':view,'candidate':candidate})
                memory.commit(value,observation['step'],{'image_id':row['image_id'],'observation_id':obs_id,
                              'image_path':row['image_path'],'mission_version':mission.version})
                append_json(run.path/'parameters/sequence_trace.jsonl',{'observation_id':obs_id,
                    'ue_id':binding.ue_id,'mission_version':mission.version,'history_before':history,
                    'history_after':memory.history(),'kb_version_after':memory.version,
                    'sequence_kind':observation['sequence_kind']})
                past=[entry['observation']['image_path'] for entry in history[-2:] if entry['observation'].get('image_path')]
                for view,candidate in views.items():
                    run.budget();case_id=obs_id+'-'+view
                    reviewed=reviewer.review([row['image_path'],*past],candidate,reference,mission,rubric,
                         history[-2:],case_id,run.path/'raw/reviewer')
                    result={'observation_id':obs_id,'ue_id':binding.ue_id,'mission_id':mission.mission_id,
                            'view':view,'review':reviewed,'direct_reference_metrics':direct_fact_metrics(candidate['claims'],reference),
                            'reference_grading_available':rubric['reference_grading_available'],
                            'history_review_scope':'last2supplied prior views;not full-history new-fact certification'}
                    results.append(result);append_json(run.path/'reviews/parameter_reviews.jsonl',result)
                print('Parameter observation',obs_id,'complete; KB version',memory.version,flush=True)
            write_json(run.path/'metrics/producer_timing.json',producer.timings)
            write_json(run.path/'metrics/parameter_summary.json',summarize(results))
        write_json(run.path/'status.json',{'status':'complete','parent_images':len(selected),
            'ue_observations':len(observations),'binary_parameter_reviews':len(results),
            'seconds':time.monotonic()-run.started,'allocated_gpu_hours':2*(time.monotonic()-run.started)/3600,
            'actual_scheduler_training':False})
        return run.path
    except Exception as error:
        write_json(run.path/'status.json',{'status':'failed','error':type(error).__name__,'message':str(error),
            'seconds':time.monotonic()-run.started,'allocated_gpu_hours':2*(time.monotonic()-run.started)/3600})
        raise
    finally:
        if producer is not None:producer.close()
        if reviewer is not None:reviewer.close()


def summarize(results):
    output={}
    for view in ('compact','structured'):
        rows=[r for r in results if r['view']==view];valid=[r for r in rows if r['review']['status']=='complete']
        total=sum(r['direct_reference_metrics'].get('reference_fact_count',
            r['direct_reference_metrics']['known_comparable_facts']+
            len(r['direct_reference_metrics'].get('missing_reference_facts_in_representation',[]))+
            len(r['direct_reference_metrics'].get('invalid_reference_fact_values',[]))) for r in rows)
        compared=sum(r['direct_reference_metrics']['known_comparable_facts'] for r in rows)
        correct=sum(r['direct_reference_metrics']['correct'] for r in rows)
        output[view]={'attempts':len(rows),'valid_reviews':len(valid),
            'consistent_fraction':sum(r['review']['parsed']['parameters_consistent'] for r in valid)/len(valid) if valid else None,
            'sufficient_fraction':sum(r['review']['parsed']['representation_sufficient'] for r in valid)/len(valid) if valid else None,
            'reference_graded_observations':sum(r['reference_grading_available'] for r in rows),
            'direct_known_fact_correct':correct,'direct_known_fact_comparisons':compared,
            'direct_reference_fact_count':total,'direct_reference_coverage':compared/total if total else None,
            'direct_all_reference_correct_fraction':correct/total if total else None,
            'interpretation':'annotation-assisted model assessment;not universal scene completeness'}
    return output
