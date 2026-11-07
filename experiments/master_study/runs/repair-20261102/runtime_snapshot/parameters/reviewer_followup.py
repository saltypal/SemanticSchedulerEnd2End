"""Paired prompt repair: exposed modality diagnostic and fresh-group controls."""
from dataclasses import asdict
from pathlib import Path
import json
import time
from .artifacts import Run,write_json
from .annotation_pilot import resolve_image
from .annotation_review import control_cases,native_reference,score_controls
from .contracts import canonical
from .model_settings import ReviewerSettings
from .pilot_producers import snapshot_model
from .qwen_reviewer import QwenAnnotationReviewer
from .ue_profiles import load_ue_profiles

PROMPTS=('candidate_provenance_v2','candidate_provenance_explicit_v4')
REVISION='ebb281ec70b05090aa6165b016eac8ec08e71b17'


def prepare_cases(records,missions,spec,manifest):
    if any(r.get('role') in ('final','locked_final') or r.get('protected_final') for r in records):
        raise ValueError('protected final source is not allowed')
    if spec['fresh_control_parents']!=8 or spec['max_review_calls']!=72:
        raise ValueError('unexpected follow-up cohort limits')
    excluded=set(spec['excluded_groups'])
    if len(excluded)<56:raise ValueError('prior reviewer control groups must all be excluded')
    by_id={r['image_id']:r for r in records}
    diagnostic=spec['diagnostic_cases']
    if len(diagnostic)!=6:raise ValueError('six exposed diagnostic cases required')
    groups={c['reference']['group'] for c in diagnostic}
    if len(groups)!=2 or not groups<=excluded:raise ValueError('diagnostic group provenance differs')
    for case in diagnostic:
        if case['reference']!=native_reference(by_id[case['image_id']]):
            raise ValueError('old diagnostic reference no longer matches source')
    fresh=control_cases(records,missions,parents=8,excluded_groups=excluded)
    selected={c['image_id'] for c in diagnostic+fresh}
    paths={key:resolve_image(by_id[key],manifest)['image_path'] for key in selected}
    for case in diagnostic+fresh:case['image_path']=paths[case['image_id']]
    return diagnostic,fresh


def evaluate_cell(reviewer,cases,missions,settings,output,budget,text_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    budget();reviewer.configure(settings)
    write_json(output/'model.json',reviewer.provenance)
    results=[];started=time.perf_counter()
    with (output/'reviews.jsonl').open('w',encoding='utf-8') as handle:
        for i,case in enumerate(cases):
            budget()
            result=reviewer.review([] if text_only else [case['image_path']],case['candidate'],
                case['reference'],missions[case['mission_id']],case['rubric'],[],case['case_id'],
                output/'raw',text_only_diagnostic=text_only)
            results.append(result);handle.write(canonical(result)+'\n');handle.flush()
            if (i+1)%6==0:print(output.name,i+1,'/',len(cases),flush=True)
    metrics=score_controls(cases,results)
    by_id={r['case_id']:r for r in results}
    joint=sum(by_id[c['case_id']]['status']=='complete' and
              all(by_id[c['case_id']]['parsed'][k]==v for k,v in c['expected'].items()) for c in cases)
    summary={'settings':asdict(settings),'text_only_diagnostic':text_only,'metrics':metrics,
             'joint_correct':joint,'wall_seconds':time.perf_counter()-started}
    write_json(output/'metrics.json',summary)
    print('REVIEW CELL END',output.name,'joint',joint,'/',len(cases),
          'valid',metrics['valid'],'gate',metrics['engineering_gate_passed'],flush=True)
    return summary


def run(manifest,output,cohort,seconds=900):
    import torch
    if torch.cuda.device_count()!=2:raise RuntimeError('two assigned GPUs required; reviewer owns GPU1')
    run=Run(output,{'seconds':seconds,'max_reviews':72,'prompts':PROMPTS,
                   'scope':'prompt repair controls; no extraction or policy training'})
    reviewer=None
    try:
        with run.stage('00_cohort_preflight'):
            spec=json.loads(Path(cohort).read_text(encoding='utf-8-sig'))
            missions,_=load_ue_profiles()
            diagnostic,fresh=prepare_cases(json.loads(Path(manifest).read_text()),missions,spec,manifest)
            write_json(run.path/'data/cohort_spec.json',spec)
            write_json(run.path/'data/diagnostic_evaluator_only.json',diagnostic)
            write_json(run.path/'data/fresh_evaluator_only.json',fresh)
            write_json(run.path/'hardware.json',{'gpu_names':[torch.cuda.get_device_name(i) for i in (0,1)],
                'torch':torch.__version__,'cuda':torch.version.cuda,'inference_device':'cuda:1',
                'gpu0_inference':False})
            print('Verified six exposed cases and 24 fresh cases;excluded groups',len(spec['excluded_groups']),flush=True)
        with run.stage('01_model_load'):
            path,revision=snapshot_model('Qwen/Qwen3-VL-4B-Instruct',revision=REVISION)
            reviewer=QwenAnnotationReviewer(path,revision)
        summaries={}
        for prompt in PROMPTS:
            settings=ReviewerSettings(max_new_tokens=192,max_pixels=512*512,image_max_side=512,
                                      prompt_variant=prompt,claim_scope='mission')
            for label,cases,text_only in (('diagnostic_image',diagnostic,False),
                                           ('diagnostic_text',diagnostic,True),('fresh_image',fresh,False)):
                key=prompt+'_'+label
                with run.stage(key):
                    summaries[key]=evaluate_cell(reviewer,cases,missions,settings,run.path/key,run.budget,text_only)
                    write_json(run.path/'summary.json',summaries)
        write_json(run.path/'status.json',{'status':'complete','review_calls':72,'seconds':time.monotonic()-run.started,
            'passing_fresh_configurations':sum(s['metrics']['engineering_gate_passed'] for key,s in summaries.items()
                                              if key.endswith('_fresh_image')),
            'selected_configuration':None,'no_scheduler_training':True})
    except Exception as error:
        write_json(run.path/'status.json',{'status':'failed','error':type(error).__name__,'message':str(error)})
        raise
    finally:
        if reviewer is not None:reviewer.close()
    return run.path
