"""Staged search contracts and independent-label score replay, with no VLM objective."""
from pathlib import Path
from itertools import product
from copy import deepcopy
import argparse
import json
import math
from .artifacts import write_json
from .contracts import digest,canonical
from .claim_repair_pilot import binary_metrics
from .image_claims import NATIVE_FACTS
from .annotation_review import REQUIRED

PREPROCESSING=('checkpoint_native','aspect_preserving_pad')
REPRESENTATIONS=('required_claims','claims_and_scores','mission_context_history','full_structured')
TARGETS=('building.present','road.present','building.affected_or_greater','road.damaged')


def project_representation(candidate,mission_id,profile):
    """Vary transmitted evidence using the same predictions; never infer new claims."""
    if profile not in REPRESENTATIONS or mission_id not in REQUIRED:raise ValueError('unknown representation/mission')
    result=deepcopy(candidate)
    if profile=='full_structured':return {**result,'search_profile':profile}
    wanted=set(REQUIRED[mission_id]);result['claims']={k:v for k,v in result.get('claims',{}).items() if k in wanted}
    result['claim_availability']={k:v for k,v in result.get('claim_availability',{}).items() if k in wanted}
    params=result.get('parameters',{});scene=params.get('scene',{}).get('value',{}) or {}
    scores={key:value for key,value in scene.get('image_claim_predictions',{}).items() if key in wanted}
    if profile=='required_claims':result['parameters']={}
    elif profile=='claims_and_scores':
        result['parameters']={'claim_scores':{'available':bool(scores),'value':scores or None,
            'method':'projection_of_existing_image_claim_scores','reason':None if scores else 'no existing claim scores'}}
    else:
        result['parameters']={k:v for k,v in params.items() if k in ('relevance','novelty','rarity','reliability')}
        result['parameters']['claim_scores']={'available':bool(scores),'value':scores or None,
            'method':'projection_of_existing_image_claim_scores','reason':None if scores else 'no existing claim scores'}
    result['search_profile']=profile
    result['output_role']='information_projection_from_identical_predictions'
    result['is_ablation']=True
    return result


def search_plan():
    native=[{'preprocessing':p,'threshold':t,'abstention_half_width':a}
            for p,t,a in product(PREPROCESSING,(.3,.5,.7),(0.,.1))]
    detector=[{'image_size':size,'confidence':conf,'iou':iou,'max_detections':300,
        'vocabulary':'canonical_16','fusion':'evaluate raw detections before fusion'}
        for size,conf,iou in product((640,1024),(.05,.15,.25),(.5,.7))]
    clip=[{'preprocessing':p,'text_source':t,'condition_prompt_set':s}
          for p,t,s in product(PREPROCESSING,('paragraph','observable_clause'),('historical','balanced_aerial'))]
    reviewer=[{'prompt_variant':p,'max_new_tokens':tokens,'claim_scope':'mission','max_pixels':pixels,
        'image_max_side':None if pixels>512*512 else 512}
        for p,tokens,pixels in product(('candidate_provenance_v2','candidate_provenance_concise_v3'),
                                     (192,384),(512*512,768*768))]
    return {'schema':'staged_approach_grid_v1','purpose':'faithful task evidence, not more approval',
        'selection':{'objective':'per-task independent-label balanced accuracy including unavailable predictions',
            'constraints':['report class supports,coverage,all-reference correctness and constant baselines',
                'no selection on old test20, c/e controls or test outputs',
                'fit thresholds/calibration only in development groups; lock fresh confirmation before scoring',
                'a profile is not favored because it transmits fewer assessable claims'],
            'tie_break':['less incremental inference cost','fewer transmitted bytes'],
            'reviewer_objective':'reject corrupt/omitted controls and agree with known annotation facts, not positive approval'},
        'stages':{
            'native_score_replay':{'configurations':native,'count':len(native),
                'model_forward_cache_keys':list(PREPROCESSING),'label_independent_forward_passes_per_image':2},
            'clip_alignment':{'configurations':clip,'count':len(clip),'note':'cosines remain continuous; no probability or native-label truth'},
            'detector':{'configurations':detector,'count':len(detector),
                'note':'box precision/recall needs localized labels; image-presence proxies cannot validate boxes',
                'cache':'one low-confidence inference per image_size/IoU; higher confidence replay only above retained floor'},
            'representations':{'profiles':list(REPRESENTATIONS),'count':len(REPRESENTATIONS),
                'native_cross_product_rows':len(native)*len(REPRESENTATIONS),
                'note':'reuse identical predictions; dropping history cannot repair claim accuracy'},
            'reviewer_controls':{'configurations':reviewer,'count':len(reviewer),
                'maximum_cases_per_configuration':24,'gates_unchanged':True,
                'note':'test prompts on separate complete/omitted/contradicted controls before judging actual extraction'}},
        'resources':{'local_first':True,'new_gpu_runs_authorized_by_this_file':False,
            'pilot_max_parents':20,'pilot_cpu_inference_seconds':180,
            'stop':'missing class support or labels is inconclusive, never a pass'},
        'not_yet_validated':['cross-view new facts','natural urgency','household/road-blockage truth','spatial geometry without box labels']}


def decide(score,threshold,half_width):
    if type(score) not in (int,float) or not 0<=score<=1:raise ValueError('finite unit score required')
    if type(threshold) not in (int,float) or not 0<threshold<1 or not 0<=half_width<min(threshold,1-threshold):
        raise ValueError('invalid threshold/abstention band')
    lower,upper=threshold-half_width,threshold+half_width
    at_edge=math.isclose(score,lower,rel_tol=0.,abs_tol=1e-12) or math.isclose(score,upper,rel_tol=0.,abs_tol=1e-12)
    return None if lower<score<upper and not at_edge else bool(score>threshold)


def evaluate_cache(caches,records,output,role='diagnostic'):
    """Diagnostic only: intentionally cannot select a winner from exposed test data."""
    if role!='diagnostic':raise ValueError('selection requires separate grouped development/final manifests; this command is diagnostic only')
    if not records or len({r['image_sha256'] for r in records})!=len(records):raise ValueError('unique nonempty image records required')
    expected={r['image_sha256'] for r in records};reverse={fact:label for label,fact in NATIVE_FACTS.items()}
    for policy,predictions in caches.items():
        if policy not in PREPROCESSING or set(predictions)!=expected:raise ValueError('cache policy/identities mismatch')
    output=Path(output);output.mkdir(exist_ok=False,parents=True);rows=[]
    configurations=search_plan()['stages']['native_score_replay']['configurations']
    for cfg in configurations:
        if cfg['preprocessing'] not in caches:continue
        cache=caches[cfg['preprocessing']];metrics={}
        for fact in TARGETS:
            cohort=[r for r in records if type(r['native_labels'].get(reverse[fact])) is bool]
            truth=[r['native_labels'][reverse[fact]] for r in cohort]
            predicted=[decide(cache[r['image_sha256']][fact]['score'],cfg['threshold'],cfg['abstention_half_width']) for r in cohort]
            metrics[fact]={'candidate':binary_metrics(truth,predicted),
                'always_positive':binary_metrics(truth,[True]*len(truth)),
                'always_negative':binary_metrics(truth,[False]*len(truth))}
        row={'configuration':cfg,'configuration_sha256':digest(cfg),'metrics':metrics,
             'source_cache_digest':digest(cache),'role':role}
        rows.append(row)
    with (output/'grid_rows.jsonl').open('w',encoding='utf-8') as handle:
        for row in rows:handle.write(canonical(row)+'\n')
    write_json(output/'status.json',{'status':'complete','evaluated_configurations':len(rows),
        'image_count':len(records),'selected_configuration':None,
        'reason':'exposed diagnostic data; no tuning or superiority claim',
        'label_record_digest':digest(records),'plan':search_plan()})
    return rows


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--output',required=True)
    p=sub.add_parser('replay');p.add_argument('--labels',required=True);p.add_argument('--output',required=True)
    p.add_argument('--native-cache',required=True);p.add_argument('--pad-cache')
    args=parser.parse_args()
    if args.command=='plan':write_json(args.output,search_plan());return
    caches={}
    for policy,path in (('checkpoint_native',args.native_cache),('aspect_preserving_pad',args.pad_cache)):
        if path:
            entries=[json.loads(line) for line in Path(path).read_text().splitlines()]
            if len({r['image_sha256'] for r in entries})!=len(entries):raise ValueError('duplicate cache identities')
            caches[policy]={r['image_sha256']:r['claims'] for r in entries}
    labels=[json.loads(line) for line in Path(args.labels).read_text().splitlines()]
    print('Executed configurations:',len(evaluate_cache(caches,labels,args.output)),flush=True)


if __name__=='__main__':main()
