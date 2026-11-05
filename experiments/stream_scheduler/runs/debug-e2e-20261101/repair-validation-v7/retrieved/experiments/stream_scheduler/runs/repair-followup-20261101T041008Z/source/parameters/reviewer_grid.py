"""Run reviewer-control configurations on one loaded model; no approval optimization.

This runner is intentionally separate from extractor search. It requires caller-
provided, resolved image cases and does not launch or allocate a cloud instance.
"""
from dataclasses import asdict
from pathlib import Path
import json
import time
from .approach_grid import search_plan
from .artifacts import write_json
from .annotation_review import score_controls
from .model_settings import ReviewerSettings
from .contracts import canonical,digest


def run(reviewer,cases,missions,output,excluded_groups=None,seconds=1800):
    """Bounded callback runner, with real reviewer inference supplied by the caller."""
    if len(cases)!=24:raise ValueError('reviewer search requires24cases from8paired groups')
    if not excluded_groups:raise ValueError('explicit nonempty prior-group exclusion evidence required')
    if len({c['reference']['group'] for c in cases})!=8:raise ValueError('reviewer search requires8distinct groups')
    if {c['reference']['group'] for c in cases}&set(excluded_groups):raise ValueError('previous reviewer groups must be excluded')
    if len({c['case_id'] for c in cases})!=24:raise ValueError('duplicate review case identities')
    groups={group:[c for c in cases if c['reference']['group']==group] for group in {c['reference']['group'] for c in cases}}
    target_balance={False:0,True:0}
    for triplet in groups.values():
        if len(triplet)!=3 or {c['control_arm'] for c in triplet}!={'complete','incomplete','contradicted'}:
            raise ValueError('each group needs one complete/incomplete/contradicted triplet')
        if len({c['image_id'] for c in triplet})!=1 or len({canonical(c['reference']) for c in triplet})!=1:
            raise ValueError('paired triplet must share image and reference')
        target=triplet[0]['reference']['facts'].get('building.affected_or_greater')
        if type(target) is not bool:raise ValueError('known boolean control corruption target required')
        target_balance[target]+=1
    if target_balance!={False:4,True:4}:raise ValueError('control corruption target must be balanced4/4 groups')
    if any(not Path(c['image_path']).is_file() for c in cases):raise ValueError('all reviewer images must be locally resolved')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();results=[]
    # Expected outcomes remain in evaluator-only files and never enter reviewer.review.
    write_json(output/'cases_evaluator_only.json',cases)
    write_json(output/'exclusions_evaluator_only.json',{'groups':sorted(set(excluded_groups)),
        'sha256':digest(sorted(set(excluded_groups)))})
    for config in search_plan()['stages']['reviewer_controls']['configurations']:
        if time.perf_counter()-started>seconds:raise TimeoutError('reviewer search time cap')
        settings=ReviewerSettings(**config);key=digest(config)[:16];reviewer.configure(settings)
        if not getattr(reviewer,'provenance',None):raise ValueError('configured reviewer needs model provenance')
        write_json(output/(key+'_model.json'),reviewer.provenance)
        reviews=[]
        with (output/(key+'.jsonl')).open('w',encoding='utf-8') as handle:
            for case in cases:
                if time.perf_counter()-started>seconds:raise TimeoutError('reviewer search time cap')
                review=reviewer.review([case['image_path']],case['candidate'],case['reference'],
                    missions[case['mission_id']],case['rubric'],[],key+'-'+case['case_id'],output/'raw')
                # Normalize callback identity only; never alter status or verdicts.
                review={**review,'case_id':case['case_id']};reviews.append(review)
                handle.write(canonical(review)+'\n');handle.flush()
            metrics=score_controls(cases,reviews)
        result={'settings':asdict(settings),'metrics':metrics};results.append(result)
        write_json(output/'progress.json',results)
        print('Reviewer setting',key,'valid',metrics['valid'],'gate',metrics['engineering_gate_passed'],flush=True)
    write_json(output/'status.json',{'status':'complete','configurations':len(results),
        'selected_prompt':None,'reason':'controls only; independent confirmation and scope-specific labels required',
        'seconds':time.perf_counter()-started})
    return results
