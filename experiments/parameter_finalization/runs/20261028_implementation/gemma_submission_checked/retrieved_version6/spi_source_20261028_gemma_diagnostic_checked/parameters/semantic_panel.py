"""Evaluator-side balanced controls and metrics for the declared semantic contract."""
from copy import deepcopy
from .annotation_review import native_reference
from .semantic_contract import BUILDING_FACTS,audit_candidate,semantic_rubric
from .official_panel import stratum


def panel_cases(records):
    counts={k:0 for k in ('absent','present_unaffected','affected')};seen=set();cases=[]
    for record in records:
        kind=stratum(record['native_labels'])
        if kind not in counts:raise ValueError('panel requires known, consistent building annotations')
        if record['group'] in seen:raise ValueError('duplicate parent group in panel')
        seen.add(record['group']);counts[kind]+=1
        reference=native_reference(record)
        truth={k:reference['facts'][k] for k in BUILDING_FACTS}
        variants={'complete':dict(truth),'condition_omitted':{BUILDING_FACTS[0]:truth[BUILDING_FACTS[0]]},
                  'wrong_presence':{**truth,BUILDING_FACTS[0]:not truth[BUILDING_FACTS[0]]},
                  'wrong_condition':{**truth,BUILDING_FACTS[1]:not truth[BUILDING_FACTS[1]]}}
        for arm,claims in variants.items():
            candidate={'representation':'semantic_contract_control_v1','claims':claims,'parameters':{}}
            if arm=='condition_omitted' and counts[kind]%2==0:
                arm='condition_masked'
                candidate['claims']={**truth,BUILDING_FACTS[1]:not truth[BUILDING_FACTS[1]]}
                candidate['claim_availability']={BUILDING_FACTS[1]:{'available':False,'reason':'producer evidence unavailable'}}
            cases.append(make_case(record,candidate,'control-'+str(record['source_row'])+'-'+arm,arm))
    if len(set(counts.values()))!=1 or not all(counts.values()):raise ValueError('panel strata must be balanced')
    return cases


def make_case(record,candidate,identity,arm='actual_extractor'):
    reference=native_reference(record);audit=audit_candidate(candidate,reference)
    return {'case_id':identity,'image_id':record['image_id'],'image_path':record['image_path'],
            'mission_id':'building_damage','stratum':stratum(record['native_labels']),
            'reference':reference,'candidate':deepcopy(candidate),'rubric':semantic_rubric(),
            'arm':arm,'audit':audit,'expected':{
                'parameters_consistent':audit['factual']['known_subset_consistent'],
                'representation_sufficient':audit['factual']['all_required_verified']}}


def score_panel(cases,reviews):
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('duplicate case identities')
    by_id={r['case_id']:r for r in reviews}
    if len(by_id)!=len(reviews) or not set(by_id)<={c['case_id'] for c in cases}:
        raise ValueError('duplicate or unknown reviewer identities')
    endpoints={}
    for key in ('parameters_consistent','representation_sufficient'):
        counts={False:0,True:0};correct={False:0,True:0};matrix={'tp':0,'tn':0,'fp':0,'fn':0};unknown=0
        for case in cases:
            expected=case['expected'][key]
            if expected is None:unknown+=1;continue
            if type(expected) is not bool:raise ValueError('expected verdict must be boolean or unavailable')
            counts[expected]+=1;row=by_id.get(case['case_id'],{})
            if row.get('status')!='complete':continue
            predicted=row['parsed'][key]
            if type(predicted) is not bool:raise ValueError('parsed verdict must be boolean')
            correct[expected]+=int(expected is predicted)
            matrix[('tp' if predicted else 'fn') if expected else ('fp' if predicted else 'tn')]+=1
        recall={str(k):correct[k]/counts[k] if counts[k] else None for k in (False,True)}
        endpoints[key]={'confusion':matrix,'class_support':{str(k):v for k,v in counts.items()},
            'class_correct_including_invalid':recall,'unknown_reference_cases':unknown,
            'balanced_accuracy':sum(recall.values())/2 if all(v is not None for v in recall.values()) else None}
    gradable=[c for c in cases if all(type(v) is bool for v in c['expected'].values())]
    joint=sum(by_id.get(c['case_id'],{}).get('status')=='complete' and
              all(by_id[c['case_id']]['parsed'][k] is v for k,v in c['expected'].items()) for c in gradable)
    conditional=[c for c in cases if c['arm'] in ('condition_omitted','condition_masked') and c['stratum']!='absent']
    recoverable=[c for c in cases if c['arm'] in ('condition_omitted','condition_masked') and c['stratum']=='absent']
    def fraction(items,key,expected):
        return sum(by_id.get(c['case_id'],{}).get('status')=='complete' and
                   by_id[c['case_id']]['parsed'][key] is expected for c in items)/len(items) if items else None
    valid=sum(r.get('status')=='complete' for r in reviews)
    missing_rejection=fraction(conditional,'representation_sufficient',False)
    closure_acceptance=fraction(recoverable,'representation_sufficient',True)
    applicable=missing_rejection is not None and closure_acceptance is not None
    passed=(valid/len(cases)>=.95 and missing_rejection>=.75 and closure_acceptance>=.75 and all(
        endpoint['balanced_accuracy'] is not None and endpoint['balanced_accuracy']>=.8 and
        all(v is not None and v>=.75 for v in endpoint['class_correct_including_invalid'].values())
        for endpoint in endpoints.values())) if applicable else None
    return {'cases':len(cases),'parent_groups':len({c['reference']['group'] for c in cases}),
        'valid':valid,'fully_gradable':len(gradable),
        'joint_correct':joint,'endpoints':endpoints,
        'genuinely_missing_condition_rejection':missing_rejection,
        'declared_recoverable_absence_acceptance':closure_acceptance,
        'engineering_gate_passed':passed,
        'scope':'versioned semantic closure and native image labels; not generic full-scene understanding'}
