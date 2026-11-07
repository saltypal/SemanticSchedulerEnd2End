"""Versioned logical decoding and native-fact evaluation, separate from VLMs."""
from copy import deepcopy
from .contracts import digest

BUILDING_FACTS=('building.present','building.affected_or_greater')
BUILDING_CONTRACT={
    'id':'ladi_building_logic_v1',
    'scope':'native LADI building presence and affected-or-greater claims only',
    'required':list(BUILDING_FACTS),
    'unknown':'missing key, null, or availability=false; never detector non-detection',
    'rules':[
        {'id':'affected_implies_present','if':['building.affected_or_greater',True],
         'then':['building.present',True]},
        {'id':'absent_implies_unaffected','if':['building.present',False],
         'then':['building.affected_or_greater',False]}],
    'truth_caution':'Deriving a claim from a prediction does not establish its factual correctness.'}


def decode_claims(candidate,contract=BUILDING_CONTRACT):
    """Never use references, imagery, scores or future observations in decoding."""
    claims=candidate.get('claims',{})
    if not isinstance(claims,dict):raise ValueError('claims must be a mapping')
    names=set(contract['required'])
    for rule in contract['rules']:names.update((rule['if'][0],rule['then'][0]))
    availability=candidate.get('claim_availability',{})
    if not isinstance(availability,dict):raise ValueError('claim availability must be a mapping')
    explicit={};origins={};unavailable=[]
    for name in sorted(names):
        value=claims.get(name)
        mask=availability.get(name,{})
        if not isinstance(mask,dict):raise ValueError('invalid claim availability row')
        if mask.get('available') is False:value=None
        if value is None:unavailable.append(name);continue
        if type(value) is not bool:raise ValueError('named claims must be boolean or null: '+name)
        explicit[name]=value;origins[name]={'kind':'explicit','field':'claims.'+name}
    decoded=dict(explicit);conflicts={}
    # Rules are monotone: they add values and never silently replace an explicit claim.
    for _ in range(len(names)+1):
        added=False
        for rule in contract['rules']:
            source,value=rule['if'];target,derived=rule['then']
            if source not in decoded or decoded[source] is not value:continue
            if target in decoded:
                if decoded[target] is not derived:
                    conflicts[(target,rule['id'])]={'field':target,'retained_value':decoded[target],
                        'derived_value':derived,'rule':rule['id'],'premise':source}
                continue
            decoded[target]=derived;origins[target]={'kind':'derived','rule':rule['id'],'premise':source}
            added=True
        if not added:break
    required=contract['required']
    return {'contract_id':contract['id'],'contract_sha256':digest(contract),
            'explicit_claims':explicit,'decoded_claims':decoded,'origins':origins,
            'literal_complete':all(name in explicit for name in required),
            'literal_missing':[name for name in required if name not in explicit],
            'semantic_complete':all(name in decoded for name in required) and not conflicts,
            'semantic_missing':[name for name in required if name not in decoded],
            'conflicts':list(conflicts.values()),'unavailable_input_claims':unavailable}


def audit_candidate(candidate,reference,contract=BUILDING_CONTRACT):
    result=decode_claims(candidate,contract)
    facts=reference.get('facts',{})
    if not isinstance(facts,dict):raise ValueError('reference facts must be a mapping')
    for name,value in facts.items():
        if value is not None and type(value) is not bool:raise ValueError('reference fact must be boolean or null: '+name)
    known={k:facts[k] for k in contract['required'] if type(facts.get(k)) is bool}
    reference_decoded=decode_claims({'claims':known},contract)
    if reference_decoded['conflicts']:raise ValueError('inconsistent native reference hierarchy')
    compared={k:{'prediction':result['decoded_claims'][k],'reference':v,
                 'correct':result['decoded_claims'][k] is v}
              for k,v in known.items() if k in result['decoded_claims']}
    wrong=[k for k,row in compared.items() if not row['correct']]
    consistency=(False if wrong or result['conflicts'] else True if compared else None)
    # Partial references remain partial even when their ontology suggests another label.
    # We do not manufacture extra source annotation truth through logical closure.
    all_verified=(False if wrong or result['conflicts'] or not result['semantic_complete'] else
                  True if len(known)==len(contract['required']) else None)
    return {**result,'factual':{'known_reference_facts':known,'compared':compared,
                'wrong_fields':wrong,'known_subset_consistent':consistency,
                'all_required_verified':all_verified,
                'reference_unknown':[k for k in contract['required'] if k not in known]},
            'assessment_scope':'declared native facts, not full scene truth'}


def semantic_rubric():
    return {'rubric_id':digest(BUILDING_CONTRACT),'required_fact_names':list(BUILDING_FACTS),
            'rule':'Decode only using the supplied contract, then compare to known native reference facts.',
            'numerical_checks':'literal completeness and logical closure are checked independently by code',
            'decoding_contract':deepcopy(BUILDING_CONTRACT)}
