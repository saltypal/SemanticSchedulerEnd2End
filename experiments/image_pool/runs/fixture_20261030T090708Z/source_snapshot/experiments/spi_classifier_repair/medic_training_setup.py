"""Freeze image-only SPI roles for the exact MEDIC detector cohort, before labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from parameters.data import add_duplicate_groups

def assign_roles(records,exposed_hashes=(),seed=17):
    rows=copy.deepcopy(records)
    if len(rows)!=300 or any(r.get('source')!='medic' for r in rows):
        raise ValueError('requires exact 300-image MEDIC-only cohort')
    identities=set()
    for r in rows:
        actual=hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()
        if actual!=r['image_sha256'] or actual in identities:raise ValueError('image content identity mismatch/duplicate')
        identities.add(actual);r.setdefault('parent_id',r['image_id'])
    rows=add_duplicate_groups(rows)
    groups={}
    for r in rows:groups.setdefault(r['group'],[]).append(r)
    exposed=set(exposed_hashes)|{r['image_sha256'] for r in rows if r.get('development_exposed') is True}
    candidates=[g for g,members in sorted(groups.items()) if not any(r['image_sha256'] in exposed for r in members)]
    candidates=np.random.default_rng(seed).permutation(candidates).tolist()
    roles={};counts={'test':0,'validation':0,'train':0}
    # Whole components are never broken for nominal 180/60/60 counts.
    for role in ('test','validation'):
        while candidates and counts[role]<60:
            group=candidates.pop();roles[group]=role;counts[role]+=len(groups[group])
    for group in groups:
        roles.setdefault(group,'train')
    for r in rows:
        r['split']=roles[r['group']];r['development_exposed']=r['image_sha256'] in exposed
    counts={role:sum(r['split']==role for r in rows) for role in counts}
    return rows,{'actual_counts':counts,'nominal_counts':{'train':180,'validation':60,'test':60},
                 'seed':seed,'grouping':'parent/exact/dHash components; event independence unverified',
                 'labels_used_for_split':False,'fresh_test_excludes_known_exposure':True}

def config():
    return {'schema':'medic_spi_training_v1','dataset':'MEDIC','images':300,'classes':['Low','Medium','High','Critical'],
      'teacher':{'selection_file':'parameters/config/final_labeling_teacher.json','thinking':False,
                 'student_target':'image_only','associated_text_target':'separate_offline_message_priority'},
      'encoders':{'clip_b32':'openai/clip-vit-base-patch32','siglip2_b16':'google/siglip2-base-patch16-224',
        'dino':'facebook/dinov3-vits16-pretrain-lvd1689m','dino_access_fallback':'facebook/dinov2-small',
        'convnextv2_tiny':'facebook/convnextv2-tiny-22k-224'},
      'vector_heads':['logistic','ridge','linear_svm','rbf_svm','knn','random_forest','extra_trees','hist_boost','mlp'],
      'token_heads':['gated_abmil','single_query_cross_attention'],'token_encoders':['siglip2_b16','dino'],
      'selection':{'hyperparameters':'grouped3fold_training_only','family':'development_macro_f1',
                    'test':'locked until complete pipeline frozen; no retuning after opening'},
      'resource_metrics':['cpu_gpu_latency_p50_p95','peak_memory','deployed_bytes','forward_flops_with_coverage',
                          'training_flops_with_coverage_or_explicit_unavailable','fit_and_search_seconds'],
      'release':{'minimum_class_recall':.70,'export_parity':1e-4,'four_classes_required':True,
                 'independent_urgency_validated':False},
      'upstream':'exact version18 MEDIC acquisition manifest; no redownload/resample or mixed LADI cohort'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--exposed-records',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    original=json.loads(Path(a.exposed_records).read_text())
    rows,receipt=assign_roles(json.loads(Path(a.records).read_text()),[r['image_sha256'] for r in original])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('records.json',rows),('split_receipt.json',receipt),('config.json',config())]:
        (out/name).write_text(json.dumps(value,indent=2))
    print('MEDIC SPI roles frozen',receipt,flush=True)
