#!/usr/bin/env python3
"""Bounded, isolated scene_arrays_v2 contract check; no detector/model inference."""
from __future__ import annotations
import hashlib, importlib.util, json, math, platform, re, struct, sys, time, traceback
from pathlib import Path

START=time.perf_counter()
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[4]
PARAM=REPO/'parameters'
DOC=REPO/'docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md'
BASE=REPO/'experiments/semantic_priority_design/runs/scene93-temporal-v2-20261030/MASTER.before.md'
SOURCE_PATHS=[PARAM/'scene_arrays.py', PARAM/'scene_vocabulary.py']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source_before={str(p):sha(p) for p in SOURCE_PATHS}
doc_hash_before=sha(DOC)
checks=[]
def check(name, condition, cases, detail, outputs=None):
    checks.append({'name':name,'status':'PASS' if condition else 'FAIL','cases':cases,'detail':detail,'outputs':outputs or {}})
    if not condition: raise AssertionError(name+': '+detail)

def load_isolated():
    # Synthetic package name prevents parameters/__init__.py from executing.
    pkg='isolated_scene_contract'
    package=type(sys)(pkg); package.__path__=[str(PARAM)]; sys.modules[pkg]=package
    vs=importlib.util.spec_from_file_location(pkg+'.scene_vocabulary',PARAM/'scene_vocabulary.py')
    vocab=importlib.util.module_from_spec(vs); sys.modules[vs.name]=vocab; vs.loader.exec_module(vocab)
    spec=importlib.util.spec_from_file_location(pkg+'.scene_arrays',PARAM/'scene_arrays.py')
    arrays=importlib.util.module_from_spec(spec); sys.modules[spec.name]=arrays; spec.loader.exec_module(arrays)
    return arrays,vocab

try:
    a,v=load_isolated()
    check('isolated_import', 'parameters' not in [x for x in sys.modules if x=='parameters'], 2,
          'Loaded actual scene modules under a synthetic namespace; regular parameters package initializer was not imported.',
          {'scene_arrays':str(PARAM/'scene_arrays.py'),'scene_vocabulary':str(PARAM/'scene_vocabulary.py')})
    check('vocabulary_dimensions',len(a.CLASSES)==93 and len(a.OBJECTS)==85 and len(a.REGIONS)==8 and len(v.COCO_CLASSES)==80,
          4,'93 total slots = 80 COCO + 5 object extensions + 8 regions; 85 object-count slots.',
          {'classes':len(a.CLASSES),'objects':len(a.OBJECTS),'regions':len(a.REGIONS),'coco':len(v.COCO_CLASSES)})
    check('wire_and_feature_dimensions',a.WIRE_BYTES==369 and a.WIRE_BYTES*8==2952 and a.FEATURE_COUNT==374 and a.FEATURE_COUNT+6+6+1==387,
          4,'Wire block and feature dimensions, plus requested 6 values + 6 masks + cold-start arithmetic.',
          {'wire_bytes':a.WIRE_BYTES,'wire_bits':a.WIRE_BYTES*8,'features':a.FEATURE_COUNT,'extended_width':a.FEATURE_COUNT+13})

    producer='a1'*32
    rows=[{'label':'person','box':[-5,10,50,60],'score':.9},
          {'label':'person','box':[20,20,80,80],'score':.8},
          {'label':'road','box':[0,40,100,100],'score':.7},
          {'label':'building','box':[70,0,120,60],'score':.6}]
    scene,audit=a.summarize(rows,100,100,producer,max_detections=10)
    scene_bytes=scene.serialize()
    received=a.SceneArray.deserialize(scene_bytes,producer)
    check('scene_summary_masks_status',scene.detector_ok and all(scene.count_available[i] for i,x in enumerate(a.CLASSES) if x in a.OBJECTS)
          and all(not scene.count_available[i] and scene.counts[i]==0 for i,x in enumerate(a.CLASSES) if x in a.REGIONS)
          and all(scene.area_available) and scene.counts[a.class_slot('person')]==2 and scene.counts[a.class_slot('building')]==1
          and math.isclose(scene.largest_area[a.class_slot('person')],.36) and math.isclose(scene.largest_area[a.class_slot('road')],.6)
          and not scene.saturated,
          len(rows)+93,'Produced box examples summarize via actual summarize(); object counts apply, region count masks clear, region area remains available, clipped largest rectangles are correct.',
          {'audit_kept':len(audit['detections']),'counts_person_building':[scene.counts[a.class_slot('person')],scene.counts[a.class_slot('building')]],
           'areas_person_road':[scene.largest_area[a.class_slot('person')],scene.largest_area[a.class_slot('road')]],'saturated':scene.saturated})
    unavailable=a.unavailable(producer)
    check('unavailable_all_masks_status',not unavailable.detector_ok and not any(unavailable.count_available) and not any(unavailable.area_available)
          and not any(unavailable.counts) and not any(unavailable.largest_area), 4*93+1,
          'Unavailable extraction has failed status, canonical zero values and every count/area mask false.')

    max_error=max(abs(x-y) for x,y in zip(scene.largest_area,received.largest_area))
    check('area_quantization_error',max_error<=.5/255+1e-15,93,'Serialize/deserialize area error bounded by half an 8-bit quantization step.',{'max_abs_error':max_error,'bound':.5/255})
    check('exact_byte_roundtrip',received.serialize()==scene_bytes and a.SceneArray.from_record(received.record()).serialize()==scene_bytes,
          2,'Wire deserialize/serialize and JSON record roundtrip reproduce exact bytes.',{'length':len(scene_bytes),'sha256':hashlib.sha256(scene_bytes).hexdigest()})
    feats=received.features(producer)
    idx=a.class_slot('person')
    check('received_feature_order_values',len(feats)==374 and math.isclose(feats[idx],math.log1p(2)) and
          math.isclose(feats[93+idx],received.largest_area[idx]) and math.isclose(feats[93+idx],round(scene.largest_area[idx]*255)/255)
          and feats[2*93+idx]==1.0 and feats[3*93+idx]==1.0 and feats[-2:]==[1.0,0.0],
          7,'Feature vector uses received quantized area; count alone receives log1p once; masks/status order matches implementation.',
          {'length':len(feats),'person_log1p_count':feats[idx],'person_area_received':feats[93+idx],'tail':feats[-2:]})

    # Constructor and wire rejection paths. Confirm ValueError for each corrupt form.
    rejects=[]
    def expect_reject(name,fn):
        try: fn()
        except (ValueError,KeyError,struct.error): rejects.append(name); return
        raise AssertionError('expected rejection: '+name)
    expect_reject('wrong_length',lambda:a.SceneArray.deserialize(scene_bytes[:-1],producer))
    bad=bytearray(scene_bytes); bad[2]^=1
    expect_reject('wrong_registry_hash',lambda:a.SceneArray.deserialize(bytes(bad),producer))
    expect_reject('wrong_expected_producer',lambda:a.SceneArray.deserialize(scene_bytes,'b2'*32))
    bad=bytearray(scene_bytes); bad[34]^=1
    expect_reject('wrong_embedded_producer',lambda:a.SceneArray.deserialize(bytes(bad),producer))
    # 93 mask bits => 3 padding bits in last byte of each 12-byte mask.
    mask_bytes=a.MASK_BYTES
    cm_last=a.PREFIX.size+3*a.N_CLASSES+mask_bytes-1
    am_last=cm_last+mask_bytes
    for name,pos in [('count_mask_padding',cm_last),('area_mask_padding',am_last)]:
        bad=bytearray(scene_bytes); bad[pos]|=0x80
        expect_reject(name,lambda data=bytes(bad):a.SceneArray.deserialize(data,producer))
    check('invalid_contract_rejections',len(rejects)==6,6,'Rejects wrong byte length, registry hash, expected/embedded producer hashes, and padding bits in both masks.',{'rejected_cases':rejects,'count_padding_byte_offset':cm_last,'area_padding_byte_offset':am_last})

    # Preserve all previous 16 method IDs and require the temporal model to be the only added registry ID.
    old=BASE.read_text(encoding='utf-8-sig'); new=DOC.read_text(encoding='utf-8-sig')
    ids_re=re.compile(r'^\|\s*([a-z][a-z0-9_]+)\s*\|',re.M)
    old_ids=ids_re.findall(old); new_ids=ids_re.findall(new)
    added=sorted(set(new_ids)-set(old_ids)); missing=sorted(set(old_ids)-set(new_ids))
    check('model_registry_preservation',len(old_ids)==16 and not missing and added==['temporal_deepsets_ranknet'],
          len(old_ids)+len(new_ids),'All sixteen baseline IDs remain; one temporal registry ID is added.',{'baseline_ids':old_ids,'current_ids':new_ids,'added':added,'missing':missing})
    def equation_lines(s):
        return [line.strip() for line in s.splitlines() if line.strip().startswith(('ell(W,delta)', 'L_D(t) =', 'y_i = ell(', 'Reward scale100', 'L_E(T)=sum_f'))]
    old_eq,new_eq=equation_lines(old),equation_lines(new)
    check('objective_math_preserved',old_eq==new_eq and len(old_eq)>=5,len(old_eq)+len(new_eq),
          'Literal D loss/target/reward and E loss formula lines match preserved baseline.',{'baseline':old_eq,'current':new_eq})
    alloc=re.search(r'\*\*Fixed216-call maximum allocation:\*\*\s*\n\s*\n((?:\|[^\n]*\|\s*\n)+)',new)
    calls=[int(x) for x in re.findall(r'^\|[^|]*\|\s*(\d+)\s*\|\s*$',alloc.group(1),re.M)] if alloc else []
    check('teacher_call_allocation',sum(calls)==216 and len(calls)==6, len(calls),
          'The fixed teacher allocation contains six rows summing to216.',{'calls':calls,'sum':sum(calls)})

    # Clarification: two-way CE on logits [sA,sB], target [t,1-t], equals pair BCE.
    ce_cases=[]
    for d in (-3.0, 0.0, 2.5):
        m=max(d,0.0); log_denom=m+math.log(math.exp(d-m)+math.exp(-m))
        for t in (0.0,0.5,1.0):
            ce=log_denom-t*d
            pair=t*(max(0.0,-d)+math.log1p(math.exp(-abs(d))))+(1-t)*(max(0.0,d)+math.log1p(math.exp(-abs(d))))
            ce_cases.append({'score_difference':d,'target':t,'two_way_ce':ce,'pair_bce':pair,'abs_diff':abs(ce-pair)})
    max_ce_diff=max(row['abs_diff'] for row in ce_cases)
    check('two_way_ce_equals_pair_bce',max_ce_diff<1e-12,len(ce_cases),
          'Two-way CE with target [t,1-t] equals RankNet pair BCE for t=0,0.5,1 and unequal/equal finite score differences.',
          {'cases':ce_cases,'max_abs_difference':max_ce_diff})

    source_after={str(p):sha(p) for p in SOURCE_PATHS}
    doc_hash_after=sha(DOC)
    check('production_sources_unchanged',source_before==source_after,2,'Hashes of the two production source modules are identical before and after validation.',{'before':source_before,'after':source_after})
    # Document can be edited concurrently. Record both observations; only report stable hash if unchanged during run.
    doc_stable=doc_hash_before==doc_hash_after
    receipt={'title':'scene_arrays_v2 bounded local contract validation','classification':'existing pure parameter module contract checks; no detector inference, scheduler adapter implementation, training, cloud or download',
      'overall':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','check_count':len(checks),'case_count':sum(c['cases'] for c in checks),
      'runtime':{'python_executable':sys.executable,'python_version':sys.version,'platform':platform.platform(),'backend':'CPU; isolated importlib namespace; standard library plus existing repo venv'},
      'command':f"& '{sys.executable}' '{Path(__file__).resolve()}'",'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__).resolve()),
      'repo':str(REPO),'source_before_sha256':source_before,'source_after_sha256':source_after,
      'document_hash_before':doc_hash_before,'document_hash_after':doc_hash_after,'document_hash_stable_during_check':doc_stable,
      'checked_document_hash':doc_hash_after,'baseline_path':str(BASE),'checks':checks,
      'limitations':['No detector/model inference; supplied boxes exercise summarize only.','No scheduler adapter was built or validated.','Document hash is the snapshot observed during this run and may change if the authorized editorial pass continues.','Current pure source modules may validate their implemented contract without establishing compatibility with a downstream consumer.']}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'overall':receipt['overall'],'check_count':receipt['check_count'],'case_count':receipt['case_count'],'document_hash_stable_during_check':doc_stable,'script_sha256':receipt['script_sha256'],'receipt':str(OUT/'receipt.json')},indent=2))
except Exception as e:
    failure={'error':repr(e),'traceback':traceback.format_exc(),'check_results':checks}
    (OUT/'failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
    raise
