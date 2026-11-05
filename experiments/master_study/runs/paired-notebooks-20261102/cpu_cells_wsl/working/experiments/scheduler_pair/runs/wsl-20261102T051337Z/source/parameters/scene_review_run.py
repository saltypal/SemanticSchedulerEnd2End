"""Pinned-cache v2 conversion and bounded Qwen offline review; no encoder work."""
from pathlib import Path
from functools import partial
import argparse
import hashlib
import json
import os
import time
import traceback
from .artifacts import Run, write_json
from .contracts import canonical, digest
from .review_only import copy_review_inputs
from .scene_json import build_scene_document, restore_bundle, validate_document
from .scene_review import make_cases, reviewer_payload, parse_result, score_cases, PROTOCOL, POLICY
from .extractor import VisualEvidence
from .ue_profiles import load_ue_profiles


def prepare(source, root, specification):
    source, root = Path(source), Path(root)
    if specification.get('operator_policy') != POLICY:
        raise ValueError('operator policy differs from frozen engineering contract')
    copy_review_inputs(source, root, specification['upstream_identity'])
    checks = json.loads((source/'checksums.json').read_text())
    feature_file = source/'extraction/features.jsonl'
    if hashlib.sha256(feature_file.read_bytes()).hexdigest() != checks['extraction/features.jsonl']:
        raise ValueError('feature cache checksum differs')
    features = {row['image_id']: VisualEvidence.from_dict(row['evidence'])
                for row in map(json.loads, feature_file.read_text().splitlines())}
    selected = set(json.loads((root/'data/review_cohort.json').read_text())['image_ids'])
    trace = json.loads((root/'extraction/trace.json').read_text())
    missions, _ = load_ue_profiles()
    ontology = specification['ontology']
    documents = []; seen = set()
    for row in trace:
        identity = row['image_id']
        if identity not in selected or identity in seen: continue
        seen.add(identity); bundle = restore_bundle(row['bundle'])
        document, audit = build_scene_document(bundle, features[identity], missions[bundle.mission_id],
            source_id=row['sender'], image_id=identity, ontology=ontology)
        validate_document(document)
        write_json(root/'documents'/(identity+'.json'), document)
        write_json(root/'audit'/(identity+'.json'), audit)
        documents.append(document)
    if seen != selected or len(documents) != specification['parent_images']:
        raise ValueError('review cohort differs from frozen specification')
    records = json.loads((root/'data/records.json').read_text())
    paths = {r['image_id']: r['image_path'] for r in records}
    cases = make_cases(documents, records, paths)
    if len(cases) != specification['planned_calls']: raise ValueError('case count differs from specification')
    write_json(root/'data/v2_cases_evaluator_only.json', cases)
    write_json(root/'data/operator_policy.json', POLICY)
    write_json(root/'data/prepare_receipt.json', dict(documents=len(documents), planned_calls=len(cases),
        protocol=PROTOCOL, model_inference_run=False, feature_sha256=checks['extraction/features.jsonl'],
        claims_source='runtime bundles and features only; annotations enter evaluator controls afterward',
        parents_previously_exposed=True, real_world_urgency_labels_available=False))
    return cases


def run(config, source, output):
    import torch
    import transformers
    from .pilot_producers import snapshot_model
    from .qwen35_reviewer import Qwen35AnnotationReviewer
    from .model_settings import ReviewerSettings
    specification = json.loads(Path(config).read_text())
    limits = specification['limits']
    if transformers.__version__ != specification['model']['transformers']:
        raise RuntimeError('reviewer runtime differs from frozen specification')
    if torch.cuda.device_count() != 2: raise RuntimeError('two assigned GPUs required; GPU1 reviewer')
    experiment = Run(output, {'seconds': limits['internal_seconds'], 'specification': specification})
    reviews = []; reviewer = None; cases = []
    try:
        with experiment.stage('00_reuse_and_convert_v2'):
            cases = prepare(source, experiment.path, specification)
            write_json(experiment.path/'hardware.json', dict(gpus=[torch.cuda.get_device_name(i) for i in (0,1)],
                torch=torch.__version__, cuda=torch.version.cuda, reviewer_device='cuda:1', gpu0_inference=False))
        with experiment.stage('01_qwen_v2_review'):
            started = time.monotonic()
            checkpoint, revision = snapshot_model(specification['model']['model_id'],
                                                   revision=specification['model']['revision'])
            reviewer = Qwen35AnnotationReviewer(checkpoint, revision,
                settings=ReviewerSettings(**specification['review_settings']))
            reviewer.maximum_input_tokens = 8192; reviewer.maximum_generation_seconds = 90
            reviewer.require_finite_preflight = True
            reviewer.load()
            from .scene_review import AUDIT_PROMPT, QA_PROMPT
            reviewer.provenance['route_prompt_sha256'] = dict(audit=digest(AUDIT_PROMPT), qa=digest(QA_PROMPT))
            write_json(experiment.path/'model.json', reviewer.provenance)
            # Largest actual JSON, then each information route, then all others.
            actual = [c for c in cases if c['route']=='audit' and c['arm']=='actual']
            initial = [max(actual, key=lambda c: len(canonical(c['document'])))]
            initial += [next(c for c in cases if c['route']==route) for route in ('json_only','image_only')]
            order = initial+[c for c in cases if c['id'] not in {x['id'] for x in initial}]
            write_json(experiment.path/'review_order.json', [c['id'] for c in order])
            state = 'complete'; reason = None
            with (experiment.path/'reviews.jsonl').open('w', encoding='utf-8') as handle:
                for index, case in enumerate(order):
                    remaining = limits['internal_seconds']-(time.monotonic()-experiment.started)
                    if remaining < 100:
                        state='partial'; reason='bounded remaining time'; break
                    payload, prompt = reviewer_payload(case)
                    images = [] if case['route']=='json_only' else [case['image_path']]
                    row = reviewer.review_structured(images, payload, prompt,
                        partial(parse_result, route=case['route'], requested=case['requested']),
                        case['id'], experiment.path/'raw', protocol=PROTOCOL+'/'+case['route'],
                        text_only_diagnostic=case['route']=='json_only')
                    row['route'] = case['route']; row['assessment_kind'] = PROTOCOL+'/'+case['route']
                    reviews.append(row); handle.write(canonical(row)+'\n'); handle.flush()
                    write_json(experiment.path/'metrics.json', score_cases(cases, reviews))
                    write_json(experiment.path/'review_progress.json', dict(attempted=len(reviews), planned=len(cases)))
                    print('V2 REVIEW',len(reviews),'/',len(cases),case['route'],row['status'],flush=True)
                    if row['status']=='model_failed':
                        state='failed';reason='model failure; no automatic retry';break
                    if index==2:
                        mean=sum(r['wall_seconds'] for r in reviews)/3
                        projected=mean*(len(cases)-3)*1.2
                        left=limits['internal_seconds']-(time.monotonic()-experiment.started)
                        valid=sum(r['status']=='complete' for r in reviews)
                        write_json(experiment.path/'technical_preflight.json',dict(valid=valid,
                            projected_remaining_seconds=projected,remaining_seconds=left,
                            worker_setup_seconds=time.monotonic()-started-sum(r['wall_seconds'] for r in reviews)))
                        if valid<2 or projected>left:
                            state='partial';reason='three-route format/runtime gate';break
            write_json(experiment.path/'status.json',dict(status=state,reason=reason,attempted=len(reviews),
                planned=len(cases),seconds=time.monotonic()-experiment.started,enable_thinking=False,
                whole_scene_depth_verified=False,real_world_urgency_verified=False,no_policy_training=True))
    except Exception as error:
        traceback.print_exc()
        write_json(experiment.path/'status.json',dict(status='failed',error=type(error).__name__,
            message=str(error),attempted=len(reviews),seconds=time.monotonic()-experiment.started))
    finally:
        if reviewer is not None: reviewer.close()
        if cases: write_json(experiment.path/'metrics.json',score_cases(cases,reviews))
        write_json(experiment.path/'checksums.json', {
            str(p.relative_to(experiment.path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in experiment.path.rglob('*') if p.is_file() and p.name!='checksums.json'})
    return experiment.path
