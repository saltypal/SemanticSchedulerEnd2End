"""Shared notebook9 dataset boundaries. No model loading or downloads on import."""
import copy
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .artifacts import write_json
from .contracts import digest
from .extractor import SenderMemory,embedding_signature
from .minimal_parameters import FinalParameterExtractor
from .pilot_parameters import context_similarity_key
from .readiness import require_spi_ready
from .spi_direct_priority import assess_direct,PROMPT


def discover_input_root(base, expected_manifest_sha256):
    """Resolve one exact input snapshot across Kaggle mount layouts.

    A matching slug alone is insufficient: pin the manifest bytes, and reject
    ambiguity rather than silently mixing dataset versions.
    """
    if not re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha256 or ''):
        raise ValueError('immutable input manifest SHA-256 required')
    base=Path(base).resolve()
    if not base.is_dir():raise FileNotFoundError('input mount directory missing: '+str(base))
    matches=[];inspected=[]
    for manifest in sorted(base.rglob('input_manifest.json')):
        if not manifest.resolve().is_relative_to(base):continue
        actual=hashlib.sha256(manifest.read_bytes()).hexdigest()
        inspected.append({'path':str(manifest),'sha256':actual})
        if actual==expected_manifest_sha256:
            folder=manifest.parent
            if not all((folder/name).is_file() for name in ('panel.json','features.npz')):
                raise ValueError('matching manifest has incomplete input files: '+str(folder))
            matches.append(folder)
    print('Input mount candidates',inspected,flush=True)
    if len(matches)!=1:
        raise ValueError('expected exactly one matching input snapshot; found '+str(len(matches)))
    return matches[0]


def materialize_current_300(input_root,output):
    """Read the existing immutable input bundle; no source downloads or resplits."""
    from .spi_campaign import verify_inputs
    input_root=Path(input_root)
    rows,receipt=verify_inputs(input_root)
    if len(rows)!=300:raise ValueError('exact current300 cohort required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if receipt['layout']=='zip':
        with zipfile.ZipFile(input_root/'images.zip') as archive:
            if any(not (output/name).resolve().is_relative_to(output.resolve()) for name in archive.namelist()):
                raise ValueError('unsafe image archive path')
            archive.extractall(output)
        folder=output
    else:folder=input_root/'images'
    records=[{**r,'image_path':str((folder/r['image_path']).resolve())} for r in rows]
    validate_records(records)
    write_json(output/'records.json',records)
    return records,input_root/'features.npz'


def validate_records(records):
    required={'image_id','image_path','image_sha256','group','split','source'}
    ids=set();groups={};contents={}
    for row in records:
        if not required<=set(row) or row['split'] not in ('train','validation','test'):
            raise ValueError('explicit image identity, provenance and grouped split required')
        if not isinstance(row['image_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',row['image_id']):
            raise ValueError('image ID must be a safe unique filename token')
        if row['image_id'] in ids:raise ValueError('duplicate image ID')
        ids.add(row['image_id'])
        for table,key in ((groups,row['group']),(contents,row['image_sha256'])):
            if key in table and table[key]!=row['split']:raise ValueError('group/content crosses splits')
            table[key]=row['split']
        if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError('image hash mismatch')
    if not records:raise ValueError('empty dataset')
    return records


def extract_runtime_features(records, provider, mission, output, batch_size=16):
    """Re-extract an immutable image cohort in this runtime's feature space.

    This creates new features and provenance; it never relabels an old cache's
    signature. Native labels and split targets are not model inputs.
    """
    validate_records(records)
    if batch_size<1:raise ValueError('positive batch size required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    vectors=[];signatures=set();provenance=[]
    for start in range(0,len(records),batch_size):
        batch=records[start:start+batch_size];images=[]
        for row in batch:
            with Image.open(row['image_path']) as im:images.append(im.copy())
        evidence=provider.extract_batch(images,[mission]*len(images))
        if len(evidence)!=len(batch):raise ValueError('feature batch ownership mismatch')
        for item in evidence:
            vector=np.asarray(item.embedding,dtype=np.float32)
            if vector.ndim!=1 or not np.isfinite(vector).all() or np.linalg.norm(vector)<=0:
                raise ValueError('invalid runtime image embedding')
            vectors.append(vector);signatures.add(embedding_signature(item));provenance.append(item.provenance)
        print('Runtime image features',len(vectors),'/',len(records),flush=True)
    if len(signatures)!=1:raise ValueError('mixed runtime feature spaces')
    destination=output/'features.npz'
    with (output/'features.partial').open('wb') as handle:
        np.savez_compressed(handle,embeddings=np.stack(vectors),
            image_ids=np.array([r['image_id'] for r in records]),
            image_sha256=np.array([r['image_sha256'] for r in records]),
            feature_signature=np.array(next(iter(signatures))))
    (output/'features.partial').replace(destination)
    write_json(output/'provenance.json',{'feature_signature':next(iter(signatures)),
        'image_count':len(records),'producer':provenance[0],
        'features_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'reason':'fresh runtime extraction; original source cache remains unchanged'})
    return destination


def label_dataset(records,reviewer,selection_receipt,output):
    """Image-only pseudo-labels. Post text/native truth never enter this call."""
    validate_records(records)
    if selection_receipt.get('accepted') is not True:raise ValueError('teacher selection not accepted')
    if selection_receipt.get('prompt_sha256')!=hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError('selected teacher prompt differs')
    reviewer.load()
    for key in ('model_id','revision'):
        if selection_receipt.get(key)!=reviewer.provenance.get(key):raise ValueError('selected teacher identity differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);labels=[]
    try:
        for row in records:
            if row['split']=='test':continue  # Keep final labels out of development selection.
            result=assess_direct(reviewer,[row['image_path']],{'instruction':'Label the urgency of the visible scene.'},
                                 row['image_id'],output/'raw')
            label={k:row[k] for k in ('image_id','image_sha256','group','split','source')}
            label.update(result,label_source='vlm_pseudo',label_unit='whole_image',rubric_sha256=digest(PROMPT))
            label.update(teacher_model_id=selection_receipt['model_id'],teacher_revision=selection_receipt['revision'])
            labels.append(label);write_json(output/'labels.json',labels)
            print('Labels',len(labels),'/',len(records),result['level'],flush=True)
    finally:
        write_json(output/'status.json',{'attempted':len(labels),'teacher':reviewer.provenance,
            'selection_receipt':selection_receipt,'scope':'image-only teacher agreement; no independent urgency validation'})
    return labels


def load_selected_teacher(receipt):
    """Validate selection before downloads and bind the exact accepted revision."""
    if receipt.get('accepted') is not True or receipt.get('prompt_sha256')!=hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError('accepted matching-prompt selection required')
    revision=receipt.get('revision','')
    if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):
        raise ValueError('immutable teacher commit SHA required')
    model_id=receipt.get('model_id')
    if model_id=='google/gemma-4-E4B-it':
        from .gemma_spi_reviewer import load_gemma
        return load_gemma({'revision':revision})
    from .spi_teachers import TEACHERS,load_teacher
    for spec in TEACHERS[:2]:
        if model_id==spec['model']:return load_teacher({**spec,'revision':revision})
    from .native_vlm_candidates import InternVL3Reviewer,Qwen25Reviewer
    for cls in (InternVL3Reviewer,Qwen25Reviewer):
        if model_id==cls.model_id:
            reviewer=cls();reviewer.revision=revision
            try:reviewer.load()
            except Exception:
                reviewer.close();raise
            return reviewer
    raise ValueError('unrecognized selected teacher model')


def _clock(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)
    return time.perf_counter()


def cache_parameters(records,profiles,provider,scene_provider,model_path,readiness_path,output,*,synthetic=False):
    """Readiness first; image shards once, six mission/history streams separately.

    Existing output is never overwritten. Explicit shards/checksums preserve
    partial evidence; incompatible or partial runs require a new output path.
    """
    classifier=require_spi_ready(model_path,readiness_path,allow_fixture=synthetic)  # Before any model extraction/output.
    validate_records(records)
    if not profiles or len({ue for ue,mission in profiles})!=len(profiles):raise ValueError('unique UE profiles required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    device=getattr(getattr(provider,'provider',provider),'device','cpu')
    identity={'records':[{k:r[k] for k in ('image_id','image_sha256','group','split','source')} for r in records],
        'synthetic':synthetic,
        'profiles':[(ue,m.version) for ue,m in profiles],
        'model_sha256':classifier.artifact['sha256'],
        'readiness_sha256':hashlib.sha256(Path(readiness_path).read_bytes()).hexdigest()}
    write_json(output/'manifest.json',identity)
    extractor=FinalParameterExtractor(spi_classifier=classifier)
    memories={};statics=[];streams=[];embeddings=[];timings=[];completed=False;persisted=0
    try:
        for row in records:
            begin=_clock(device)
            with Image.open(row['image_path']) as raw:image=raw.copy()
            loaded=_clock(device)
            evidence=provider.extract_batch([image],[profiles[0][1]])[0]
            visual=_clock(device)
            scene=scene_provider.measure(image)
            detected=_clock(device)
            spi=classifier.predict(evidence.embedding,embedding_signature(evidence))
            predicted=_clock(device)
            static={'image_id':row['image_id'],'image_sha256':row['image_sha256'],
                'feature_signature':embedding_signature(evidence),'spi':spi,'scene':scene.values(),
                'scene_status':scene.status,'scene_reason':scene.reason,'scene_audit':scene.audit}
            statics.append(static);embeddings.append(np.asarray(evidence.embedding,np.float32))
            image_streams=[]
            for numeric,(ue,mission) in enumerate(profiles,1):
                key=(row['split'],ue,mission.version)
                memory=memories.setdefault(key,SenderMemory(ue,mission.mission_id,32))
                local=copy.deepcopy(evidence);start=_clock(device)
                try:
                    vector=provider.context_embedding(mission)
                    local.similarities={context_similarity_key(mission):float(np.asarray(local.embedding)@vector)}
                except (ValueError,FloatingPointError) as error:
                    local.similarities={}
                    local.provenance.update(context_status='unavailable',context_reason=str(error))
                relevant=_clock(device)
                metadata={'source_id':ue,'source_numeric_id':numeric,'session_id':1,'message_id':len(statics),
                    'profile_id':numeric,'sequence':memory.last_sequence+1,
                    'capture_time_ms':row.get('capture_time_ms'),'timestamp_trusted':row.get('timestamp_trusted') is True}
                before=memory.version
                bundle,wire=extractor.serialize_and_commit(local,mission,metadata,memory,scene)
                ended=_clock(device)
                streams.append({'image_id':row['image_id'],'split':row['split'],'ue':ue,'mission_version':mission.version,
                    'sequence':metadata['sequence'],'kb_version_before':before,'kb_version_after':memory.version,
                    'parameters':bundle['compact'],'header_hex':wire.hex(),'header_bytes':len(wire),
                    'relevance_and_text_cache_seconds':relevant-start,
                    'bundle_head_kb_serialization_commit_seconds':ended-relevant})
                image_streams.append(streams[-1])
            timing={'image_id':row['image_id'],'image_load_seconds':loaded-begin,
                'visual_extract_including_preprocessing_first_mission_seconds':visual-loaded,
                'detector_and_measurement_seconds':detected-visual,'spi_head_seconds':predicted-detected,
                'end_to_end_seconds':_clock(device)-begin,'flops':None,
                'flops_status':'not_instrumented; not zero','cold_or_first_call':len(statics)==1}
            timings.append(timing)
            shard={'static':static,'timing':timing,'streams':image_streams}
            write_json(output/'shards'/f'{row["image_id"]}.json',shard)
            persisted+=1
            print('Cache',len(statics),'/',len(records),flush=True)
        signature=statics[0]['feature_signature']
        if any(r['feature_signature']!=signature for r in statics):raise ValueError('mixed feature spaces')
        np.savez_compressed(output/'features.npz',image_ids=np.array([r['image_id'] for r in records]),
            image_sha256=np.array([r['image_sha256'] for r in records]),
            embeddings=np.stack(embeddings),feature_signature=np.array(signature))
        write_json(output/'streams.json',streams);write_json(output/'timings.json',timings)
        completed=True
    finally:
        write_json(output/'status.json',{'status':'complete' if completed else 'incomplete',
            'synthetic':synthetic,
            'completed_images':persisted,'extracted_images':len(statics),'offered_images':len(records),
            'native_labels_in_runtime':False,'timing_scope':'measured calls; no claimed warm latency or complete FLOPs'})
        files=[p for p in output.rglob('*') if p.is_file() and p.name!='checksums.json']
        write_json(output/'checksums.json',{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return output


def export_spi_training(features_path,labels_path,output):
    """Package verified identities for existing SPI trainers; do not fit a model."""
    from .spi import prepare
    x,ids,indices,targets,identity=prepare(features_path,labels_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(features_path,output/'features.npz')
    shutil.copy2(labels_path,output/'labels.json')
    write_json(output/'training_manifest.json',{'schema':'spi_training_bundle_v1',**identity,
        'rows':len(ids),'dimension':x.shape[1],'roles':{s:len(v) for s,v in indices.items()},
        'native_labels_are_runtime_inputs':False,'training_performed':False})
    return output


def train_spi_for_cache(features_path,labels_path,selection_receipt,output,current_manifest,*,root=None,seconds=600):
    """Fit four outputs, validate/export and create a run-local cache receipt.

    Teacher acceptance is a user decision; student quality checks remain real.
    The installed historical release pointer is never changed by this function.
    """
    from .spi_model_selection import select,support_gate
    from .spi_release import candidate_release
    if selection_receipt.get('accepted') is not True:raise ValueError('final labeling teacher not accepted')
    if selection_receipt.get('prompt_sha256')!=hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError('teacher prompt differs from labels protocol')
    labels=json.loads(Path(labels_path).read_text())
    for row in labels:
        if row['split']=='test':continue
        if (row.get('teacher_model_id'),row.get('teacher_revision'))!=(selection_receipt.get('model_id'),selection_receipt.get('revision')):
            raise ValueError('training labels do not match final teacher identity')
    output=Path(output)
    report=select(features_path,labels_path,output,seconds=seconds)
    model=output/'models/spi.json'
    receipt=candidate_release(model,output/'metrics/validation.json',current_manifest,
                              output/'release_candidate.json',root=root)
    support=support_gate(labels)
    receipt.update(class_support=support['details'],teacher_selection=selection_receipt,
        reference_type='teacher_agreement',independent_urgency_validated=False,
        diagnostic_review={'accepted':False,'unresolved_decisive_contradictions':None,
            'reason':'Teacher chosen explicitly by user; known limitations retained, not independently certified.'})
    write_json(output/'cache_readiness.json',receipt)
    return {'model_path':str(model),'readiness_path':str(output/'cache_readiness.json'),
            'cache_allowed':receipt['runtime_promotion_allowed'],'validation':report['student']}
