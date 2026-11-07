"""Pinned mini-pool data, independent silver references and causal feature inputs."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import numpy as np
from Evaluation.compact_scene_transport import CompactSceneTransport
from Evaluation.semantic_learning_env import LEVELS,PROFILES
from Scheduler.semantic_learning_policy import digest

ARCHIVE_SHA='0f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c'
CLIP_ID='openai/clip-vit-base-patch32'
CLIP_REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_pool(archive,output,vector_extension=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if file_hash(archive)!=ARCHIVE_SHA:raise ValueError('mini archive identity mismatch')
    with tarfile.open(archive) as tar:
        rows=[json.loads(x) for x in tar.extractfile('Mini_dataset/handoff/image_parameters.jsonl') if x.strip()]
        labels=json.load(tar.extractfile('Mini_dataset/student/training_records.json'))
        receipt=json.load(tar.extractfile('Mini_dataset/student/training_receipt.json'))
        registry=json.load(tar.extractfile('Mini_dataset/scene/registry.json'))
        encoder=json.load(tar.extractfile('Mini_dataset/global_embeddings/encoder.json'))
        image_index=json.load(tar.extractfile('Mini_dataset/portable_image_index.json'))
        sqlite_path=output/'cached_vectors.sqlite';sqlite_path.write_bytes(tar.extractfile('Mini_dataset/global_embeddings/vectors.sqlite').read())
    labels_by_hash={}
    for r in labels:
        if r['level'] in LEVELS:labels_by_hash.setdefault(r['image_sha256'],set()).add(LEVELS.index(r['level']))
    producer_ids={r['scene']['producer_sha256'] for r in rows}
    if len(rows)!=6000 or len(producer_ids)!=1:raise ValueError('unexpected source count/producer mixture')
    profile={'schema':'scene93_received387_v1','vocabulary_sha256':digest(registry),
       'producer_sha256':next(iter(producer_ids)),'feature_order':'scene374+SPI_prob4_C_H_M_L+R+N+availability6+cold1',
       'semantic_producer':encoder,'missions_sha256':digest(PROFILES),
       'spi_score_order':['Critical','High','Medium','Low'],'spi_score_calibration':'uncalibrated producer scores; not urgency probabilities',
       'spi_producers':sorted({r['spi'].get('student_bundle_sha256','unavailable') for r in rows}),
       'scalar_quantization':'R unsigned8[-1,1];N unsigned8[0,1];cold convention hidden',
       'sender_kb':'FIFO128; appearance cosine query then commit all observations'}
    profile['sha256']=digest(profile)
    codec=CompactSceneTransport(1,profile['vocabulary_sha256'],profile['producer_sha256'],[r['count_applicable'] for r in registry['classes']])
    features={};seen_ids=set();group_roles={};hash_roles={};prepared=[]
    for row in rows:
        ident=row['image_id'];raw=row['image_sha256'];wire=bytes.fromhex(row['scene_wire_hex'])
        if ident in seen_ids:raise ValueError('duplicate data identity')
        seen_ids.add(ident);compact=codec.encode(wire)
        if codec.decode(compact)!=wire:raise ValueError('compact received parity')
        count=np.frombuffer(wire,dtype='<u2',count=93,offset=66).astype(float)
        area=np.frombuffer(wire,dtype='u1',count=93,offset=252)/255.
        cm=int.from_bytes(wire[345:357],'little');am=int.from_bytes(wire[357:369],'little')
        scene=np.concatenate([np.log1p(count),area,[float(bool(cm&(1<<i))) for i in range(93)],
             [float(bool(am&(1<<i))) for i in range(93)],[float(bool(wire[1]&1)),float(bool(wire[1]&2))]])
        spi=row['spi'];available=bool(spi['available'])
        if available and spi['level'] not in LEVELS:raise ValueError('invalid SPI class')
        scores=np.array([float(available and spi['level']==z) for z in LEVELS])
        if available and 'scores' in spi:
            spi_scores=np.asarray(spi['scores'],dtype=float)
            if spi_scores.shape!=(4,) or not np.isfinite(spi_scores).all() or np.any(spi_scores<0) or abs(spi_scores.sum()-1)>1e-4:
                raise ValueError('invalid SPI probability vector')
            # The registered mini student encodes Low, Medium, High, Critical.
            scores=spi_scores[::-1]
        x=np.concatenate([scene,scores,[0.,0.],[float(available)]*4,[0.,0.,0.]])
        if x.shape!=(387,) or not np.isfinite(x).all():raise ValueError('input dimension/finite check failed')
        features[ident]=x.astype(np.float32)
        targets=labels_by_hash.get(raw,set());tier=next(iter(targets)) if len(targets)==1 else -1
        r={k:row[k] for k in ('image_id','image_sha256','dataset','source_id','group','split')}
        r.update(reference_tier=tier,reference_scope='mixed-protocol image-level VLM silver; not independent urgency',
            _profile_id=profile['sha256'],_profile_identity=profile,_compact_scene=compact)
        prepared.append(r);group_roles.setdefault(r['group'],set()).add(r['split']);hash_roles.setdefault(raw,set()).add(r['split'])
    if any(len(v)>1 for v in list(group_roles.values())+list(hash_roles.values())):raise ValueError('split leakage')
    vectors=read_vectors(sqlite_path)
    if vector_extension is not None:
        extension=Path(vector_extension);binding=json.loads((extension.parent/'identity.json').read_text())
        if binding['archive_sha256']!=ARCHIVE_SHA or binding['encoder']!=encoder or binding['vectors_sha256']!=file_hash(extension):raise ValueError('embedding extension identity mismatch')
        with np.load(extension,allow_pickle=False) as extra:
            for key in extra.files:
                if key in vectors:raise ValueError('extension attempts to replace cached embedding')
                v=extra[key]
                if v.shape!=(512,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-10:raise ValueError('invalid extension embedding')
                vectors[key]=v/np.linalg.norm(v)
    report={'images':len(rows),'raw_unique':len(hash_roles),'cached_embeddings':len(vectors),'reference_known':sum(r['reference_tier']>=0 for r in prepared),
        'reference_conflicts':sum(len(v)>1 for v in labels_by_hash.values()),'reference_scope':receipt['labels'],
        'source_counts':{d:sum(r['dataset']==d for r in prepared) for d in sorted({r['dataset'] for r in prepared})},
        'group_split_counts':{role:sum(r['split']==role for r in prepared) for role in ('train','validation','test')},'input_profile':profile}
    (output/'data_audit.json').write_text(json.dumps(report,indent=2));(output/'encoder.json').write_text(json.dumps(encoder,indent=2))
    (output/'image_index.json').write_text(json.dumps(image_index));return prepared,features,vectors,profile,report


def read_vectors(path):
    # Read the versioned JSON payload schema, never unpickle database values.
    db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
    info=db.execute('PRAGMA table_info(vectors)').fetchall()
    cols=[r[1] for r in info]
    if not cols:raise ValueError('expected vectors table')
    rows=db.execute('SELECT * FROM vectors').fetchall();db.close();result={}
    for row in rows:
        values=dict(zip(cols,row));key=values.get('hash')
        payload=values.get('value')
        if key is None or not isinstance(payload,bytes):raise ValueError('incompatible embedding SQLite schema')
        v=np.load(io.BytesIO(payload),allow_pickle=False).astype(np.float32)
        if v.shape!=(512,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-10:raise ValueError('invalid cached visual vector')
        result[str(key)]=v/np.linalg.norm(v)
    return result


def mission_vectors(output):
    import torch
    from transformers import CLIPModel,AutoTokenizer
    from huggingface_hub import snapshot_download
    path=snapshot_download(CLIP_ID,revision=CLIP_REV,allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json','preprocessor_config.json'])
    identity=json.loads((Path(output)/'encoder.json').read_text())
    for name,expected in identity['weights_sha256'].items():
        if file_hash(Path(path)/name)!=expected:raise ValueError('CLIP checkpoint hash differs')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=CLIPModel.from_pretrained(path,local_files_only=True,use_safetensors=False).eval()
    with torch.no_grad():
        v=model.get_text_features(**tokenizer([p[1] for p in PROFILES],padding=True,truncation=True,return_tensors='pt'))
        if not isinstance(v,torch.Tensor):v=v.pooler_output
        v=v/v.norm(dim=-1,keepdim=True)
    result=v.cpu().numpy().astype(np.float32)
    np.save(Path(output)/'mission_vectors.npy',result)
    (Path(output)/'mission_vector_identity.json').write_text(json.dumps({'repo':CLIP_ID,'revision':CLIP_REV,'mission_texts':PROFILES,'shape':list(result.shape),'image_encoder_forward_calls':0},indent=2))
    return result


def split_records(rows,role,require_reference=False):
    seen=set();out=[]
    for r in rows:
        if r['split']!=role or require_reference and r['reference_tier']<0:continue
        if r['image_sha256'] in seen:continue
        seen.add(r['image_sha256']);out.append(r)
    return out


def load_mission_vectors(path):
    """Reuse a version-bound producer cache; do not reload CLIP for scheduler tests."""
    path=Path(path)
    identity=json.loads((path.parent/"mission_vector_identity.json").read_text())
    if identity.get("repo")!=CLIP_ID or identity.get("revision")!=CLIP_REV or identity.get("mission_texts")!=[list(row) for row in PROFILES]:
        raise ValueError("mission vector producer/text identity mismatch")
    vectors=np.load(path,allow_pickle=False)
    if vectors.shape!=(len(PROFILES),512) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-5):
        raise ValueError("finite normalized mission vectors required")
    return vectors


def development_cohorts(rows,require_reference=True):
    """One grouped role assignment shared by every training driver and teacher."""
    eligible=[];seen=set()
    for r in rows:
        if r["split"]!="train" or require_reference and r["reference_tier"]<0:continue
        raw=r.get("image_sha256",r["image_id"])
        if raw in seen:continue
        seen.add(raw);eligible.append(r)
    held={r["group"] for r in eligible if int(hashlib.sha256(r["group"].encode()).hexdigest()[:8],16)%5==0}
    return ([r for r in eligible if r["group"] not in held],[r for r in eligible if r["group"] in held])

def cohort_coverage(rows):
    return {source:{"offered":sum(r["dataset"]==source for r in rows),
                    "assessable":sum(r["dataset"]==source and r["reference_tier"]>=0 for r in rows)}
            for source in sorted({r["dataset"] for r in rows})}
