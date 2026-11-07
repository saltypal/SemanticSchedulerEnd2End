"""Actual cached-latent replay through the Scene JSON completion controller.

No annotation-derived fields are manufactured here. Source v2 documents must
be supplied separately and joined by image hash to common received payloads.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time
from Evaluation.decoder_service import DecoderService
from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.scene_queue import SceneQueue,context_builder,validate_scene
from Scheduler.training.scene_prepare import write_json
from Scheduler.scene_policy import canonical,digest


def replay_scene_codec(adapter,received,documents,registry,policy,cache_root,output,
                       checkpoint_sha,device='cpu',decode_function=None,horizon=None):
    from Codec.integration.decoder_payload_cache import decode_cached
    decode_function=decode_function or decode_cached
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=received.get('records',[])
    if not received.get('complete') or not records:raise ValueError('complete received cache required')
    if len({r['image_id'] for r in records})!=len(records):raise ValueError('duplicate received images')
    docs={d['observation']['image_id']:validate_scene(d) for d in documents}
    if len(docs)!=len(documents) or set(docs)!={r['image_id'] for r in records}:
        raise ValueError('exactly one independently produced v2 document per received image required')
    missions={(d['mission']['id'],d['mission']['version']) for d in docs.values()}
    if len(missions)!=1:raise ValueError('one mission per replay required')
    specifications={d['provenance']['extraction_specification']:digest(d['provenance']['producers']) for d in docs.values()}
    queue=SceneQueue(policy,registry,next(iter(missions)),specifications)
    if policy.artifact is not None and horizon is None:raise ValueError('learned codec replay requires an explicit episode horizon')
    service=DecoderService(policy,queue=queue,context_builder=context_builder(registry,horizon or 1e9))
    for index,record in enumerate(records):
        document=docs[record['image_id']]
        service.enqueue(ReceivedMessage(document['observation']['source_id'],'scene-codec',index,
            canonical(document).encode(),record,True,0.,index))
    clock=0.;trace=[]
    while len(service):
        begin=time.perf_counter();job=service.dispatch(clock);selection=time.perf_counter()-begin
        record=job.message.payload_ref;begin=time.perf_counter()
        try:
            result=decode_function(adapter,record,cache_root,
                output/(hashlib.sha256(record['image_id'].encode()).hexdigest()+'.png'),checkpoint_sha,device)
            duration=float(result['decode_seconds'])
            if duration<0 or not math.isfinite(duration):raise ValueError('invalid measured decode time')
            outcome='succeeded';error=None
        except Exception as exception:
            duration=time.perf_counter()-begin;outcome='failed';result=None
            error={'type':type(exception).__name__,'message':str(exception)}
        clock+=selection+duration
        completion=service.complete(job.message.message_id,outcome,clock)
        trace.append({'image_id':record['image_id'],'decision':asdict(job.decision),
            'completion':asdict(completion),'selection_seconds':selection,'decode':result,'error':error,
            'header_bytes':len(job.message.header_bytes),'payload_sha256':record['payload_sha256']})
        write_json(output/'trace.json',trace)
    summary={'status':'complete','trace':trace,'succeeded':sum(r['error'] is None for r in trace),
        'source_payloads':{r['image_id']:r['payload_sha256'] for r in records},
        'semantic_preservation':'not independently assessed',
        'scope':'actual codec only if decode.execution_scope is measured_checkpoint_inference',
        'timeline':'measured selection plus decode; successful artifact I/O excluded'}
    write_json(output/'summary.json',summary)
    return summary


def main():
    import argparse
    from Codec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
    from Codec.integration.decoder_payload_cache import cache_received,sha256_file
    from Scheduler.training.scene_corpus import default_registry
    from Scheduler.scene_policy import ScenePolicy
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('documents','images','model-dir','upstream-root','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--limit',type=int,choices=(1,5),default=1)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    docs={}
    for path in sorted(args.documents.glob('*.json')):
        document=validate_scene(json.loads(path.read_text()))
        docs.setdefault(document['observation']['image_id'],document)
    chosen=list(docs.values())[:args.limit]
    rows=[]
    for document in chosen:
        identity=document['observation']['image_id'];matches=list(args.images.glob(identity+'.*'))
        if len(matches)!=1:raise ValueError('one matching source image required: '+identity)
        if sha256_file(matches[0])!=identity:raise ValueError('source/document image hash mismatch')
        rows.append({'image_id':identity,'image_path':str(matches[0])})
    begin=time.perf_counter()
    adapter=load_stage1_swinjscc(args.model_dir,device=args.device,upstream_root=args.upstream_root)
    for module in adapter._require_built():module.eval()
    checkpoint=sha256_file(args.model_dir/'stage1_swinjscc_full.pt')
    received=cache_received(adapter,rows,args.output/'received',checkpoint,device=args.device)
    results={}
    for name in ('fifo','random'):
        results[name]=replay_scene_codec(adapter,received,chosen,default_registry(),ScenePolicy(kind=name,seed=1729),
            args.output/'received',args.output/name,checkpoint,args.device)
    if results['fifo']['source_payloads']!=results['random']['source_payloads']:
        raise RuntimeError('policy-independent received payload check failed')
    write_json(args.output/'receipt.json',{'status':'complete','count':len(rows),'seconds':time.perf_counter()-begin,
        'common_received_payload_hashes':True,'checkpoint_sha256':checkpoint,'results':results,
        'scope':'development compatibility images; real decoding, no independent semantic-preservation assessment'})


if __name__=='__main__':main()
