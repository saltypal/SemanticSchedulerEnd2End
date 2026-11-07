"""Local bounded pilot and single/batch parameter extraction entrypoints."""
import argparse
import json
import os
from pathlib import Path
import time

from .artifacts import Run,write_json
from .contracts import MISSIONS,MethodSpecification,digest,flatten_bundle
from .data import balanced_ladi_records,medic_records,DownloadBudget,add_duplicate_groups
from .extractor import SemanticParameterExtractor,SenderMemory,VisualEvidence
from .serialization import HeaderCodec


def _original():return Path(__file__).resolve().parents[2]/'SemanticSchedulerEnd2End'


def historical_predictions(records,cache):
    """Optional old producer outputs; never use native labels as predictions."""
    import math
    if not Path(cache).is_file():return
    wanted={r['image_id']:r for r in records}
    with Path(cache).open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);identity=row.get('image_id')
            if identity not in wanted:continue
            evidence=row.get('evidence',row);logits=evidence.get('logits',{})
            def probability(name):
                value=logits.get(name)
                return None if value is None else 1/(1+math.exp(-max(-700,min(700,float(value)))))
            damaged=probability('buildings_affected_or_greater');minor=probability('buildings_minor_or_greater');road=probability('roads_damage')
            scores={}
            if damaged is not None and minor is not None:scores['building_damage']=damaged/3+2*minor/3
            if road is not None:scores['road_damage']=road
            wanted[identity]['historical_i']=scores


def pilot(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1)
    from PIL import Image
    import psutil
    from .vision import FrozenVisionProvider
    from .study import local_study
    run=Run(args.output,vars(args));records=[];cache={};profiles={};errors={}
    try:
        with run.stage('01_data_audit'):
            records,exclusions=balanced_ladi_records(args.ladi_manifest,limit=min(100,args.max_images))
            write_json(run.path/'data/ladi_audit.json',{'records':len(records),'exclusions':exclusions,'scope':'existing exposed local subset'})
            print('LADI native records',len(records),flush=True)
            if args.medic and len(records)<args.max_images:
                try:
                    more,failures,source=medic_records(run.path/'data/medic_images',limit=min(80,args.max_images-len(records)),budget=DownloadBudget(args.download_bytes))
                    records+=more;write_json(run.path/'data/medic_audit.json',{'records':len(more),'failures':failures,'source':source})
                    print('MEDIC native records',len(more),'downloaded bytes',source['downloaded_bytes'],flush=True)
                except Exception as error:
                    errors['medic']=str(error);print('MEDIC unavailable:',type(error).__name__,str(error),flush=True)
            records=add_duplicate_groups(records)
            historical_predictions(records,args.historical_cache)
            write_json(run.path/'data/records.json',records)
            write_json(run.path/'data/annotation_audit.json',{
                'parents':len({r['parent_id'] for r in records}),'groups':len({r['group'] for r in records}),
                'labels':{mission:{str(v):sum(r['labels'].get(mission)==v for r in records) for v in (0,1,None)} for mission in MISSIONS},
                'event_identity_verified':False,'notes':['Missing labels remain unknown.','Native observation atoms do not prove distinct assets.']})
            if len(records)<32:raise RuntimeError('insufficient real parents for 32-job experiment')
        for backbone in args.backbones.split(','):
            run.budget()
            with run.stage('02_'+backbone+'_vision'):
                model_path=Path(args.models_root)/backbone
                provider=FrozenVisionProvider(backbone,model_path,device='cpu',batch_size=1,local_files_only=True)
                output={};started=time.perf_counter();peak=psutil.Process().memory_info().rss
                try:
                    with (run.path/'features'/(backbone+'.jsonl')).open('x',encoding='utf-8') as handle:
                        for begin in range(0,len(records),10):
                            run.budget();chunk=records[begin:begin+10];images=[]
                            for record in chunk:
                                with Image.open(record['image_path']) as image:images.append(image.convert('RGB'))
                            predictions=provider.extract_batch(images,[MISSIONS['building_damage']]*len(images))
                            for record,prediction in zip(chunk,predictions):
                                output[record['image_id']]=prediction
                                handle.write(json.dumps({'image_id':record['image_id'],'evidence':prediction.to_dict()},allow_nan=False)+'\n')
                            handle.flush();peak=max(peak,psutil.Process().memory_info().rss)
                            print(backbone,'images',len(output),'/',len(records),'elapsed',round(time.perf_counter()-started,2),flush=True)
                            write_json(run.path/'features'/(backbone+'_progress.json'),{'count':len(output),'seconds':time.perf_counter()-started})
                    profiles[backbone]={'images':len(output),'wall_seconds':time.perf_counter()-started,
                        'forward_seconds':sum(b['forward_seconds'] for b in provider.timings),'sampled_process_peak_rss_bytes':peak,
                        'device':'cpu','producer':provider.provenance}
                finally:provider.close()
                cache[backbone]=output
                write_json(run.path/'metrics/extraction_profiles.json',profiles)
            with run.stage('03_'+backbone+'_development_study'):
                result=local_study(records,cache[backbone],run.path/'metrics'/backbone,run.budget,episodes=args.episodes,backbone=backbone)
                print('Restricted pilot grid rows',result['grid_rows'],flush=True)
        with run.stage('04_provisional_report'):
            lines=['# parameters laptop pilot','',
                   'Scope: real local CPU extraction and exposed-development parameter comparison. No final generalization or superiority claim.',
                   f'Parents: {len(records)}. Event identities are not verified. Native observation atoms and exact-copy controls are separate.',
                   'No neural codec inference or VLM labels are used. No automatic distress/urgency labels are created.','']
            decisions={}
            for backbone,profile in profiles.items():
                lines += [f'## {backbone}',f"Measured extraction wall time: {profile['wall_seconds']:.3f} s; forward-only: {profile['forward_seconds']:.3f} s."]
                summary=json.loads((run.path/'metrics'/backbone/'summary.json').read_text())
                for mission,value in summary.items():
                    selected=value.get('selected');decisions[backbone+'/'+mission]=value
                    if selected:
                        lines += [f"- {mission}: provisional groups {selected['groups']}, methods {selected['method']}, development A10={selected['mean_A10']:.4f}; header={value['header']['header_bytes']} bytes."]
                    else:lines += [f"- {mission}: insufficient supported data for selection."]
            lines += ['', '## Provisional decisions',
                      '- Keep context and causal KB mechanisms as candidates; the selected subset is not a universal result.',
                      '- No trusted capture times: deadline values remain unavailable in this static corpus.',
                      '- Spatial/complexity/geographic groups require verified producers/metadata; absence is not evidence that they are useless.',
                      '- Individual distress, universal urgency and natural cross-view identity remain unverified.',
                      '- Repeated queue seeds are not independent groups. This pilot uses a single grouped development split.',
                      '- Proceed to grouped Kaggle selection only after dual-GPU and data-access preflight.',
                      '', 'All predictions, traces, grids, specs and header registries are retained under this run directory.']
            if errors:lines += ['', '## Source failures',json.dumps(errors,indent=2)]
            report='\n'.join(lines)+'\n';(run.path/'PILOT_REPORT.md').write_text(report,encoding='utf-8')
            write_json(run.path/'metrics/provisional_decisions.json',decisions)
            write_json(run.path/'status.json',{'status':'complete','scope':'provisional laptop pilot','seconds':time.monotonic()-run.started})
            print(report,flush=True)
    except Exception as error:
        import traceback
        write_json(run.path/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'seconds':time.monotonic()-run.started})
        raise
    return run.path


def extract_command(args):
    from PIL import Image
    from .vision import FrozenVisionProvider
    spec=MethodSpecification(**json.loads(Path(args.spec).read_text())) if args.spec else MethodSpecification(backbone=args.backbone)
    from .contracts import MissionSpec,canonical
    from .learning import ClaimHeads,HeadedProvider
    mission=MissionSpec(**json.loads(Path(args.mission_file).read_text())) if args.mission_file else MISSIONS[args.mission]
    provider=FrozenVisionProvider(spec.backbone,Path(args.models_root)/spec.backbone,batch_size=8)
    wrapped=provider
    expected_heads=spec.producer_revisions.get('claim_heads_sha256')
    if expected_heads:
        head_path=Path(args.heads) if args.heads else Path(args.spec).parent/'claim_heads.json'
        artifact=json.loads(head_path.read_text())
        if digest(artifact)!=expected_heads:raise ValueError('claim head artifact differs from selected specification')
        wrapped=HeadedProvider(provider,ClaimHeads.from_dict(artifact))
    elif spec.relevance=='claim_probability':
        raise ValueError('claim_probability extraction requires a selected head artifact')
    extractor=SemanticParameterExtractor(spec,wrapped);memories={}
    codec=HeaderCodec(spec.feature_names,spec.sha256,spec.quantization_bits) if spec.feature_names else None
    images=[args.image] if args.image else json.loads(Path(args.records).read_text())
    rows=[]
    try:
        provider.load()
        expected_weights=spec.producer_revisions.get('image_weights_sha256')
        if expected_weights and provider.provenance['weights_sha256']!=expected_weights:
            raise ValueError('image checkpoint differs from frozen extraction specification')
        for begin in range(0,len(images),8):
            pixels=[];metadata=[];batch_memories=[]
            for sequence,value in enumerate(images[begin:begin+8],begin):
                item={'image_path':value} if isinstance(value,str) else value
                path=Path(item['image_path']);source=str(item.get('source_id',args.source))
                memories.setdefault(source,SenderMemory(source,mission.mission_id,spec.kb_capacity))
                with Image.open(path) as image:pixels.append(image.convert('RGB'))
                meta={key:item[key] for key in ('capture_time_ms','now_ms','timestamp_trusted','coordinates','location_trusted','expected_decode_ms') if key in item}
                meta.update(image_token=str(item.get('image_token',path.name)),source_id=source,sequence=sequence)
                metadata.append(meta);batch_memories.append(memories[source])
            serialize=codec.encode if codec else lambda bundle:canonical(bundle.to_dict()).encode()
            for bundle,wire in extractor.transmit_stream(pixels,[mission]*len(pixels),metadata,batch_memories,serialize):
                rows.append({'bundle':bundle.to_dict(),'header_bytes':len(wire),'encoding':'BSE1' if codec else 'analysis_json',
                             'header_hex':wire.hex() if codec else None})
        write_json(args.output,rows)
    finally:provider.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('pilot');p.add_argument('--output',required=True)
    p.add_argument('--ladi-manifest',default=str(_original()/'results/ladi-curation/v2/ladi260_semantic_scheduler_v2.csv'))
    p.add_argument('--historical-cache',default=str(Path(__file__).resolve().parents[1]/'results/disaster_decoder_v3/20261022_stageA_v3/extraction/features.jsonl'))
    p.add_argument('--models-root',default=str(_original()/'models/priority_classifiers'))
    p.add_argument('--max-images',type=int,default=180);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--download-bytes',type=int,default=1_000_000_000);p.add_argument('--medic',action='store_true')
    p.add_argument('--backbones',default='clip,siglip');p.add_argument('--episodes',type=int,default=4)
    e=commands.add_parser('extract');e.add_argument('--image');e.add_argument('--records');e.add_argument('--output',required=True)
    e.add_argument('--mission',choices=list(MISSIONS),default='building_damage');e.add_argument('--source',default='ue0')
    e.add_argument('--spec');e.add_argument('--heads');e.add_argument('--mission-file')
    e.add_argument('--backbone',default='clip');e.add_argument('--models-root',default=str(_original()/'models/priority_classifiers'))
    args=parser.parse_args()
    if args.command=='pilot':
        if not 1<=args.max_images<=200 or not 1<=args.seconds<=3600 or not 0<=args.download_bytes<=1_000_000_000:
            parser.error('laptop pilot exceeds approved limits')
        pilot(args)
    else:
        if bool(args.image)==bool(args.records):parser.error('choose one image or records list')
        extract_command(args)


if __name__=='__main__':main()
