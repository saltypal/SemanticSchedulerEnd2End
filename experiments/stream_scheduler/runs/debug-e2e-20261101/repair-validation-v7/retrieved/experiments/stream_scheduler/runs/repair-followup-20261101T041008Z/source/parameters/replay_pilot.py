"""Corrected, cache-only post-review deployment and quantization replay."""
from dataclasses import replace
from pathlib import Path
import json
import time
from .artifacts import Run,write_json
from .contracts import MethodSpecification,digest
from .extractor import VisualEvidence
from .learning import ClaimHeads,Scorer
from .study import episodes_for,score_episodes
from .cloud_study import project_episode
from .serialization import HeaderCodec
from .evaluation import replay


def run_replay(source,output):
    source=Path(source).resolve();run=Run(output,{'stage':'post_review_cached_replay','source_run':str(source),'seconds':1200})
    records=json.loads((source/'data/records.json').read_text());results={}
    config=json.loads((source/'config.json').read_text())
    for backbone in ('clip','siglip'):
        with run.stage(backbone+'_selected_header_replay'):
            raw={}
            with (source/'features'/(backbone+'.jsonl')).open() as handle:
                for line in handle:
                    row=json.loads(line);raw[row['image_id']]=VisualEvidence.from_dict(row['evidence'])
            directory=source/'metrics'/backbone
            head_artifact=json.loads((directory/'claim_heads.json').read_text())
            evidence=ClaimHeads.from_dict(head_artifact).apply(raw)
            split=json.loads((directory/'split.json').read_text())
            validation=[r for r in records if r['group'] in set(split['validation_groups'])]
            original=json.loads((directory/'summary.json').read_text())
            for mission,value in original.items():
                if value.get('selected') is None:continue
                spec=MethodSpecification(**json.loads((directory/(mission+'_specification.json')).read_text()))
                if spec.producer_revisions.get('claim_heads_sha256')!=digest(head_artifact):raise ValueError('head/spec mismatch')
                policy=Scorer(json.loads((directory/(mission+'_scorer.json')).read_text()))
                episodes,missing=episodes_for(validation,evidence,mission,spec,count=config['episodes'],seed=1007)
                full=score_episodes(episodes,policy)
                expected=value['selected_validation']['mean_A10']
                if full['mean_A10'] is None or abs(full['mean_A10']-expected)>1e-12:raise RuntimeError('cached full-precision replay does not reproduce pilot')
                summary={'full_precision_A10':full['mean_A10'],'original_source_manifest':str(source/'manifest.json'),'quantization':{}}
                for bits in (8,16):
                    qspec=replace(spec,quantization_bits=bits);codec=HeaderCodec(policy.encoder.fields,qspec.sha256,bits)
                    transmitted=[project_episode(ep,qspec,codec) for ep in episodes]
                    score=score_episodes(transmitted,policy)
                    summary['quantization'][str(bits)]={'A10':score['mean_A10'],
                        'delta_from_full':score['mean_A10']-full['mean_A10'],
                        'orders_changed':sum(replay(a,policy)['order']!=replay(b,policy)['order'] for a,b in zip(episodes,transmitted)),
                        'header':codec.describe(),'trace':score['rows']}
                results[backbone+'/'+mission]=summary
                print(backbone,mission,'full',summary['full_precision_A10'],
                      '8bit',summary['quantization']['8']['A10'],'16bit',summary['quantization']['16']['A10'],flush=True)
    write_json(run.path/'metrics/replay.json',results)
    write_json(run.path/'status.json',{'status':'complete','fresh_vision_inference':False,'seconds':time.monotonic()-run.started})
    return run.path


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run_replay(a.source,a.output)
