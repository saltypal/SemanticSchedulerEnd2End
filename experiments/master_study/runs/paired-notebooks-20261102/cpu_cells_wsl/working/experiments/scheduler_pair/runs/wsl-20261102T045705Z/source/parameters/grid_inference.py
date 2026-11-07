"""CPU-first two-preprocessor native score cache, bounded and resumeless by design."""
from pathlib import Path
import argparse
import json
import time
import hashlib
from .image_claims import NativeAerialClaimProvider
from .approach_grid import evaluate_cache,PREPROCESSING
from .artifacts import write_json
from .contracts import digest


def run(manifest,output,seconds=180):
    from PIL import Image
    root=Path(manifest).resolve().parent;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    meta=json.loads(Path(manifest).read_text())
    records=[json.loads(line) for line in (root/meta['evaluator_labels_file']).read_text().splitlines()]
    if len(records)>20:raise ValueError('CPU diagnostic capped at20parents')
    provider=NativeAerialClaimProvider(device='cpu',batch_size=1);caches={};started=time.perf_counter()
    try:
        provider.load();write_json(output/'model_base.json',provider.provenance)
        for policy in PREPROCESSING:
            provider.preprocessing=policy
            provider.provenance={**provider.provenance,'input_policy':policy,
                'preprocessing_sha256':digest({'processor':provider.processor.to_dict(),'input_policy':policy})}
            predictions={}
            with (output/(policy+'.jsonl')).open('w',encoding='utf-8') as handle:
                for index,row in enumerate(records):
                    if time.perf_counter()-started>seconds:raise TimeoutError('bounded grid inference expired')
                    path=(root/row['image_path']).resolve()
                    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
                        raise ValueError('grid image identity mismatch')
                    with Image.open(path) as image:claims=provider.predict([image.convert('RGB')])[0]
                    predictions[row['image_sha256']]=claims
                    handle.write(json.dumps({'image_sha256':row['image_sha256'],'claims':claims,
                        'preprocessing':policy,'producer':provider.provenance})+'\n');handle.flush()
                    if (index+1)%5==0:print(policy,index+1,'/',len(records),'elapsed',round(time.perf_counter()-started,2),flush=True)
            caches[policy]=predictions
        rows=evaluate_cache(caches,records,output/'replay')
        write_json(output/'status.json',{'status':'complete','parents':len(records),'configurations':len(rows),
            'role':'diagnostic','no_winner_selected':True,'seconds':time.perf_counter()-started,
            'new_gpu_hours':0,'selection_reason':'prior exposed test sample; no final tuning allowed'})
        write_json(output/'timings.json',provider.timings)
    except Exception as error:
        write_json(output/'status.json',{'status':'failed','error':type(error).__name__,'message':str(error)})
        raise
    finally:provider.close()
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--seconds',type=int,default=180)
    args=parser.parse_args();run(args.manifest,args.output,args.seconds)
