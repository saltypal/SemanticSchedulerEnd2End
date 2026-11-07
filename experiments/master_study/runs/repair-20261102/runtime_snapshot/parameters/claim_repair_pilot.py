"""Bounded native-claim repair experiment; no scheduler or model training."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
from .artifacts import write_json
from .image_claims import NativeAerialClaimProvider,NATIVE_FACTS

TARGETS=('building.present','road.present','building.affected_or_greater','road.damaged')


def binary_metrics(truth,predicted):
    if len(truth)!=len(predicted) or any(type(value) is not bool for value in truth):
        raise ValueError('equal-length boolean references required')
    counts={key:0 for key in ('tp','tn','fp','fn','unknown')}
    unknown_positive=0
    for actual,value in zip(truth,predicted):
        if type(value) is not bool:
            counts['unknown']+=1;unknown_positive+=int(actual);continue
        counts[('tp' if value else 'fn') if actual else ('fp' if value else 'tn')]+=1
    tp,tn,fp,fn=(counts[k] for k in ('tp','tn','fp','fn'))
    positives=sum(truth);negatives=len(truth)-positives
    return {**counts,'support':len(truth),'positive_support':positives,'negative_support':negatives,
        'coverage':(tp+tn+fp+fn)/len(truth) if truth else None,
        'accuracy_including_unknown':(tp+tn)/len(truth) if truth else None,
        'precision':tp/(tp+fp) if tp+fp else None,
        'recall_including_unknown':tp/positives if positives else None,
        'balanced_accuracy_including_unknown':(tp/positives+tn/negatives)/2 if positives and negatives else None,
        'f1':2*tp/(2*tp+fp+fn+unknown_positive) if 2*tp+fp+fn+unknown_positive else None,
        'unknown_positive_counted_as_missed':unknown_positive}


def evaluate(records,predictions):
    metrics={};native_for={fact:label for label,fact in NATIVE_FACTS.items()}
    for target in TARGETS:
        pairs=[(r['native_labels'][native_for[target]],predictions[r['image_sha256']][target]['value'])
               for r in records if type(r['native_labels'].get(native_for[target])) is bool]
        truth=[v[0] for v in pairs];predicted=[v[1] for v in pairs]
        metrics[target]={'candidate':binary_metrics(truth,predicted),
            'always_positive':binary_metrics(truth,[True]*len(truth)),
            'always_negative':binary_metrics(truth,[False]*len(truth))}
    return metrics


def run(manifest,output,seconds=180):
    from PIL import Image
    import torch,transformers
    root=Path(manifest).resolve().parent;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path(manifest).read_text())
    if source['split']!='test':raise ValueError('this frozen check requires official test-source records')
    records=[json.loads(line) for line in (root/source['evaluator_labels_file']).read_text().splitlines()]
    if len(records)>20:raise ValueError('bounded check supports at most20 images')
    if len({r['image_sha256'] for r in records})!=len(records):raise ValueError('duplicate test images')
    # Only image paths/pixels enter the producer; labels are scored afterward.
    provider=NativeAerialClaimProvider(device='cpu',batch_size=1)
    started=time.perf_counter();predictions={}
    write_json(output/'config.json',{'threshold':.5,'selection':'fixed first20officialtestrows',
        'seconds_after_load':seconds,'no_training_or_threshold_selection':True,
        'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest()})
    try:
        print('START reference model load, no training',flush=True);provider.load()
        write_json(output/'model.json',provider.provenance)
        began=time.perf_counter()
        with (output/'predictions.jsonl').open('w',encoding='utf-8') as handle:
            for index,row in enumerate(records):
                if time.perf_counter()-began>seconds:raise TimeoutError('bounded CPU inference exceeded limit')
                path=(root/row['image_path']).resolve()
                if not path.is_relative_to(root):raise ValueError('image escaped sample directory')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('input checksum mismatch')
                with Image.open(path) as image:claims=provider.predict([image.convert('RGB')])[0]
                predictions[row['image_sha256']]=claims
                handle.write(json.dumps({'image_sha256':row['image_sha256'],'claims':claims})+'\n');handle.flush()
                print('Image',index+1,'/',len(records),'elapsed',round(time.perf_counter()-began,2),flush=True)
        metrics=evaluate(records,predictions);write_json(output/'metrics.json',metrics)
        write_json(output/'status.json',{'status':'complete','images':len(records),'inference_seconds':time.perf_counter()-began,
            'total_seconds':time.perf_counter()-started,'torch':torch.__version__,'transformers':transformers.__version__,
            'python':sys.executable,'platform':platform.platform(),'device':'cpu','new_gpu_hours':0,
            'scope':'20ordered official-test-source images; descriptive repair feasibility, not representative or final',
            'model_exposure':'reference checkpoint trained official LADI train; dataset versions may differ',
            'metric_threshold':.5,'calibrated':False})
        write_json(output/'timings.json',provider.timings)
        print('COMPLETE',metrics,flush=True)
    except Exception as error:
        write_json(output/'status.json',{'status':'failed','completed_images':len(predictions),
            'error':type(error).__name__,'message':str(error),'total_seconds':time.perf_counter()-started})
        raise
    finally:provider.close()
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--seconds',type=int,default=180)
    args=parser.parse_args();run(args.manifest,args.output,args.seconds)
