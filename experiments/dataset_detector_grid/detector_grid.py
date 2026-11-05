"""Development sensitivity grid; accuracy selection requires separate box truth."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

def configurations(model):
    if model=='yolo_world':
        rows=[dict(confidence=c,iou=i,image_size=s,max_detections=300) for c,i,s in
              itertools.product((.15,.25,.40),(.3,.5,.7),(640,960))]
        base=dict(confidence=.25,iou=.5,image_size=640)
    elif model=='grounding_dino':
        rows=[dict(box_threshold=b,text_threshold=t,iou=i,image_size=s,max_detections=300) for b,t,i,s in
              itertools.product((.25,.35,.45),(.2,.25,.35),(.3,.5,.7),(640,800))]
        base=dict(box_threshold=.35,text_threshold=.25,iou=.5,image_size=800)
    else:raise ValueError('unknown model')
    return rows+[{**base,'max_detections':n} for n in (100,600)]

def run(model,checkpoint,records,output,device='cpu',limit_configs=None,seconds=4800):
    import torch
    from PIL import Image,ImageOps
    from parameters.scene_measurements import measure_detections
    from parameters.grounding_scene import suppress_duplicates
    from parameters.pilot_parameters import VOCABULARY
    from download_test import save
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    settings=configurations(model)
    deadline=time.monotonic()+seconds
    if limit_configs:settings=settings[:limit_configs]
    if model=='yolo_world':
        from ultralytics import YOLOWorld
        detector=YOLOWorld(str(checkpoint)).to(device)
        # Reuse the project's existing OpenAI text weights during local checks.
        # Ultralytics otherwise downloads another copy into its default cache.
        local_clip=Path(__file__).resolve().parents[2]/'models/clip_openai/ViT-B-32.pt'
        import clip
        original_load=clip.load
        def cached_load(name,*args,**kwargs):
            return original_load(str(local_clip) if name=='ViT-B/32' and local_clip.is_file() else name,*args,**kwargs)
        try:
            clip.load=cached_load
            detector.set_classes(list(VOCABULARY))
        finally:clip.load=original_load
    else:
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        processor=AutoProcessor.from_pretrained(checkpoint)
        detector=AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(device).eval()
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    save(output/'manifest.json',{'model':model,'checkpoint':str(checkpoint),'device':device,'vocabulary':VOCABULARY,
        'settings':settings,'records':records,'scope':'development sensitivity only; no accuracy winner without reference boxes'})
    for k,setting in enumerate(settings):
        rows=[]
        for record in records:
            if time.monotonic()>deadline:
                save(output/'status.json',{'status':'time_budget_exhausted','completed_settings':k,'partial_images':len(rows)})
                return
            with Image.open(record['image_path']) as source:image=ImageOps.exif_transpose(source).convert('RGB')
            actual=hashlib.sha256(Path(record['image_path']).read_bytes()).hexdigest()
            if record.get('image_sha256',actual)!=actual:raise ValueError('image hash mismatch')
            sync();started=time.perf_counter()
            if model=='yolo_world':
                result=detector.predict(image,conf=setting['confidence'],iou=setting['iou'],imgsz=setting['image_size'],
                    max_det=setting['max_detections'],agnostic_nms=False,half=False,augment=False,device=device,verbose=False)[0]
                boxes=[{'label':result.names[int(c)],'score':float(s),'box':b} for b,s,c in
                    zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
            else:
                inputs=processor(images=image,text='. '.join(VOCABULARY)+'.',return_tensors='pt',
                    size={'shortest_edge':setting['image_size'],'longest_edge':1333}).to(device)
                with torch.inference_mode():raw=detector(**inputs)
                result=processor.post_process_grounded_object_detection(raw,inputs.input_ids,threshold=setting['box_threshold'],
                    text_threshold=setting['text_threshold'],target_sizes=[(image.height,image.width)])[0]
                names=result.get('text_labels',result.get('labels'))
                boxes=[{'label':n,'score':float(s),'box':b.tolist()} for n,s,b in zip(names,result['scores'],result['boxes'])]
                valid=measure_detections(boxes,image.width,image.height).audit['detections']
                boxes=suppress_duplicates(valid,iou=setting['iou'],maximum=setting['max_detections'])
            sync();elapsed=time.perf_counter()-started
            summary=measure_detections(boxes,image.width,image.height,max_detections=setting['max_detections'])
            rows.append({'image_id':record['image_id'],'image_sha256':actual,'seconds':elapsed,
                         'values':summary.values(),'audit':summary.audit,'saturated':summary.saturated})
            save(output/f'setting_{k:03d}.json',{'settings':setting,'rows':rows})
        print(model,k+1,'/',len(settings),'images',len(rows),flush=True)
    save(output/'status.json',{'status':'complete','settings':len(settings),'images':len(records),
                             'accuracy_selection':'unavailable_without_reference_boxes'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['yolo_world','grounding_dino'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--records',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--limit-configs',type=int);p.add_argument('--seconds',type=int,default=4800)
    args=p.parse_args();run(args.model,args.checkpoint,json.loads(Path(args.records).read_text()),args.output,args.device,args.limit_configs,args.seconds)
