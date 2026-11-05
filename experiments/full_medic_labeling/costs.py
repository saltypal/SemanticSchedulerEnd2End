"""Executed component and complete-pipeline cost measurements, never summed percentiles."""
import gc,json,time,platform
from pathlib import Path
import numpy as np
import torch,psutil,joblib
from PIL import Image,ImageOps
from experiments.spi_classifier_repair import medic_campaign as study
from experiments.spi_classifier_repair.resume_store import atomic_json


def operator_flops(call,**shape):
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as trace:
            call()
        events=trace.key_averages()
        return {'supported_flops':sum(e.flops for e in events),'unsupported_or_uncounted_ops':[e.key for e in events if not e.flops],
                'status':'partial','convention':'two FLOPs per multiply-accumulate','shape':shape}
    except Exception as error:return {'supported_flops':None,'status':'failed','reason':str(error),'shape':shape}


def summarize(samples):
    return {'p50_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),'samples_ms':samples}


def full_pipelines(rows,output):
    from ultralytics import YOLOWorld
    output=Path(output);dest=output/'full_pipeline_profiles';dest.mkdir(exist_ok=True)
    fixed=[r for r in rows if r['split']=='train'][:10]
    if len(fixed)!=10:raise ValueError('ten fixed development images required')
    candidates=json.loads((output/'frozen_pair_selections.json').read_text())
    for name in study.REPOS:
        identity=json.loads((output/'features'/name/'identity.json').read_text())
        for device in ('cpu','cuda:0'):
            print('COMPLETE PIPELINE LOAD',name,device,flush=True)
            begin=time.perf_counter();core,projection,proc,actual=study.load_encoder(name,device,pinned_identity=identity)
            encoder_load=time.perf_counter()-begin
            clip_core=clip_proj=clip_proc=None
            if name!='clip_b32':
                clip_identity=json.loads((output/'features/clip_b32/identity.json').read_text())
                begin=time.perf_counter();clip_core,clip_proj,clip_proc,_=study.load_encoder('clip_b32',device,pinned_identity=clip_identity)
                clip_load=time.perf_counter()-begin
            else:clip_load=0
            begin=time.perf_counter();detector=YOLOWorld('yolov8s-worldv2.pt').to(device);detector_load=time.perf_counter()-begin
            for c in [c for c in candidates if c['encoder']==name]:
                target=dest/(name+'_'+c['head']+'_'+device.replace(':','_')+'.json')
                if target.exists():continue
                neural=c['kind']=='token';begin=time.perf_counter()
                if neural:
                    state=torch.load(c['artifact'],map_location='cpu',weights_only=False)
                    head=study.TokenHead(state['dimension'],state['kind']);head.load_state_dict(state['state']);head=head.to(device).eval()
                else:head=joblib.load(c['artifact'])
                head_load=time.perf_counter()-begin
                last={};stage_rows=[]
                def sync():
                    if device.startswith('cuda'):torch.cuda.synchronize(0)
                def invoke(row,record=False):
                    stages={}
                    def step(label,fn):
                        sync();start=time.perf_counter();result=fn();sync();stages[label]=(time.perf_counter()-start)*1000;return result
                    def read():
                        with Image.open(row['image_path']) as im:return ImageOps.exif_transpose(im).convert('RGB')
                    image=step('image_loading',read)
                    def pixels(processor):
                        pad=ImageOps.pad(image,(224,224),color=(127,127,127));kw={'do_resize':False}
                        if hasattr(processor,'do_center_crop'):kw['do_center_crop']=False
                        return processor(images=[pad],return_tensors='pt',**kw)['pixel_values'].to(device)
                    px=step('encoder_preprocessing',lambda:pixels(proc))
                    with torch.inference_mode():
                        vector,tokens=step('encoder',lambda:study.encode(core,projection,px,name))
                        if clip_core is not None:
                            cp=step('retained_clip_preprocessing',lambda:pixels(clip_proc))
                            step('retained_clip',lambda:study.encode(clip_core,clip_proj,cp,'clip_b32'))
                        side=int(np.sqrt(tokens.shape[1]));yy,xx=np.meshgrid((np.arange(side)+.5)*224/side,(np.arange(side)+.5)*224/side,indexing='ij')
                        scale=min(224/image.width,224/image.height);w,h=image.width*scale,image.height*scale
                        mask=torch.tensor(((xx>=(224-w)/2)&(xx<(224+w)/2)&(yy>=(224-h)/2)&(yy<(224+h)/2)).reshape(1,-1),device=device)
                        if not bool(mask.any()):
                            half=112/side
                            mask=torch.tensor(((xx+half>(224-w)/2)&(xx-half<(224+w)/2)&
                                (yy+half>(224-h)/2)&(yy-half<(224+h)/2)).reshape(1,-1),device=device)
                            if not bool(mask.any()):raise ValueError('invalid image extent')
                        pred=step('head',lambda:head(tokens,mask) if neural else head.predict(vector.cpu().numpy()))
                        detected=step('detector',lambda:detector.predict(image,device=device,conf=.25,iou=.5,imgsz=640,max_det=300,verbose=False)[0])
                        step('serialization',lambda:json.dumps({'spi':pred.detach().cpu().tolist() if neural else pred.tolist(),
                            'boxes':detected.boxes.xyxy.cpu().tolist(),'availability':True}))
                        last.update(px=px,tokens=tokens,mask=mask)
                    if record:stage_rows.append({'image_id':row['image_id'],**stages})
                with torch.inference_mode():
                    for _ in range(10):invoke(fixed[0])
                    samples=[]
                    for row in fixed:
                        for _ in range(10 if device.startswith('cuda') else 3):
                            sync();start=time.perf_counter();invoke(row,True);sync();samples.append((time.perf_counter()-start)*1000)
                    batch_start=time.perf_counter()
                    for row in fixed:invoke(row)
                    sync();throughput=10/(time.perf_counter()-batch_start)
                    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(0)
                    invoke(fixed[0]);sync()
                    memory={'rss_bytes':psutil.Process().memory_info().rss,'peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if device.startswith('cuda') else None,
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(0) if device.startswith('cuda') else None}
                    eflops=operator_flops(lambda:study.encode(core,projection,last['px'],name),input=list(last['px'].shape),dtype=str(last['px'].dtype))
                    hflops=operator_flops(lambda:head(last['tokens'],last['mask']),tokens=list(last['tokens'].shape)) if neural else {'status':'sklearn formula estimates stored separately'}
                    dflops=operator_flops(lambda:detector.predict(fixed[0]['image_path'],device=device,conf=.25,iou=.5,imgsz=640,max_det=300,verbose=False),image_size=640,vocabulary=80)
                atomic_json(target,{'encoder':name,'head':c['head'],'device':device,'host':platform.node(),'image_ids':[r['image_id'] for r in fixed],
                    'warmups':10,'batch_size':1,'batch_one':summarize(samples),'stages':stage_rows,'throughput_images_per_second':throughput,
                    'throughput_protocol':'ten sequential batch-one calls, not vectorized batch ten','memory':memory,
                    'loads_seconds':{'encoder':encoder_load,'retained_clip':clip_load,'detector':detector_load,'head':head_load},
                    'load_scope':'fresh model object; OS/download cache may be warm','parameters':actual['parameters'],
                    'artifact_bytes':Path(c['artifact']).stat().st_size,'encoder_flops':eflops,'head_flops':hflops,'detector_flops':dflops,
                    'skipped_runtime_fields':['relevance','KB','capture_metadata'],'scope':'image to SPI and detector JSON; includes retained CLIP for non-CLIP pipeline'} )
                print('COMPLETE PIPELINE SAVED',target,flush=True)
                del head;gc.collect();torch.cuda.empty_cache()
            del core,projection,clip_core,clip_proj,detector;gc.collect();torch.cuda.empty_cache()


def teacher_profile(teacher,rows,output,system_prompt):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[]
    for row in [r for r in rows if r['split']=='train'][:10]:
        with Image.open(row['image_path']) as im:image=ImageOps.exif_transpose(im).convert('RGB')
        messages=[{'role':'system','content':[{'type':'text','text':system_prompt}]},
                  {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Assess this image.'}]}]
        inputs=teacher.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda:1')
        receipt={'image_id':row['image_id'],'input_tokens':int(inputs['input_ids'].shape[1]),
            'image_grid_thw':inputs.get('image_grid_thw').tolist() if inputs.get('image_grid_thw') is not None else None,
            'pixel_tensor_shape':list(inputs['pixel_values'].shape),'dtype':str(inputs['pixel_values'].dtype),'quantization':'NF4 FP16 compute, not zero arithmetic'}
        with torch.inference_mode():
            captured={}
            def prefill():
                captured['out']=teacher.model(**inputs,use_cache=True,logits_to_keep=1)
            torch.cuda.synchronize(1);start=time.perf_counter();prefill();torch.cuda.synchronize(1)
            receipt['prefill_seconds']=time.perf_counter()-start
            del captured['out']
            receipt['prefill_flops']=operator_flops(prefill,tokens=receipt['input_tokens'],visual_grid=receipt['image_grid_thw'])
            out=captured['out'];token=out.logits[:,-1:].argmax(-1)
            mask=torch.cat([inputs['attention_mask'],torch.ones_like(token)],dim=-1)
            def decode():
                return teacher.model(input_ids=token,attention_mask=mask,past_key_values=out.past_key_values,
                    cache_position=torch.tensor([inputs['input_ids'].shape[1]],device='cuda:1'),use_cache=False,logits_to_keep=1)
            # One cached decode step, not a repeated mutated-cache benchmark.
            receipt['decode_one_token_flops']=operator_flops(decode,batch=1,new_tokens=1,cached_tokens=receipt['input_tokens'])
            receipt['teacher_result']=teacher.review(image)
        records.append(receipt);atomic_json(output/'teacher_prefill_decode_profiles.json',records)
        print('TEACHER FLOPS SAVED',len(records),flush=True)
        del inputs,out,captured;gc.collect();torch.cuda.empty_cache()
