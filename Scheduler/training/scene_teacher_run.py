"""Bounded multi-image Qwen worker; control failure stops automatic labeling."""
from pathlib import Path
import argparse,json,time,hashlib
from Scheduler.training.scene_prepare import write_json
from Scheduler.training.scene_teacher import QwenChoiceTeacher,control_gate,TeacherLedger,aspect_preferences
from Scheduler.scene_policy import canonical,digest

MODEL='Qwen/Qwen3.5-9B'
REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'


def load_teacher():
    import torch
    import importlib.metadata
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    if torch.cuda.device_count()<2:raise RuntimeError('GPU0 extraction/GPU1 teacher profile requires two visible GPUs')
    assets=snapshot_download(MODEL,revision=REVISION)
    processor=AutoProcessor.from_pretrained(assets,local_files_only=True,size={'shortest_edge':65536,'longest_edge':262144})
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(assets,local_files_only=True,dtype=torch.float16,
        device_map={'':1},quantization_config=q,attn_implementation='sdpa').eval()
    if {str(p.device) for p in model.parameters()}!={'cuda:1'}:raise RuntimeError('teacher placement differs')
    return QwenChoiceTeacher(model,processor,{'model_id':MODEL,'revision':REVISION,'quantization':'NF4',
        'runtime':{p:importlib.metadata.version(p) for p in ('transformers','torch','bitsandbytes','accelerate','huggingface-hub')}},'cuda:1')


def run(data,requests,output,*,seconds=3600,maximum_calls=216,prior_feedback=None):
    from PIL import Image
    data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();rows=json.loads((data/'records.json').read_text())
    train=[r for r in rows if r['role']=='train'];positives=[r for r in train if r['damage_positive']]
    negatives=[r for r in train if not r['damage_positive']]
    if len(positives)<8 or len(negatives)<8:raise ValueError('insufficient independent teacher controls')
    teacher=load_teacher();ledger=TeacherLedger(output/'responses.jsonl');calls=0;durations=[]
    previous=json.loads(Path(prior_feedback).read_text()) if prior_feedback else None
    data_sha=hashlib.sha256((data/'records.json').read_bytes()).hexdigest()
    if previous and (previous.get('model')!=teacher.identity or previous.get('data_sha256')!=data_sha):
        raise ValueError('prior teacher model/data identity differs')
    if previous:calls=previous['calls']
    cached={}
    def query_key(hashes,instruction,history,aspects):
        return digest({'images':hashes,'mission':instruction,'history':history,'aspects':aspects})
    if previous:
        prior_log=Path(prior_feedback).parent/'responses.jsonl'
        for line in prior_log.read_text().splitlines():
            row=json.loads(line);request=row['request']
            if row.get('status')=='valid':
                cached[query_key(row['input_hashes'],request['mission'],request.get('history_summary',''),request.get('aspects',False))]=row['response']
    gray=output/'unassessable.png';Image.new('RGB',(256,256),(127,127,127)).save(gray)
    controls=[];aspect_controls=[];by_id={r['image_id']:r for r in rows}
    mission='Document visible damage to buildings for post-event assessment. A clear scene without supported damage is less relevant than clear damage. If both images are assessable and equivalent return tie. If neither provides assessable visual evidence return uncertain. Never infer rescue urgency.'
    replay_instruction=' Both options are the same clip; preference must be tie. Emit training_value integer: 0 for exact repeated assessable pictures, 2 for a visibly contrasting damaged/intact pair, 1 for unassessable evidence. These categories are a fixed replay heuristic, not operational importance.'
    def call(images,aspects=False,history='',instruction=''):
        nonlocal calls
        hashes=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for _,path in images]
        cache_key=query_key(hashes,mission+instruction,history,aspects)
        if cache_key in cached:return cached[cache_key]
        projected=float(sum(durations[-3:])/len(durations[-3:])) if durations else 0.
        if calls>=maximum_calls or time.perf_counter()-start+projected+30>=seconds:
            raise TimeoutError('teacher call/time cap; preserve partial feedback')
        try:
            result=teacher.compare(images,mission+instruction,history_summary=history,aspects=aspects)
            if result.get('parse_error'):result['error']=result['parse_error']
            durations.append(result['seconds']);calls+=1
            feedback['calls']=calls
            ledger.append(request={'images':[(i,str(p)) for i,p in images],'mission':mission+instruction,
                'history_summary':history,'aspects':aspects},response=result,model=teacher.identity,
                          prompt_version='scene_choice_v1',input_hashes=result['input_hashes'],status='invalid' if 'error' in result else 'valid')
            return result
        except (ValueError,KeyError,json.JSONDecodeError) as e:
            calls+=1;feedback['calls']=calls
            ledger.append(request={'images':[(i,str(p)) for i,p in images]},response={'error':str(e)},model=teacher.identity,
                          prompt_version='scene_choice_v1',input_hashes=[],status='invalid')
            return {'parsed':{'preference':'uncertain'},'error':str(e)}
    feedback={'schema':'scene_teacher_feedback_v1','fixture':False,'model':teacher.identity,
              'data_sha256':data_sha,'calls':0,'controls':[],'gates':{},'image_pairs':[],'prefix_pairs':[],'replay':[]}
    if previous:feedback=previous
    def strip_paths(paths,times,name):
        from PIL import ImageOps,ImageDraw
        canvas=Image.new('RGB',(1024,280),'white');draw=ImageDraw.Draw(canvas)
        for j,(path,finish) in enumerate(zip(paths,times)):
            with Image.open(path) as image:tile=ImageOps.contain(image.convert('RGB'),(250,250))
            canvas.paste(tile,(j*256,25));draw.text((j*256+4,4),'Completed t='+str(finish),fill='black')
        path=output/(name+'.png');canvas.save(path);return path
    try:
        if previous is None or not all(key in feedback.get('gates',{}) for key in ('aspects','prefix','replay')):
            preflights=[]
            from PIL import ImageOps
            largest=[]
            for label,row in zip(('A','B'),(positives[0],negatives[0])):
                path=output/('largest-input-'+label+'.png')
                with Image.open(data/row['image_path']) as image:
                    ImageOps.pad(image.convert('RGB'),(512,512),color='white').save(path)
                largest.append((label,path))
            for images in ([('A',data/positives[0]['image_path'])],
                           [('A',data/positives[0]['image_path']),('B',data/negatives[0]['image_path'])],largest):
                result=call(images)
                preflights.append(result);write_json(output/'technical_preflight.json',preflights)
                if 'error' in result:raise ValueError('one/two-image technical preflight failed')
                if result.get('image_count')!=len(images) or 'pixel_values' not in result.get('input_shapes',{}):
                    raise ValueError('teacher did not receive actual image tensor inputs')
            for label in ('A','B','tie','uncertain'):
                for i in range(4):
                    a=data/positives[i]['image_path'];b=data/negatives[i]['image_path']
                    images=[('A',a),('B',b)] if label=='A' else ([('A',b),('B',a)] if label=='B' else
                        [('A',a),('B',a)] if label=='tie' else [('A',gray),('B',gray)])
                    result=call(images,aspects=True)
                    controls.append({'expected':label,'observed':result['parsed']['preference'],'valid':'error' not in result})
                    try:
                        pairs=aspect_preferences(result['parsed'],('A','B'))
                        observed='A' if (0,1,1.) in pairs else ('B' if (1,0,1.) in pairs else 'tie')
                        if label=='uncertain':observed=result['parsed']['preference']
                        aspect_controls.append({'expected':label,'observed':observed,'valid':'error' not in result})
                    except ValueError:aspect_controls.append({'expected':label,'observed':'invalid','valid':False})
                    write_json(output/'control_progress.json',controls)
                    print('Image/aspect control',len(controls),'/16',flush=True)
            gate=control_gate(controls);feedback['controls']=controls;feedback['gate']=gate
            feedback['gates']['aspects']={**control_gate(aspect_controls),'scope':'aspect-derived order concordance; individual aspect scores are weak supervision'}
            write_json(output/'feedback.json',feedback)
            if not gate['passed']:
                write_json(output/'status.json',{'status':'blocked','reason':'task-specific teacher gate failed','calls':calls,'gate':gate})
                return feedback
            prefix_controls=[]
            timeline_instruction=' These are delivery timelines at equal endpoint t=4. Prefer distinct useful building-damage evidence exposed earlier. Repeated exact pictures add no new evidence. Completion times are printed above tiles.'
            for label in ('A','B','tie','uncertain'):
                for i in range(2):
                    positive=data/positives[4+i]['image_path'];negative=data/negatives[4+i]['image_path']
                    paths_a=[positive,negative] if i==0 else [positive,data/positives[6]['image_path']]
                    paths_b=[negative,positive] if i==0 else [positive,positive]
                    a=strip_paths(paths_a,[1,2],'prefix-control-'+label+str(i)+'-a')
                    b=strip_paths(paths_b,[1,2],'prefix-control-'+label+str(i)+'-b')
                    images=[('A',a),('B',b)] if label=='A' else ([('A',b),('B',a)] if label=='B' else
                        [('A',a),('B',a)] if label=='tie' else [('A',gray),('B',gray)])
                    result=call(images,instruction=timeline_instruction)
                    prefix_controls.append({'expected':label,'observed':result['parsed']['preference'],'valid':'error' not in result})
            feedback['gates']['prefix']={**control_gate(prefix_controls),'controls':prefix_controls}
            replay_controls=[]
            for i,expected in enumerate((0,0,0,0,2,2,1,1)):
                positive=data/positives[i]['image_path'];negative=data/negatives[i]['image_path']
                paths=[positive,positive] if expected==0 else ([positive,negative] if expected==2 else [gray,gray])
                clip=strip_paths(paths,[1,2],'replay-control-'+str(i))
                result=call([('A',clip),('B',clip)],instruction=replay_instruction)
                replay_controls.append({'expected':expected,'observed':result['parsed'].get('training_value'),
                                        'valid':'error' not in result})
            rates={str(k):sum(r['valid'] and r['observed']==k for r in replay_controls if r['expected']==k)/sum(r['expected']==k for r in replay_controls) for k in (0,1,2)}
            feedback['gates']['replay']={'passed':all(v>=.75 for v in rates.values()),'class_rates':rates,
                'controls':replay_controls,'scope':'declared heuristic rule-following, not proof of useful replay value'}
            write_json(output/'feedback.json',feedback)
        request_paths=sorted(Path(requests).glob('*requests*.jsonl')) if Path(requests).is_dir() else [Path(requests)]
        request_rows=[json.loads(line) for path in request_paths for line in path.read_text().splitlines() if line.strip()]
        seen={r['request_id'] for key in ('image_pairs','prefix_pairs','replay') for r in feedback[key]}
        for request in [r for r in request_rows if r['request_id'] not in seen and r.get('kind','image')=='image' and feedback['gates']['aspects']['passed']][:40]:
            images=[(r['id'],data/by_id[r['image_id']]['image_path']) for r in request['images']]
            forward=call(images,aspects=True,history=request.get('history_summary',''))
            reverse=call([('A',images[1][1]),('B',images[0][1])],aspects=True,history=request.get('history_summary',''))
            if 'error' in forward or 'error' in reverse:continue
            feedback['image_pairs'].append({**request,'group':request['request_id'],'forward':forward,'reverse':reverse})
            write_json(output/'feedback.json',feedback)
        def strip(identity_list,times,name):
            return strip_paths([data/by_id[identity]['image_path'] for identity in identity_list],times,name)
        for request in [r for r in request_rows if r['request_id'] not in seen and r.get('kind')=='prefix' and feedback['gates']['prefix']['passed']][:40]:
            a=strip(request['images_left'],request['times_left'],request['request_id']+'-left')
            b=strip(request['images_right'],request['times_right'],request['request_id']+'-right')
            instruction=' Compare these two delivery timelines at the SAME elapsed endpoint '+str(request['endpoint'])+'. Value useful distinct visible building-damage evidence exposed earlier. Exact repeated pictures add no new evidence. Do not prefer a timeline simply for having more tiles. Completion times are printed above each image.'
            forward=call([('A',a),('B',b)],instruction=instruction)
            reverse=call([('A',b),('B',a)],instruction=instruction)
            if 'error' not in forward and 'error' not in reverse:
                feedback['prefix_pairs'].append({**request,'forward':forward,'reverse':reverse})
            write_json(output/'feedback.json',feedback)
        for request in [r for r in request_rows if r['request_id'] not in seen and r.get('kind')=='replay' and feedback['gates']['replay']['passed']][:24]:
            clip=strip(request['images'],request['times'],request['request_id'])
            result=call([('A',clip),('B',clip)],instruction=replay_instruction)
            value=result['parsed'].get('training_value')
            valid='error' not in result and type(value) is int and value in (0,1,2)
            feedback['replay'].append({**request,'training_value':value if valid else None,'valid':valid})
            write_json(output/'feedback.json',feedback)
        write_json(output/'status.json',{'status':'complete','calls':calls,
            'image_pairs':len(feedback['image_pairs']),'prefix_pairs':len(feedback['prefix_pairs']),
            'replay':len(feedback['replay']),'seconds':time.perf_counter()-start})
    except Exception as e:
        write_json(output/'feedback.json',feedback)
        write_json(output/'status.json',{'status':'partial','error':type(e).__name__,'reason':str(e),'calls':calls,'seconds':time.perf_counter()-start})
        raise
    return feedback

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--requests',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--prior-feedback')
    a=p.parse_args();run(a.data,a.requests,a.output,seconds=a.seconds,prior_feedback=a.prior_feedback)
