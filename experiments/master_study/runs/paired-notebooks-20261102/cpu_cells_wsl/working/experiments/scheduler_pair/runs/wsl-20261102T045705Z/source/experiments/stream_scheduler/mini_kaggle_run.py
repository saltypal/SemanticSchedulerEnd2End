"""Bounded real-cache transport and dynamic baseline replay. No invented teacher labels."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tarfile
import time


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_transport(root):
    path = root/'Evaluation/compact_scene_transport.py'
    spec = importlib.util.spec_from_file_location('compact_scene_transport',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CompactSceneTransport


def replay(rows,ue_count,message_count,load,seed,out):
    out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    selected=rng.sample(rows,min(message_count,len(rows)))
    window=len(selected)/load
    arrivals=sorted([0.]*min(8,len(selected))+[rng.uniform(0,window) for _ in range(max(0,len(selected)-8))])
    jobs=[{'message_id':f'{seed}:{i}', 'image_id':r['image_id'], 'source_id':rng.randrange(ue_count),
           'arrival':a, 'spi':r['spi']['level'], 'sequence':i} for i,(r,a) in enumerate(zip(selected,arrivals))]
    horizon=window+32
    ranks={'Critical':0,'High':1,'Medium':2,'Low':3}
    results={}
    for policy in ['fifo','round_robin','random','predicted_spi_fifo']:
        pending=deque(jobs); ready=[]; now=0.; last_source=-1; events=[]; chooser=random.Random(seed+91)
        times=[]
        while now<horizon and (pending or ready):
            if not ready and pending and pending[0]['arrival']>now:
                now=pending[0]['arrival']
            while pending and pending[0]['arrival']<=now:
                ready.append(pending.popleft())
            if not ready or now>=horizon: break
            began=time.perf_counter_ns()
            if policy=='round_robin':
                sources=sorted({j['source_id'] for j in ready})
                chosen_source=next((s for s in sources if s>last_source),sources[0])
                j=min((j for j in ready if j['source_id']==chosen_source),key=lambda j:(j['arrival'],j['sequence']))
                last_source=chosen_source
            elif policy=='random': j=chooser.choice(ready)
            elif policy=='predicted_spi_fifo':j=min(ready,key=lambda j:(ranks[j['spi']],j['arrival'],j['sequence']))
            else:j=min(ready,key=lambda j:(j['arrival'],j['sequence']))
            times.append((time.perf_counter_ns()-began)/1e6)
            ready.remove(j)
            events.append({**j,'dispatch':now,'wait':now-j['arrival'],'finish':now+1,'observed_until':min(now+1,horizon),'complete':now+1<=horizon})
            now+=1
        picked={j['message_id'] for j in events}
        per={}
        for level in ranks:
            offered=[j for j in jobs if j['spi']==level]
            delivered=[j for j in events if j['spi']==level]
            per[level]={'offered':len(offered),'selected':len(delivered),
                'selected_mean_wait':sum(j['wait'] for j in delivered)/len(delivered) if delivered else None,
                'unselected':sum(j['message_id'] not in picked for j in offered)}
        p=out/f'{policy}.jsonl'
        p.write_text(''.join(json.dumps(j)+'\n' for j in events))
        results[policy]={'offered':len(jobs),'selected':len(events),'unselected':len(jobs)-len(events),
            'selected_mean_wait':sum(j['wait'] for j in events)/len(events) if events else None,
            'predicted_spi_descriptive_wait':per,'selection_mean_ms':sum(times)/len(times) if times else None,
            'J_D':None,'ndcg':None,'reference_scope':'NO independent urgency/preference references; descriptive input-class delays only'}
    dump(out/'offered.json',{'jobs':jobs,'service':'synthetic unit duration','horizon':horizon,'ue_count':ue_count,
         'mode':'MIXED_IID not real video chronology','seed':seed,'load':load})
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    out=args.output;root=Path(__file__).resolve().parents[2];start=time.perf_counter()
    print('START real cached scene transport + dynamic baseline replay',flush=True)
    h=hashlib.sha256()
    with args.archive.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!='0f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c':raise ValueError('wrong mini archive')
    C=load_transport(root)
    with tarfile.open(args.archive) as t:
        rows=[json.loads(x) for x in t.extractfile('Mini_dataset/handoff/image_parameters.jsonl') if x.strip()]
        registry=json.load(t.extractfile('Mini_dataset/scene/registry.json'))
    if len(rows)!=6000:raise ValueError('wrong image count')
    profiles={};sizes=[]
    for i,r in enumerate(rows):
        s=r['scene'];key=(s['registry_sha256'],s['producer_sha256'])
        if key not in profiles:
            profiles[key]=C(len(profiles)+1,*key,[c['count_applicable'] for c in registry['classes']])
        codec=profiles[key];original=bytes.fromhex(r['scene_wire_hex']);compact=codec.encode(original)
        if codec.decode(compact)!=original:raise ValueError('transport roundtrip mismatch')
        sizes.append(len(compact))
        if (i+1)%1000==0:print('TRANSPORT',i+1,'/6000 exact roundtrips',flush=True)
    dump(out/'profile_registry.json',{'profiles':[{'profile_id':p.profile_id,'registry_sha256':k[0],'producer_sha256':k[1]} for k,p in profiles.items()],
          'scope':'configured profile lookup; provisioning bytes accounted separately; no authentication claim'})
    train=[r for r in rows if r['split']=='train']
    traces={}
    for k,(ues,n,load) in enumerate([(4,48,.5),(50,256,.9),(100,512,1.2)]):
        print('REPLAY',ues,n,load,flush=True)
        traces[str(k)]=replay(train,ues,n,load,1729+k,out/'traces'/str(k))
    sizes.sort()
    summary={'status':'complete','scene_roundtrips':len(sizes),'scene_bytes_mean':sum(sizes)/len(sizes),
        'scene_bytes_p95':sizes[int(.95*(len(sizes)-1))],'scene_bytes_max':max(sizes),'profiles':len(profiles),
        'scene_bytes_scope':'actual encoded scene blocks only; SPI/relevance/novelty/routing/payload and provisioning excluded',
        'traces':traces,'channels_retained':93,'field_selection':'not performed; lossless sparse transport retains all channels',
        'training':'not performed; this CPU component run is not the P/neural/RL study',
        'relevance_novelty':'stream-dependent inputs; unavailable in cache; not needed by these four baselines',
        'elapsed_seconds':time.perf_counter()-start}
    dump(out/'summary.json',summary)
    dump(out/'validation.json',{'archive_sha256':h.hexdigest(),'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'Evaluation/compact_scene_transport.py']},
          'python':sys.version,'interpreter':sys.executable,'claims':'lossless transport and MIXED_IID baseline component behavior only'})
    dump(out/'artifact_manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
