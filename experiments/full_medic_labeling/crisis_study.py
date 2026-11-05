"""CrisisMMD adapter: saved teacher labels, immutable image hashes, symmetric heads."""
import gc, hashlib, io, json, time, urllib.request
from pathlib import Path
import numpy as np
from experiments.spi_classifier_repair import medic_campaign as study
from experiments.spi_classifier_repair.resume_store import atomic_json


def prepare(selected, labels_path, output, frozen_records=None):
    output=Path(output); images=output/'images';images.mkdir(parents=True,exist_ok=True)
    labels={r['item_id']:r for r in map(json.loads,Path(labels_path).read_text().splitlines())}
    if set(labels)!={r['image_id'] for r in selected}:raise ValueError('label/cohort identity mismatch')
    # Connected components across both tweet identity and exact byte identity.
    parent={r['image_id']:r['image_id'] for r in selected}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for r in selected:
        i=r['image_id']
        for key in ('tweet:'+str(r['tweet_id']),'sha:'+labels[i]['image_sha256']):
            if key in seen:parent[root(i)]=root(seen[key])
            else:seen[key]=i
    priority={'train':0,'dev':1,'test':2};roles={}
    for r in selected:
        g=root(r['image_id']);role=r['split']
        if priority[role]>priority.get(roles.get(g),-1):roles[g]=role
    rows=[];excluded=[]
    for n,r in enumerate(selected):
        i=r['image_id'];label=labels[i];path=images/(i+'.jpg')
        if not path.exists():
            url='https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/10a5626ba112ab9c50369c473a2062bcf205d708/'+r['image_path']
            raw=None
            for attempt in range(5):
                try:
                    # Cache-busting avoids stale signed redirects on transient CDN failures.
                    request_url=url+('?download=true&retry='+str(attempt) if attempt else '')
                    with urllib.request.urlopen(request_url,timeout=90) as response:raw=response.read()
                    break
                except Exception as error:
                    print('DOWNLOAD RETRY',i,attempt+1,type(error).__name__,str(error),flush=True)
                    if attempt==4:raise
                    time.sleep(min(16,2**attempt))
            tmp=path.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=label['image_sha256']:raise ValueError('image checksum mismatch '+i)
        role=roles[root(i)]
        if label.get('available') is not True or label.get('level') not in study.LEVELS:
            excluded.append({'id':i,'reason':'unavailable_teacher'});continue
        rows.append({'image_id':i,'image_sha256':label['image_sha256'],'image_path':str(path),
                     'group':root(i),'split':'validation' if role=='dev' else role,
                     'level':label['level'],'available':True})
        if n%25==0:print('IMAGE VERIFIED',n+1,len(selected),flush=True)
    if frozen_records is not None:
        saved=json.loads(Path(frozen_records).read_text())
        reference={r['image_id']:r for r in saved}
        if len(reference)!=len(saved) or set(reference)!={r['image_id'] for r in rows}:
            raise ValueError('frozen fitting cohort identity mismatch')
        assigned={}
        for r in rows:
            prior=reference[r['image_id']]
            if (r['image_sha256'],r['level'])!=(prior['image_sha256'],prior['level']):
                raise ValueError('frozen image or label mismatch')
            if prior['split'] not in ('train','validation','test'):raise ValueError('invalid frozen role')
            for key in ('group','split'):r[key]=prior[key]
            for key in ('group:'+r['group'],'sha:'+r['image_sha256'],'component:'+root(r['image_id'])):
                if key in assigned and assigned[key]!=r['split']:raise ValueError('frozen split leakage')
                assigned[key]=r['split']
        atomic_json(output/'records.json',rows);atomic_json(output/'exclusions.json',excluded)
        atomic_json(output/'split_protocol.json',{'mode':'exact_saved_assignments','source_sha256':hashlib.sha256(Path(frozen_records).read_bytes()).hexdigest()})
        atomic_json(output/'support.json',{role:{level:sum(r['split']==role and r['level']==level for r in rows) for level in study.LEVELS} for role in ('train','validation','test')})
        print('FROZEN SPLITS RESTORED',len(rows),flush=True)
        return rows
    # Repartition DEVELOPMENT only. Exact-copy/tweet components remain intact;
    # native test components stay sealed. Native train alone lacks Medium support.
    from sklearn.model_selection import StratifiedGroupKFold
    pool=[i for i,r in enumerate(rows) if r['split']!='test']
    y=np.array([study.LEVELS.index(rows[i]['level']) for i in pool])
    groups=np.array([rows[i]['group'] for i in pool])
    selected_split=None
    for seed in range(17,117):
        for a,b in StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(pool)),y,groups):
            if all(len(set(groups[a][y[a]==k]))>=2 for k in range(4)) and set(y[b])==set(range(4)):
                selected_split=(a,b,seed);break
        if selected_split is not None:break
    if selected_split is None:raise ValueError('insufficient independent four-class development groups')
    a,b,seed=selected_split
    for ix in a:rows[pool[ix]]['split']='train'
    for ix in b:rows[pool[ix]]['split']='validation'
    atomic_json(output/'split_protocol.json',{'seed':seed,'scope':'native train+dev repartitioned; native test components reserved',
        'criterion':'class/group support only; no feature/model scores used','near_duplicate_status':'not audited; exact SHA and tweet groups enforced'})
    atomic_json(output/'records.json',rows);atomic_json(output/'exclusions.json',excluded)
    support={role:{level:sum(r['split']==role and r['level']==level for r in rows) for level in study.LEVELS} for role in ('train','validation','test')}
    atomic_json(output/'support.json',support);print('ACTUAL SUPPORT',support,flush=True)
    return rows


def run(rows,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def loader(records,name,out):
        # All balance studies use one common, checksum-validated feature bank.
        return study.features(records,name,output/'features'/name)
    outcomes=[]
    for balance in ('original','class_weighted','random_oversampling','moderate_undersampling'):
        print('BALANCE STUDY START',balance,flush=True)
        try:
            winner,receipt=study.compare(rows,rows,output/balance,feature_loader=loader,
                token_encoders=tuple(study.REPOS),balance_method=balance)
            outcomes.append({'balance':balance,'status':'complete','winner':winner,'receipt':receipt})
        except Exception as error:
            outcomes.append({'balance':balance,'status':'failed','error':str(error)})
            import traceback;traceback.print_exc()
        atomic_json(output/'study_status.json',outcomes)
    if any(r['status']!='complete' for r in outcomes):raise RuntimeError('some comparisons failed; see study_status.json')
    finalize(rows,output)
    from experiments.full_medic_labeling.costs import full_pipelines
    full_pipelines(rows,output)
    return outcomes


def finalize(rows,output):
    """Freeze one development-selected balance per pair before opening test."""
    import joblib,torch,psutil,platform
    output=Path(output);candidates=[]
    for method in ('original','class_weighted','random_oversampling','moderate_undersampling'):
        for c in json.loads((output/method/'candidates.json').read_text()):
            candidates.append({**c,'balance':method})
    selected={}
    for c in candidates:
        key=(c['encoder'],c['head'])
        if key not in selected or c['development']['macro_f1']>selected[key]['development']['macro_f1']:selected[key]=c
    atomic_json(output/'frozen_pair_selections.json',list(selected.values()))
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    y=np.array([study.LEVELS.index(rows[i]['level']) for i in test]);results=[]
    fixed=np.array([i for i,r in enumerate(rows) if r['split']=='train'][:10])
    for (encoder,head),c in selected.items():
        print('FINAL TEST AND HEAD PROFILE',encoder,head,flush=True)
        with np.load(output/'features'/encoder/'features.npz',allow_pickle=False) as f:
            x=f['embeddings'];t=f['tokens'];m=f['mask']
        neural=c['kind']=='token'
        if neural:
            saved=torch.load(c['artifact'],map_location='cpu',weights_only=False)
            model=study.TokenHead(saved['dimension'],saved['kind']);model.load_state_dict(saved['state']);model.eval()
            pred=study.token_predict(model,t[test],m[test]).argmax(1)
        else:
            model=joblib.load(c['artifact']);pred=model.predict(x[test])
        profiles={}
        for device in ('cpu','cuda:0'):
            if not neural and device!='cpu':
                profiles[device]={'available':False,'reason':'sklearn estimator CPU only; GPU encoder + CPU head is deployed hybrid'};continue
            if neural:model.to(device)
            def predict(i):
                if neural:
                    with torch.inference_mode():return model(torch.tensor(t[i:i+1],device=device),torch.tensor(m[i:i+1],device=device))
                return model.predict(x[i:i+1])
            def sync():
                if device.startswith('cuda'):torch.cuda.synchronize(0)
            for _ in range(10):predict(int(fixed[0]))
            sync();times=[]
            for i in fixed:
                for _ in range(10 if device.startswith('cuda') else 3):
                    sync();start=time.perf_counter();predict(int(i));sync();times.append((time.perf_counter()-start)*1000)
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(0)
            predict(int(fixed[0]));sync()
            profiles[device]={'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
                'samples_ms':times,'host':platform.node(),'rss_bytes':psutil.Process().memory_info().rss,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if device.startswith('cuda') else None,
                'scope':'head only; tensor creation/transfer included, encoder excluded; memory separate pass'}
        results.append({'encoder':encoder,'head':head,'balance':c['balance'],'test':study.score(y,pred),
            'profiles':profiles,'artifact_bytes':Path(c['artifact']).stat().st_size})
        atomic_json(output/'final_test_and_head_profiles.json',results)
        del model,x,t,m;gc.collect();torch.cuda.empty_cache()
    atomic_json(output/'cost_scope.json',{'teacher_cost':'saved generation wall time/tokens/memory; prefill/decode FLOPs unavailable for prior run',
        'encoder_finetuning':'not performed; frozen study','relevance_KB_serialization':'skipped as requested',
        'FLOPs':'2 per MAC; neural supported operators partial; sklearn training not instrumented, never zero',
        'remaining_cost_gap':'complete combined CLIP+alternate encoder+head+detector latency not measured by summing component percentiles',
        'labels':'pseudo-label agreement only; Medium support severely limited'})
