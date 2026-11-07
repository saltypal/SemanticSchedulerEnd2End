"""Calibration, image-runtime validation, profiles and fail-closed release record."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from parameters.artifacts import write_json
from experiments.spi_classifier_repair.medic_campaign import LEVELS,folds,build,fit_token,token_predict,score
from experiments.spi_classifier_repair.medic_deployment import SPIImagePredictor,fit_temperature,probabilities
from experiments.spi_classifier_repair.resume_store import ResumeStore,fingerprint,file_hash,atomic_json,implementation_identity

def calibration(winner,rows,labels,features,output):
    y=np.array([LEVELS.index(r['level']) if r.get('available') else -1 for r in labels]);tr=np.array([i for i,r in enumerate(rows) if r['split']=='train' and y[i]>=0])
    groups=np.array([rows[i]['group'] for i in tr]);x,t,m=features;oof=np.full((len(tr),4),np.nan)
    for a,b in folds(y[tr],groups):
        if winner['kind']=='vector':
            head=build(winner['head'],winner['selected']['config']);head.fit(x[tr[a]],y[tr[a]])
            if hasattr(head,'predict_proba'):raw=np.log(np.clip(head.predict_proba(x[tr[b]]),1e-12,1))
            else:raw=head.decision_function(x[tr[b]])
            for j,c in enumerate(head.classes_):oof[b,int(c)]=raw[:,j]
        else:
            cfg=max(winner['grid'],key=lambda r:r['mean']);epochs=max(1,int(np.median(cfg['epochs'])))
            head,_,_=fit_token(t[tr[a]],m[tr[a]],y[tr[a]],winner['head'],cfg['lr'],epochs=epochs)
            oof[b]=np.log(np.clip(token_predict(head,t[tr[b]],m[tr[b]]),1e-12,1));del head
    if not np.isfinite(oof).all():raise ValueError('incomplete calibration OOF predictions')
    result=fit_temperature(oof,y[tr]);write_json(output,result);return result

def probability_metrics(p,y):
    p=np.asarray(p);y=np.asarray(y);one=np.eye(4)[y]
    ece=0.
    for lower in np.linspace(0,.9,10):
        select=(p.max(1)>=lower)&(p.max(1)<lower+.1+(1e-12 if lower>=.9 else 0))
        if select.any():ece+=select.mean()*abs((p.argmax(1)[select]==y[select]).mean()-p.max(1)[select].mean())
    return {'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
        'brier_multiclass':float(np.square(p-one).sum(1).mean()),'ece_10_equal_width_bins':float(ece),
        'rps':float(np.square(p.cumsum(1)[:,:-1]-one.cumsum(1)[:,:-1]).mean())}

def profile_pipeline(predictor,rows,output,clip_path=None,yolo_path=None,mission=None):
    """Profile the actually instantiated image pipeline, costs never summed across hosts."""
    import psutil
    from PIL import Image
    clip=None;detector=None
    if clip_path is not None:
        from parameters.minimal_vision import MinimalVisionProvider
        clip=MinimalVisionProvider(clip_path,[mission],device=predictor.device);clip.provider.load();clip.context_embedding(mission)
    if yolo_path is not None:
        from parameters.scene_measurements import YOLOWorldSceneProvider
        detector=YOLOWorldSceneProvider(yolo_path,device=predictor.device);detector.load()
    samples=[]
    for row in rows[:10]:
        with Image.open(row['image_path']) as image:samples.append(image.copy())
    if not samples:raise ValueError('no profile images')
    memory=None;extractor=None;next_sequence=128
    if clip is not None and detector is not None:
        from parameters.extractor import SenderMemory
        from parameters.minimal_parameters import FinalParameterExtractor
        extractor=FinalParameterExtractor(provider=clip,scene_provider=detector,spi_image_provider=predictor)
        memory=SenderMemory('profile_ue',mission.mission_id,capacity=128)
        seed_evidence=clip.extract_batch(samples,[mission]*len(samples))
        for i in range(128):memory.commit(seed_evidence[i%len(seed_evidence)],i)
    def call(image):
        nonlocal next_sequence
        if extractor is not None:
            meta={'source_id':'profile_ue','source_numeric_id':1,'session_id':1,'message_id':next_sequence,
                  'profile_id':1,'sequence':next_sequence,'capture_time_ms':None,'timestamp_trusted':False}
            result=extractor.transmit_batch([image],[mission],[meta],[memory])[0];next_sequence+=1
            if result[0]['scene'].status!='ok':raise RuntimeError('profile scene extraction failed: '+str(result[0]['scene'].reason))
            return result
        spi=predictor.predict_image(image)
        if clip:
            clip.extract_batch([image],[mission])
        if detector:
            scene=detector.measure(image)
            if scene.status!='ok':raise RuntimeError('profile detector unavailable: '+str(scene.reason))
        # Header serialization included; history setup is not capture-time truth.
        from parameters.minimal_parameters import MinimalHeader,ExtendedParameterHeader
        from parameters.scene_measurements import SceneMeasurements
        ExtendedParameterHeader(MinimalHeader(1,1,1,1,spi['scheduling_class']),SceneMeasurements.unavailable('profile')).to_bytes()
        return spi
    def sync():
        if predictor.device.startswith('cuda'):torch.cuda.synchronize(predictor.device)
    for i in range(10):call(samples[i%len(samples)])
    measurements=[]
    repeats=3 if predictor.device=='cpu' else 10
    for image in samples:
        for _ in range(repeats):
            sync();started=time.perf_counter();call(image);sync();measurements.append((time.perf_counter()-started)*1000)
    # Resource collection is a separate pass; does not perturb latency samples.
    rss=[]
    if predictor.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(predictor.device)
    for image in samples:rss.append(psutil.Process().memory_info().rss);call(image);sync();rss.append(psutil.Process().memory_info().rss)
    result={'device':predictor.device,'p50_ms':float(np.median(measurements)),'p95_ms':float(np.percentile(measurements,95)),
        'samples_ms':measurements,'warmup_calls':10,'cpu_threads':torch.get_num_threads(),'batch_size':1,
        'scope':'SPI image encoder/head'+(' + CLIP relevance' if clip else '')+(' + YOLO scene measurements' if detector else '')+' + header serialization',
        'process_rss_observed_max_bytes':max(rss),'rss_scope':'absolute shared process; sampled, not guaranteed transient peak',
        'novelty_scope':'128-entry engineering history seeded by repeated real development images; query/commit measured; not a novelty-accuracy result',
        'kb_capacity':128 if memory is not None else None,'kb_dot_product_flops_proxy':2*128*512 if memory is not None else None}
    result['components']={'spi':True,'clip_relevance':clip is not None,'scene_detector':detector is not None,'header':True}
    if extractor is not None:
        import copy
        from parameters.minimal_parameters import FinalParameterExtractor,ExtendedParameterHeader
        probes=[]
        for image in samples:
            bundle,wire=call(image);prediction=bundle.get('spi_audit',bundle['spi'])
            class ResearchOnlySPIBridge:
                def predict_image(self,image):return {**prediction,'runtime_promotion_allowed':True}
            # Controlled local protocol check, not a promotion or transmission.
            bridge=FinalParameterExtractor(spi_image_provider=ResearchOnlySPIBridge())
            preview=bridge._apply_image_spi(copy.deepcopy(bundle),image)
            raw=preview['header'].to_bytes();decoded=ExtendedParameterHeader.from_bytes(raw)
            probes.append({'level':prediction['level'],'expected_class':prediction['scheduling_class'],
                'decoded_class':decoded.core.scheduling_class,'bytes':len(raw),
                'passed':decoded.core.scheduling_class==prediction['scheduling_class'] and len(raw)==35})
        result['wire_integration_probe']={'passed':all(p['passed'] for p in probes),'cases':probes,
            'fixture_only':True,'runtime_promotion_changed':False,'scope':'actual model output through production SPI-to-wire adapter; preview headers not transmitted'}
    if predictor.device.startswith('cuda'):result.update(cuda_peak_allocated=torch.cuda.max_memory_allocated(predictor.device),cuda_peak_reserved=torch.cuda.max_memory_reserved(predictor.device))
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as trace:call(samples[0]);sync()
        events=trace.key_averages();result['flops']={'supported_forward_flops':sum(e.flops for e in events),
            'status':'partial supported operators, 2 FLOPs/MAC; Python, sklearn, image processing and unsupported kernels not included',
            'operators_without_counts':[e.key for e in events if not e.flops]}
    except Exception as error:result['flops']={'supported_forward_flops':None,'status':'unsupported profiling','error':str(error)}
    write_json(output,result)
    if clip:clip.close()
    return result

def finalize(winner,rows,labels,comparison,output,diagnostic_review=None,clip_path=None,yolo_path=None,mission=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=ResumeStore(out/'checkpoints')
    encoder_dir=Path(comparison)/winner['encoder'];identity=json.loads((encoder_dir/'identity.json').read_text())
    with np.load(encoder_dir/'features.npz',allow_pickle=False) as c:x,t,m=c['embeddings'],c['tokens'],c['mask']
    spec={'model_sha256':file_hash(winner['artifact']),'identity':identity,'train_labels':[{k:r.get(k) for k in ('image_id','group','split','level','available')} for r in labels if r['split']=='train']}
    spec['feature_sha256']=file_hash(encoder_dir/'features.npz')
    spec['implementation']=implementation_identity(calibration,fit_temperature,fit_token,token_predict,build,SPIImagePredictor)
    done=store.read('calibration',spec)
    if done is None:
        directory=store.directory('calibration',spec);cal=calibration(winner,rows,labels,(x,t,m),directory/'calibration.json')
        done=store.complete('calibration',spec,[directory/'calibration.json'],cal)
    cal=done['value'];write_json(out/'calibration.json',cal)
    dev=[i for i,r in enumerate(rows) if r['split']=='validation' and labels[i].get('available')]
    train=[i for i,r in enumerate(rows) if r['split']=='train' and labels[i].get('available')]
    profiles={};parity=[];dev_predictions=[]
    for device in ('cuda:0','cpu'):
        start=time.perf_counter();model=SPIImagePredictor(winner,identity,device,cal['temperature']);load_seconds=time.perf_counter()-start
        if device=='cuda:0':
            for i in dev:
                prediction=model.predict_image(rows[i]['image_path']);dev_predictions.append(prediction)
            for i in dev[:10]:
                with torch.inference_mode():cached_logits=model.logits_from_features(torch.tensor(x[i:i+1],device=device),torch.tensor(t[i:i+1],device=device),torch.tensor(m[i:i+1],device=device))
                cached=probabilities(cached_logits,cal['temperature'])[0];actual=model.predict_image(rows[i]['image_path'])['probabilities']
                parity.append(float(np.max(np.abs(cached-np.array(actual)))))
        import platform,importlib.metadata
        def producer_hashes(path):
            if path is None:return None
            path=Path(path)
            if path.is_file():return {path.name:file_hash(path)}
            return {p.name:file_hash(p) for p in sorted(path.iterdir()) if p.suffix in ('.json','.safetensors','.bin','.pt')}
        profile_spec={**spec,'device':device,'clip_weights':producer_hashes(clip_path),'yolo_weights':producer_hashes(yolo_path),
            'temperature':cal['temperature'],'images':[rows[i]['image_sha256'] for i in dev[:10]],
            'hardware':{'host':platform.node(),'platform':platform.platform(),'processor':platform.processor(),
                        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            'implementation':implementation_identity(profile_pipeline,SPIImagePredictor),
            'mission':mission.paragraph if mission else None,'profile_protocol':'batch1_cpu30_gpu100_v1'}
        source=Path(__file__).resolve().parents[2]/'parameters'
        profile_spec['runtime_modules']={name:file_hash(source/name) for name in ('minimal_parameters.py','minimal_vision.py','vision.py',
            'scene_measurements.py','extractor.py','model_settings.py','pilot_parameters.py')}
        prior=store.read('profile',profile_spec)
        if prior is None:
            directory=store.directory('profile',profile_spec)
            p=profile_pipeline(model,[rows[i] for i in dev],directory/'profile.json',clip_path,yolo_path,mission)
            p['cold_load_seconds']=load_seconds;atomic_json(directory/'profile.json',p)
            prior=store.complete('profile',profile_spec,[directory/'profile.json'],p)
        profiles[device]=prior['value'];del model;gc.collect();torch.cuda.empty_cache()
    write_json(out/'profiles.json',profiles)
    y=np.array([LEVELS.index(labels[i]['level']) for i in dev]);p=np.array([v['probabilities'] for v in dev_predictions])
    quality={**score(y,p.argmax(1)),**probability_metrics(p,y)};write_json(out/'development.json',quality)
    support={}
    for role,ix,minimum,groups_min in [('train',train,12,6),('validation',dev,5,3)]:
        support[role]={}
        for level in LEVELS:
            members=[i for i in ix if labels[i]['level']==level];n_groups=len({rows[i]['group'] for i in members})
            support[role][level]={'rows':len(members),'groups':n_groups,'passes':len(members)>=minimum and n_groups>=groups_min}
    audit=diagnostic_review or {'accepted':False,'unresolved_decisive_contradictions':None,'reason':'No accepted independent diagnostic receipt provided'}
    checks={'all_class_support':all(v['passes'] for role in support.values() for v in role.values()),
        'per_class_recall':min(quality['recall'])>=.70,'export_parity':bool(parity) and max(parity)<=1e-4,
        'runtime_integration':all(p.get('wire_integration_probe',{}).get('passed') is True for p in profiles.values()),
        'diagnostic_review':audit.get('accepted') is True and audit.get('unresolved_decisive_contradictions')==0}
    receipt={'schema':'medic_spi_release_v2','checks':checks,'runtime_promotion_allowed':False,'final_test_gate_passed':False,
        'class_support':support,'max_parity_error':max(parity) if parity else None,'diagnostic_review':audit,
        'model_sha256':spec['model_sha256'],'identity':identity,'temperature':cal['temperature'],
        'reference':'teacher agreement','independent_urgency_validated':False}
    write_json(out/'readiness.json',receipt)
    # Final test is only opened when the complete pipeline is frozen. No future tuning.
    acceptance={'calibration_frozen':True,'export_verified':checks['export_parity'],'runtime_integration_verified':checks['runtime_integration'],
                'complete_pipeline_profiled':all('p50_ms' in value and all(value.get('components',{}).values())
                    and set(value.get('components',{}))=={'spi','clip_relevance','scene_detector','header'} for value in profiles.values())}
    if all(acceptance.values()):
        test=[i for i,r in enumerate(rows) if r['split']=='test' and labels[i].get('available')]
        model=SPIImagePredictor(winner,identity,'cuda:0',cal['temperature']);pred=[model.predict_image(rows[i]['image_path']) for i in test]
        py=np.array([LEVELS.index(labels[i]['level']) for i in test]);pp=np.array([v['probabilities'] for v in pred])
        metrics={**score(py,pp.argmax(1)),**probability_metrics(pp,py)} if test else {'status':'undefined_no_test_labels'}
        baseline_path=Path(comparison)/'clip_b32/logistic.joblib'
        baseline=None;interval=None
        if test and baseline_path.exists():
            import joblib
            with np.load(Path(comparison)/'clip_b32/features.npz',allow_pickle=False) as values:bx=values['embeddings'][test]
            baseline_prediction=joblib.load(baseline_path).predict(bx);baseline=score(py,baseline_prediction)
            groups=np.array([rows[i]['group'] for i in test]);unique=np.unique(groups);rng=np.random.default_rng(17);effects=[]
            for _ in range(1000):
                sampled=rng.choice(unique,len(unique),replace=True);ix=np.concatenate([np.where(groups==g)[0] for g in sampled])
                effects.append(score(py[ix],pp.argmax(1)[ix])['macro_f1']-score(py[ix],baseline_prediction[ix])['macro_f1'])
            interval={'metric':'paired macro-F1 difference versus predeclared CLIP logistic baseline','mean_observed':metrics['macro_f1']-baseline['macro_f1'],
                'percentile_95':np.percentile(effects,[2.5,97.5]).tolist(),'bootstrap_groups':len(unique),'replicates':1000,
                'grouping':'known duplicate/parent components; event independence unverified','reference':'teacher agreement only'}
        write_json(out/'locked_test.json',{'metrics':metrics,'predictions':pred,'ids':[rows[i]['image_id'] for i in test],
            'scope':'classification-heldout teacher targets; prior detector-study exposure disclosed','acceptance':acceptance,
            'baseline_clip_logistic':baseline,'paired_uncertainty':interval})
        test_support={level:sum(labels[i]['level']==level for i in test) for level in LEVELS}
        receipt['final_test_class_support']=test_support
        receipt['final_test_gate_passed']=bool(test) and all(n>=5 for n in test_support.values()) and min(metrics['recall'])>=.70
        receipt['runtime_promotion_allowed']=all(checks.values()) and receipt['final_test_gate_passed']
        write_json(out/'readiness.json',receipt);del model;torch.cuda.empty_cache()
    return receipt
