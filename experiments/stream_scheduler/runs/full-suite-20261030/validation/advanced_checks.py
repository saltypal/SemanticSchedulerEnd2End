exec(open('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/stream_scheduler/runs/full-suite-20261030/validation/run_fixtures_v3.py').read().split('tests=')[0])
arms=('linear_choice','mlp_choice','delay_huber','ranknet','priority_classifier','deepsets_ranknet','set_transformer_ranknet','temporal_deepsets_ranknet')
# Pair-view lifecycle status: omitted ready token remains waiting if visible in either selected candidate history.
def pair_status():
 ms=[msg(i,0.,i%4,source=0,session=0) for i in range(3)]
 e=Environment(ep(ms,[0,1,2],[1.,1.,1.],[False]*3,10));s=e.snapshot()
 assert s.waiting_tokens==frozenset({0,1,2})
 pv=s.pair_view([0,2]);assert pv.waiting_tokens==frozenset({0,1,2}),pv.waiting_tokens
 pre=fit_normalizer([e.episode]);a=arrays(pv,pre,True)
 found=False
 for wi,w in enumerate(pv.ue_windows):
  for tj,m in enumerate(w[-3:]):
   if m.token==1:
    assert a['ue'][wi,len(w[-3:])-len(w[-3:])+tj,388]==0.
    found=True
 assert found,'omitted still-waiting candidate missing from context'
 forbidden={'reference_tiers','fact_ids','service','failures','labels','targets'}
 assert forbidden.isdisjoint(Snapshot.__dataclass_fields__)
 return {'original_waiting':[0,1,2],'pair_view_waiting':sorted(pv.waiting_tokens),'omitted_candidate_status':'waiting','snapshot_label_fields':sorted(forbidden.intersection(Snapshot.__dataclass_fields__)),'arrays_keys':sorted(a)}
def make_samples():
 e,s,pre=fixture();y=np.array([0.,.15,.55,1.]);tiers=np.array([0,1,2,3]);samples=[]
 for k,order in enumerate(([0,1,2,3],[2,3,0,1],[1,3,2,0],[3,0,1,2])):
  ss=Snapshot(tuple(s.ready[i] for i in order),s.processed,tuple(s.ue_windows[i] for i in order),s.active_windows,s.now,s.attempts)
  samples.append(lm.Sample(ss,y[list(order)],tiers[list(order)],f'fixture-episode-{k}',target_source='synthetic_fixture_D'))
 return e,s,pre,samples
def actual_lgb():
 e,s,pre,samples=make_samples();out=O/'lightgbm_actual_fixture'
 art=lm.fit_lambdamart(samples,pre,out,seed=19);r=NumpyRanker(art)
 assert art['source']['groups']==[4,4,4,4] and len(art['trees'])==100
 for sample in samples:
  z=r.scores(sample.snapshot);assert z.shape==(4,) and np.isfinite(z).all()
 return {'actual_lightgbm_fit':True,'groups':art['source']['groups'],'trees':len(art['trees']),'grade_mapping':[int(np.floor(255*v+.5)) for v in [0.,.15,.55,1.]],'portable_predictions_finite':True}
def resume_smoke():
 e,s,pre,samples=make_samples();rows=[];before=sh()
 for arm in arms:
  out=O/'supervised_resume_v1'/arm
  calls=0
  def validate(a):
   nonlocal calls
   calls+=1;r=NumpyRanker(a)
   for sample in samples:
    scores=r.scores(sample.snapshot)
    assert scores.shape==(len(sample.snapshot.ready),) and np.isfinite(scores).all(),(arm,scores.shape)
   return 0.,{'fixture_portable_shapes':True}
  a1,h1=lm.fit_supervised(samples,pre,arm,out,validate,seed=77,epochs=1,patience=8,task='D')
  state1=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);sd1={k:v.clone() for k,v in state1['model'].items()}
  a3,h3=lm.fit_supervised(samples,pre,arm,out,validate,seed=77,epochs=3,patience=8,resume=out/'resume.pt',task='D')
  state3=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
  changed=any(not torch.equal(sd1[k],state3['model'][k]) for k in sd1)
  assert len(h1)==1 and len(h3)==3 and state3['epoch']==2 and state3['updates']>=3 and changed,(arm,len(h1),len(h3),state3.get('updates'),changed)
  r=NumpyRanker(a3);assert r.kind==lm.KINDS[arm]
  rows.append({'arm':arm,'kind':r.kind,'epochs_total':state3['epoch']+1,'optimizer_updates':state3['updates'],'changed_after_resume':changed,'parity_artifact_written':(out/'parity.json').exists(),'checkpoint_reloaded':True,'source_hashes':sh()})
  print('SUPERVISED_RESUME_PASS',json.dumps(rows[-1]),flush=True)
 return {'arms':rows,'claim':'synthetic fixture objective plumbing only; no real records or scientific result'}
res={'scope':'WSL Python3.14 CPU fixture tests only; no remote/data/model inference','python':sys.version,'checks':[]}
for name,fn in [('pair_view_waiting_status_and_label_separation',pair_status),('actual_lightgbm_fixture_fit_and_export',actual_lgb),('eight_supervised_arms_three_epoch_resume_smoke',resume_smoke)]:
 b=sh();t=time.perf_counter()
 try:v=fn();st='PASS';err=None
 except BaseException:st='FAIL';v=None;err=traceback.format_exc()
 a=sh();row={'name':name,'status':st,'seconds':time.perf_counter()-t,'source_sha256_before':b,'source_sha256_after':a,'source_changed':b!=a,'result':v,'error':err};res['checks'].append(row);print('CHECK',json.dumps(row,allow_nan=False),flush=True);(O/'advanced_validation_v1.json').write_text(json.dumps(res,indent=2,allow_nan=False))
 if st=='FAIL':sys.exit(1)
(O/'advanced_validation_v1.json').write_text(json.dumps(res,indent=2,allow_nan=False))