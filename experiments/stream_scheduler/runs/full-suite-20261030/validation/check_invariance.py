exec(open('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/stream_scheduler/runs/full-suite-20261030/validation/run_fixtures.py').read().split('tests=')[0])
e,s,pre=fixture();rows=[]
for k in ('linear','classifier','deepsets','attention','temporal'):
 torch.manual_seed(900+len(rows));m=lm.Scorer(k);a=lm.export(m,pre,'fixture',{});n=NumpyRanker(a);base=n.scores(s).reshape(len(s.ready),-1)
 ix=[2,0,3,1];sp=Snapshot(tuple(s.ready[i] for i in ix),s.processed,tuple(s.ue_windows[i] for i in ix),s.active_windows,s.now,s.attempts);perm=n.scores(sp).reshape(len(ix),-1);back=np.empty_like(perm)
 for j,i in enumerate(ix):back[i]=perm[j]
 assert np.allclose(base,back,rtol=2e-5,atol=2e-5),(k,'permutation',np.max(np.abs(base-back)))
 def cl(m):return Message(m.token+1000,m.source+50,m.session+9,m.arrival,m.sequence+100,m.features,m.header_bytes)
 sr=Snapshot(tuple(cl(m) for m in s.ready),tuple((cl(m),t) for m,t in s.processed),tuple(tuple(cl(m) for m in w) for w in s.ue_windows),tuple(tuple(cl(m) for m in w) for w in s.active_windows),s.now,s.attempts)
 assert np.allclose(base,n.scores(sr).reshape(base.shape),rtol=2e-5,atol=2e-5),(k,'relabel')
 rows.append({'kind':k,'candidate_permutation':True,'identity_relabel':True,'portable_raw_shape':list(base.shape)})
torch.manual_seed(404);a=lm.export(lm.Scorer('temporal'),pre,'fixture',{});n=NumpyRanker(a);rev=Snapshot(s.ready,s.processed,tuple(tuple(reversed(w)) for w in s.ue_windows),s.active_windows,s.now,s.attempts);delta=float(np.max(np.abs(n.scores(s)-n.scores(rev))));assert delta>1e-7,delta
print(json.dumps({'status':'PASS','rows':rows,'temporal_history_order_delta':delta,'source_sha256':sh()},allow_nan=False))
(O/'invariance_checks.json').write_text(json.dumps({'rows':rows,'temporal_history_order_delta':delta,'source_sha256':sh()},indent=2,allow_nan=False))