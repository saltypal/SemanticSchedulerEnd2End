exec(open('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/stream_scheduler/runs/full-suite-20261030/validation/run_fixtures.py').read().split('tests=')[0])
e,s,pre=fixture()
for k in ('linear','mlp','classifier','deepsets','attention','temporal'):
 torch.manual_seed(121)
 m=lm.Scorer(k);a=lm.export(m,pre,'fixture',{})
 yt=m(lm.tensors(s,pre)).detach().numpy()
 yn=NumpyRanker(a).raw(s)
 print(k,'torch',yt[:3].tolist(),'numpy',yn[:3].tolist(),'max',float(np.max(np.abs(yt-yn))),flush=True)
