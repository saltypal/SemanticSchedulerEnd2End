"""Fixed nested grouped classifier matrix with separate cost and quality records."""
from pathlib import Path
import json,time,sys,subprocess,argparse,io,threading,gc,warnings
import numpy as np
import psutil,joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,confusion_matrix,precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from parameters.spi import prepare
from parameters.artifacts import write_json
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'

GRIDS={'logistic':[{'C':.1},{'C':1.}], 'ridge':[{'alpha':1.},{'alpha':10.}],
 'linear_svm':[{'C':.1},{'C':1.}], 'rbf_svm':[{'C':1.},{'C':10.}],
 'knn':[{'n_neighbors':3},{'n_neighbors':7}],
 'random_forest':[{'min_samples_leaf':2},{'min_samples_leaf':5}],
 'extra_trees':[{'min_samples_leaf':2},{'min_samples_leaf':5}],
 'hist_boost':[{'max_leaf_nodes':7},{'max_leaf_nodes':15}],
 'mlp':[{'hidden_layer_sizes':(32,)},{'hidden_layer_sizes':(128,)}]}

def build(family,config):
 if family=='logistic':
  from sklearn.linear_model import LogisticRegression
  return LogisticRegression(class_weight='balanced',max_iter=1000,random_state=17,**config)
 if family=='ridge':
  from sklearn.linear_model import RidgeClassifier
  return RidgeClassifier(class_weight='balanced',**config)
 if family=='linear_svm':
  from sklearn.svm import LinearSVC
  return LinearSVC(class_weight='balanced',max_iter=10000,random_state=17,dual='auto',**config)
 if family=='rbf_svm':
  from sklearn.svm import SVC
  return SVC(class_weight='balanced',gamma='scale',probability=False,**config)
 if family=='knn':
  from sklearn.neighbors import KNeighborsClassifier
  return KNeighborsClassifier(weights='distance',n_jobs=1,**config)
 if family in ('random_forest','extra_trees'):
  from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
  cls=RandomForestClassifier if family=='random_forest' else ExtraTreesClassifier
  return cls(n_estimators=100,class_weight='balanced',n_jobs=1,random_state=17,**config)
 if family=='hist_boost':
  from sklearn.ensemble import HistGradientBoostingClassifier
  return HistGradientBoostingClassifier(class_weight='balanced',max_iter=50,early_stopping=False,random_state=17,**config)
 if family=='mlp':
  from sklearn.neural_network import MLPClassifier
  return MLPClassifier(alpha=.01,max_iter=100,early_stopping=False,random_state=17,**config)
 raise ValueError('unsupported classifier family')

def score(y,pred):
 precision,recall,f,_=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
 return {'macro_f1':float(f1_score(y,pred,labels=range(4),average='macro',zero_division=0)),
  'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
  'precision':precision.tolist(),'recall':recall.tolist(),'f1':f.tolist(),'worst_class_recall':float(min(recall)),
  'ordinal_mae':float(np.mean(np.abs(np.array(y)-pred))),
  'confusion':confusion_matrix(y,pred,labels=range(4)).tolist()}

def folds(x,y,groups):
 result=list(StratifiedGroupKFold(3,shuffle=True,random_state=17).split(x,y,groups))
 for tr,va in result:
  if set(groups[tr])&set(groups[va]) or set(y[tr])!=set(range(4)):raise ValueError('unsupported/leaking grouped fold')
 return result

def choose(family,x,y,groups):
 rows=[]
 for config in GRIDS[family]:
  values=[]
  for tr,va in folds(x,y,groups):
   model=build(family,config);model.fit(x[tr],y[tr]);values.append(score(y[va],model.predict(x[va]))['macro_f1'])
  rows.append({'config':config,'fold_macro_f1':values,'mean_macro_f1':float(np.mean(values))})
 return max(rows,key=lambda r:r['mean_macro_f1']),rows

class MemorySample:
 def __enter__(self):
  self.stop=threading.Event();self.peak=psutil.Process().memory_info().rss
  def sample():
   while not self.stop.wait(.002):self.peak=max(self.peak,psutil.Process().memory_info().rss)
  self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
 def __exit__(self,*args):
  self.peak=max(self.peak,psutil.Process().memory_info().rss);self.stop.set();self.thread.join()

def run_family(features,labels,family,output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
 x,ids,ix,ys,identity=prepare(features,labels);lookup={r['image_id']:r for r in json.loads(Path(labels).read_text())}
 xt=x[ix['train']];yt=np.array(ys['train']);groups=np.array([lookup[ids[i]]['group'] for i in ix['train']]);outer=[];oof=np.full(len(yt),-1)
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught,MemorySample() as search_memory:
  warnings.simplefilter('always')
  for fold,(tr,va) in enumerate(folds(xt,yt,groups)):
   selected,grid=choose(family,xt[tr],yt[tr],groups[tr]);model=build(family,selected['config']);model.fit(xt[tr],yt[tr]);pred=model.predict(xt[va]);oof[va]=pred
   outer.append({'fold':fold,'selection':selected,'grid':grid,'quality':score(yt[va],pred)})
   write_json(out/'outer_progress.json',outer)
  selected,grid=choose(family,xt,yt,groups)
 search_seconds=time.monotonic()-began
 del model;gc.collect()
 with threadpool_limits(limits=1),MemorySample() as fit_memory:
  model=build(family,selected['config']);start=time.perf_counter();model.fit(xt,yt);fit_seconds=time.perf_counter()-start
  dev=score(np.array(ys['validation']),model.predict(x[ix['validation']]))
 artifact=out/'classifier.joblib';joblib.dump(model,artifact,compress=0)
 # Timing is head-only batch1; complete pipelines are measured separately.
 sample=x[ix['validation'][:1]];times=[]
 with threadpool_limits(limits=1),MemorySample() as infer_memory:
  for _ in range(10):model.predict(sample)
  for _ in range(100):
   t=time.perf_counter_ns();model.predict(sample);times.append((time.perf_counter_ns()-t)/1e6)
 state={'feature_dimension':x.shape[1],'fitted_artifact_bytes':artifact.stat().st_size,
  'support_vectors':int(len(model.support_vectors_)) if hasattr(model,'support_vectors_') else None,
  'stored_training_vectors':int(len(model._fit_X)) if hasattr(model,'_fit_X') else None,
  'training_iterations':np.asarray(model.n_iter_).tolist() if hasattr(model,'n_iter_') else None}
 flop={'training':None,'training_status':'not instrumented; sklearn solver/branch work not represented by neural FLOPs counter',
  'inference':None,'inference_status':'not instrumented; see operation proxy where available'}
 if hasattr(model,'coef_'):
  flop.update(inference_dense_matmul_flops=int(2*np.asarray(model.coef_).size),inference_status='dense dot-product arithmetic proxy only; excludes normalization, bias, activation, control and allocation')
 if family=='mlp':flop.update(inference_dense_matmul_flops=int(2*sum(v.size for v in model.coefs_)),inference_status='dense layer arithmetic proxy only; excludes activation/control')
 result={'family':family,'selected':selected,'grid':grid,'outer_folds':outer,'oof_quality':score(yt,oof),'development_quality':dev,
  'full_search_seconds':search_seconds,'final_fit_seconds':fit_seconds,'head_batch1_ms':{'p50':float(np.median(times)),'p95':float(np.percentile(times,95)),'samples':times},
  'memory_bytes':{'search_process_peak_rss':search_memory.peak,'final_fit_process_peak_rss':fit_memory.peak,'inference_process_peak_rss':infer_memory.peak,
   'scope':'same isolated family process including interpreter/libraries/data and retained allocator state; not incremental model-only RAM'},
  'complexity':state,'flops':flop,'warnings':sorted(set(str(w.message) for w in caught)),
  'data_role':'Qwen teacher agreement; all development-exposed; no final-test claim','runtime_promotion_allowed':False}
 write_json(out/'result.json',result);print('RESULT',family,result['oof_quality']['macro_f1'],flush=True)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--features');parser.add_argument('--labels');parser.add_argument('--family');parser.add_argument('--output');args=parser.parse_args()
 if args.family:run_family(args.features,args.labels,args.family,args.output);return
 run=ROOT/'experiments/spi_classifier_repair/runs'/('joint_head_pilot_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True);results=[]
 sources={'clip':BASE/'spi_training/features.npz','siglip':ROOT/'experiments/spi_classifier_repair/runs/frozen_siglip_20261028_141307/features.npz'}
 for encoder,features in sources.items():
  for family in GRIDS:
   out=run/encoder/family;command=[sys.executable,'-m','experiments.spi_classifier_repair.head_matrix','--features',str(features),'--labels',str(BASE/'spi_training/labels.json'),'--family',family,'--output',str(out)]
   out.parent.mkdir(exist_ok=True)
   try:
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
    (out.parent/(family+'.log')).write_text(completed.stdout+'\n'+completed.stderr)
    if completed.returncode:raise RuntimeError(completed.stderr[-1500:])
    result=json.loads((out/'result.json').read_text());results.append({'encoder':encoder,**result});print(encoder,family,'OOF',result['oof_quality']['macro_f1'],'dev',result['development_quality']['macro_f1'],flush=True)
   except (subprocess.TimeoutExpired,RuntimeError) as error:
    results.append({'encoder':encoder,'family':family,'status':'failed','reason':str(error)});print('FAILED',encoder,family,str(error)[:150],flush=True)
   write_json(run/'results.json',results)
 print('MATRIX_OUTPUT',run,flush=True)
if __name__=='__main__':main()
