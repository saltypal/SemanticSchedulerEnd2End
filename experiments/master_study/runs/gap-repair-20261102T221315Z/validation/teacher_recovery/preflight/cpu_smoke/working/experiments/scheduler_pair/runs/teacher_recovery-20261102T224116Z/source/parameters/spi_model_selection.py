"""Grouped four-class SPI model comparison on fixed image features and labels."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from .artifacts import Run,write_json
from .contracts import digest
from .spi import LEVELS,SPIClassifier,metrics,prepare


def support_gate(labels):
    """A small engineering support gate; not an independent accuracy claim."""
    result={}
    for split,min_rows,min_groups in [('train',12,6),('validation',5,3)]:
        result[split]={}
        for name in LEVELS:
            cohort=[r for r in labels if r['split']==split and r.get('available') is True and r.get('level')==name]
            result[split][name]={'rows':len(cohort),'groups':len({r['group'] for r in cohort}),
                                 'passes':len(cohort)>=min_rows and len({r['group'] for r in cohort})>=min_groups}
    return {'passed':all(item['passes'] for split in result.values() for item in split.values()),
            'details':result,'meaning':'engineering feasibility, not statistical power'}


def mlp_fit(x,y,balanced,seed,eval_x=None,eval_y=None,epochs=50,history=None):
    import torch
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.GELU(approximate='tanh'),
        torch.nn.Dropout(.1),torch.nn.Linear(128,4))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    counts=np.bincount(y,minlength=4)
    if np.any(counts==0):raise ValueError('fold training has a missing class')
    weight=torch.tensor(len(y)/(4*counts),dtype=torch.float32) if balanced else None
    xt=torch.as_tensor(x,dtype=torch.float32);yt=torch.as_tensor(y,dtype=torch.long)
    rng=np.random.default_rng(seed);best=float('inf');best_state=None;best_epoch=0;stale=0
    xv=None if eval_x is None else torch.as_tensor(eval_x,dtype=torch.float32)
    yv=None if eval_y is None else torch.as_tensor(eval_y,dtype=torch.long)
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(xt))
        for start in range(0,len(order),64):
            idx=order[start:start+64];optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(xt[idx]),yt[idx],weight=weight)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward();optimizer.step()
        if history is not None:
            model.eval()
            with torch.inference_mode():
                train_logits=model(xt)
                entry={'epoch':epoch+1,'train_ce':float(torch.nn.functional.cross_entropy(train_logits,yt)),
                    'train_accuracy':float((train_logits.argmax(1)==yt).float().mean())}
                if xv is not None:
                    val_logits=model(xv)
                    entry.update(validation_ce=float(torch.nn.functional.cross_entropy(val_logits,yv)),
                        validation_accuracy=float((val_logits.argmax(1)==yv).float().mean()))
            history.append(entry)
        if xv is not None:
            model.eval()
            with torch.inference_mode():value=float(torch.nn.functional.cross_entropy(model(xv),yv))
            if not np.isfinite(value):raise FloatingPointError('nonfinite validation loss')
            if value<best-1e-8:
                best=value;best_epoch=epoch+1;stale=0
                best_state={key:value.detach().clone() for key,value in model.state_dict().items()}
            else:stale+=1
            if stale>=8:break
    if xv is not None:
        model.load_state_dict(best_state)
    else:best_epoch=epochs
    model.eval();return model,best_epoch


def candidate_fit(config,x,y,seed,eval_x=None,eval_y=None,epochs=50):
    if config['family']=='linear':
        if len(set(y))!=4:raise ValueError('fold training has a missing class')
        model=LogisticRegression(C=config['C'],class_weight='balanced' if config['balanced'] else None,
                                 max_iter=1000,solver='lbfgs',random_state=seed)
        model.fit(x,y)
        if list(model.classes_)!=[0,1,2,3]:raise ValueError('unexpected model class order')
        return model,None
    return mlp_fit(x,y,config['balanced'],seed,eval_x,eval_y,epochs)


def predict(model,family,x):
    if family=='linear':return model.predict_proba(x)
    import torch
    with torch.inference_mode():return model(torch.as_tensor(x,dtype=torch.float32)).softmax(1).numpy()


def stopping_split(targets, groups, seed):
    """Choose early stopping within a fitting fold; never consume its score fold.

    If group support cannot sustain another split, use a fixed epoch budget.
    """
    if any(len(set(groups[targets == k])) < 2 for k in range(4)):
        return None
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    for fit, stop in splitter.split(np.zeros(len(targets)), targets, groups):
        if set(targets[fit]) == set(range(4)) and set(targets[stop]) == set(range(4)):
            if set(groups[fit]) & set(groups[stop]):
                raise ValueError('early stopping group leakage')
            return fit, stop
    return None


def select(features_path,labels_path,output,seconds=600):
    run=Run(output,{'seconds':seconds,'seed':17,'purpose':'four-class grouped development model selection',
                    'claim':'teacher/rubric agreement, not independent urgency accuracy'})
    try:
        labels=json.loads(Path(labels_path).read_text(encoding='utf8'))
        gate=support_gate(labels);write_json(run.path/'metrics/support_gate.json',gate)
        if not gate['passed']:raise ValueError('four-class engineering support gate failed')
        x,ids,ix,ys,identity=prepare(features_path,labels_path)
        lookup={row['image_id']:row for row in labels};groups=np.array([lookup[ids[i]]['group'] for i in ix['train']])
        xt=x[ix['train']];yt=np.asarray(ys['train'],dtype=int);xv=x[ix['validation']];yv=np.asarray(ys['validation'],dtype=int)
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=17)
        folds=list(splitter.split(xt,yt,groups))
        for train_index,val_index in folds:
            if set(groups[train_index]) & set(groups[val_index]):raise ValueError('group leakage')
            if set(yt[train_index])!=set(range(4)):raise ValueError('grouped training fold lacks a class')
        configs=[{'family':'linear','C':c,'balanced':balance} for c in (.1,1.,10.) for balance in (False,True)]
        configs += [{'family':'mlp','balanced':balance} for balance in (False,True)]
        candidates=[]
        with run.stage('grouped_selection'):
            with threadpool_limits(limits=1):
                for ci,config in enumerate(configs):
                    run.budget();fold_rows=[]
                    for fi,(tr,va) in enumerate(folds):
                        run.budget()
                        early = stopping_split(yt[tr], groups[tr], 117 + fi) if config['family']=='mlp' else None
                        epoch = None
                        if early is not None:
                            fit, stop = early
                            _, epoch = candidate_fit(config, xt[tr][fit], yt[tr][fit], 17+fi,
                                xt[tr][stop], yt[tr][stop])
                        # Refit on the whole fitting fold at the selected/fixed
                        # epoch count. The scoring fold has not affected weights.
                        model,_=candidate_fit(config,xt[tr],yt[tr],17+fi,epochs=epoch or 50)
                        p=predict(model,config['family'],xt[va]);pred=p.argmax(1)
                        row={'fold':fi,'train':len(tr),'validation':len(va),'group_count':len(set(groups[va])),
                            'macro_f1':float(f1_score(yt[va],pred,labels=list(range(4)),average='macro',zero_division=0)),
                            'critical_recall':float(np.mean(pred[yt[va]==3]==3)) if np.any(yt[va]==3) else None,
                            'validation_class_support':np.bincount(yt[va],minlength=4).tolist(),
                            'best_epoch':epoch or (50 if config['family']=='mlp' else None),
                            'early_stopping_scope':'fit-fold groups only' if early is not None else 'fixed budget or linear model'}
                        fold_rows.append(row)
                    crit=[r['critical_recall'] for r in fold_rows if r['critical_recall'] is not None]
                    candidates.append({'config':config,'folds':fold_rows,'mean_macro_f1':float(np.mean([r['macro_f1'] for r in fold_rows])),
                                       'mean_critical_recall':float(np.mean(crit)) if crit else -1.})
                    write_json(run.path/'metrics/candidates.json',candidates)
                    print('Candidate',ci+1,'/',len(configs),config,'macro-F1',candidates[-1]['mean_macro_f1'],flush=True)
            best_value=max(r['mean_macro_f1'] for r in candidates)
            best_row=max(candidates,key=lambda r:r['mean_macro_f1'])
            se=float(np.std([r['macro_f1'] for r in best_row['folds']],ddof=1)/np.sqrt(len(folds)))
            tied=[r for r in candidates if best_value-r['mean_macro_f1']<=max(se,1e-12)]
            # Do not trade a material Critical-recall loss for simplicity.
            best_critical=max(r['mean_critical_recall'] for r in tied)
            tied=[r for r in tied if best_critical-r['mean_critical_recall']<=.05+1e-12]
            selected=min(tied,key=lambda r:(r['config']['family']!='linear',r['config'].get('C',0),r['config']['balanced']))
        config=selected['config'];epochs=[r['best_epoch'] for r in selected['folds'] if r['best_epoch'] is not None]
        final_epochs=int(np.median(epochs)) if epochs else 0
        with run.stage('fit_selected'):
            with threadpool_limits(limits=1):
                model,_=candidate_fit(config,xt,yt,17,epochs=final_epochs or 50)
                reference=predict(model,config['family'],xv)
            artifact={'schema':'better_spi_linear_v1' if config['family']=='linear' else 'better_spi_head_v2',
                'levels':list(LEVELS),'unsupported_levels':[],'input_dimension':x.shape[1],
                'feature_signature':identity['feature_signature'],'rubric_sha256':identity['rubric_sha256'],
                'label_source':identity['label_source'],'target':'visible response attention under fixed rubric',
                'human_calibrated':False,'score_calibration':'uncalibrated',
                'class_order':'Low,Medium,High,Critical','training_config':config}
            if config['family']=='linear':artifact.update(w2=model.coef_.tolist(),b2=model.intercept_.tolist())
            else:artifact.update(hidden_dimension=128,w1=model[0].weight.detach().tolist(),b1=model[0].bias.detach().tolist(),
                                 w2=model[3].weight.detach().tolist(),b2=model[3].bias.detach().tolist())
            artifact['sha256']=digest(artifact);write_json(run.path/'models/spi.json',artifact)
            exported=SPIClassifier.load(run.path/'models/spi.json')
            p=exported.predict_proba(xv,identity['feature_signature'])
            difference=float(np.max(np.abs(p-reference)))
            if not np.allclose(p,reference,rtol=1e-4,atol=1e-6):raise RuntimeError('exported classifier mismatch')
            prior=np.bincount(yt,minlength=4)/len(yt)
            report={'student':metrics(p,yv),'training_prior':metrics(np.tile(prior,(len(yv),1)),yv),
                'selected':selected,'final_epochs':final_epochs,'support':identity['support'],
                'export_max_absolute_difference':difference,'test_evaluated':False,
                'selection_rule':'one-SE macro-F1; <=.05 Critical-recall loss; prefer linear/stronger regularization',
                'best_mean_macro_f1':best_value,'selection_standard_error':se,
                'claim':'development teacher agreement; no independent human urgency accuracy'}
            write_json(run.path/'metrics/validation.json',report)
            write_json(run.path/'status.json',{'status':'trained','model_sha256':artifact['sha256'],
                       'test_evaluated':False,'claim':report['claim']})
        return report
    except Exception as error:
        write_json(run.path/'status.json',{'status':'incomplete','reason':str(error),'error':type(error).__name__})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args();print(json.dumps(select(a.features,a.labels,a.output,a.seconds),indent=2))
