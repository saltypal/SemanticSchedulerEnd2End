"""Fold-local prediction heads and portable contextual scorers. No vision imports."""
from copy import deepcopy
import json
import numpy as np
from .evaluation import FeatureEncoder
from .contracts import canonical,digest


class ClaimHeads:
    def __init__(self):self.models={};self.support={};self.exported={}

    def fit(self,records,evidence,claims=('building_damage','road_damage','affected_people')):
        from sklearn.linear_model import LogisticRegression
        for claim in claims:
            rows=[r for r in records if r['labels'].get(claim) in (0,1) and evidence[r['image_id']].embedding is not None]
            y=np.array([r['labels'][claim] for r in rows]);counts={str(v):int((y==v).sum()) for v in (0,1)}
            self.support[claim]=counts
            if min(counts.values())<5:continue
            x=np.stack([evidence[r['image_id']].embedding for r in rows])
            self.models[claim]=LogisticRegression(C=1.,max_iter=500,random_state=7).fit(x,y)
        return self

    def apply(self,evidence):
        output={}
        for key,row in evidence.items():
            clone=deepcopy(row)
            if row.embedding is not None:
                for claim,model in self.models.items():
                    clone.probabilities[claim]=float(model.predict_proba(np.asarray(row.embedding).reshape(1,-1))[0,1])
                for claim,model in self.exported.items():
                    logit=float(np.asarray(row.embedding)@np.asarray(model['coefficient'])+model['intercept'])
                    clone.probabilities[claim]=float(1/(1+np.exp(-np.clip(logit,-700,700))))
            output[key]=clone
        return output

    def to_dict(self):
        return {'schema':'better_claim_heads_v1','support':self.support,'calibrated':False,
                'models':{**self.exported,**{name:{'coefficient':m.coef_[0].tolist(),'intercept':float(m.intercept_[0])} for name,m in self.models.items()}}}

    @classmethod
    def from_dict(cls,value):
        if value.get('schema')!='better_claim_heads_v1':raise ValueError('invalid head artifact')
        result=cls();result.support=value['support'];result.exported=value['models']
        for model in result.exported.values():
            coefficient=np.asarray(model['coefficient'],float)
            if coefficient.ndim!=1 or not np.isfinite(coefficient).all() or not np.isfinite(model['intercept']):raise ValueError('invalid head coefficients')
        return result


class HeadedProvider:
    """Apply an already fitted head artifact to frozen visual features."""
    def __init__(self,provider,heads):self.provider,self.heads=provider,heads
    def extract_batch(self,images,missions):
        predictions=self.provider.extract_batch(images,missions)
        enriched=self.heads.apply({str(i):row for i,row in enumerate(predictions)})
        return [enriched[str(i)] for i in range(len(predictions))]


class Scorer:
    def __init__(self,artifact):
        if artifact.get('sha256')!=digest({k:v for k,v in artifact.items() if k!='sha256'}):raise ValueError('corrupt scorer artifact')
        self.artifact=artifact
        self.encoder=FeatureEncoder(artifact['fields'],artifact['medians'])
        self.mean=np.asarray(artifact['mean'],float);self.scale=np.asarray(artifact['scale'],float)
        if self.mean.shape!=(4*len(self.encoder.fields)+7,) or self.scale.shape!=self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale<=0):raise ValueError('invalid scaler')
        model=artifact['model']
        if model['kind']=='linear':
            coefficient=np.asarray(model['coefficient'],float)
            if coefficient.shape!=self.mean.shape or not np.isfinite(coefficient).all() or not np.isfinite(model['intercept']):raise ValueError('invalid linear model')
        elif model['kind']=='tree':
            from Scheduler.contextual_decoder_scheduler import validate_tree
            for tree in model['trees']:validate_tree(tree,len(self.mean))
        else:raise ValueError('unsupported model')

    @staticmethod
    def tree_predict(tree,x):
        result=[]
        for row in np.asarray(x,dtype=np.float32):
            node=0
            while tree['left'][node]!=-1:
                node=tree['left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
            result.append(tree['value'][node])
        return np.asarray(result)

    def predict(self,x):
        z=(x-self.mean)/self.scale;model=self.artifact['model']
        if model['kind']=='linear':return z@np.asarray(model['coefficient'])+model['intercept']
        return model['initial']+model['learning_rate']*sum((self.tree_predict(tree,z) for tree in model['trees']),start=np.zeros(len(z)))

    def scores(self,ready,history,attempts,total):
        return self.predict(self.encoder.matrix(ready,history,attempts,total))


def fit_scorer(decisions,groups,alpha=1.,kind='linear',depth=2,min_leaf=5,provenance=None):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    encoder=FeatureEncoder.fit(decisions,groups)
    xs=[];ys=[];weights=[]
    for d in decisions:
        x=encoder.matrix(d['ready'],d['history'],d['attempts'],d['total'])
        xs.append(x);ys.extend(d['targets']);weights.extend([d['weight']/len(x)]*len(x))
    x=np.concatenate(xs);y=np.asarray(ys);w=np.asarray(weights)
    mean=np.average(x,axis=0,weights=w);variance=np.average((x-mean)**2,axis=0,weights=w)
    scale=np.where((np.ptp(x,axis=0)==0)|(variance==0),1.,np.sqrt(variance));z=(x-mean)/scale
    if kind=='linear':
        model=Ridge(alpha=alpha).fit(z,y,sample_weight=w)
        exported={'kind':'linear','coefficient':model.coef_.tolist(),'intercept':float(model.intercept_)}
    elif kind=='tree':
        from Scheduler.training.decoder_context_training import export_tree
        model=GradientBoostingRegressor(n_estimators=50,max_depth=depth,min_samples_leaf=min_leaf,learning_rate=.05,random_state=7).fit(z,y,sample_weight=w)
        exported={'kind':'tree','initial':float(model.init_.constant_[0,0]),'learning_rate':.05,'trees':[export_tree(m[0].tree_) for m in model.estimators_]}
    else:raise ValueError('unsupported family')
    artifact={'schema':'better_scheduler_scorer_v1','groups':list(groups),'fields':list(encoder.fields),'medians':encoder.medians.tolist(),
              'mean':mean.tolist(),'scale':scale.tolist(),'model':exported,'provenance':provenance or {}}
    artifact['sha256']=digest(artifact);result=Scorer(artifact)
    expected=model.predict(z);actual=result.predict(x)
    if not np.allclose(expected,actual,rtol=1e-7,atol=1e-9):raise RuntimeError('portable inference differs from training')
    return result,{'weighted_mse':float(np.average((actual-y)**2,weights=w)),'rows':len(y),'export_max_error':float(np.max(np.abs(actual-expected)))}
