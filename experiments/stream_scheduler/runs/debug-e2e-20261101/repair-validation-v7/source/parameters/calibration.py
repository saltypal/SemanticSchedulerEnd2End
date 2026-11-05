"""Calibration is fitted on an inner-training subgroup, never a scored fold."""
import numpy as np
from .learning import ClaimHeads


def fit_fold_heads(records,evidence,seed=7):
    groups=sorted({r['group'] for r in records});rng=np.random.default_rng(seed);rng.shuffle(groups)
    calibration_groups=set(groups[:max(1,len(groups)//5)])
    fitting=[r for r in records if r['group'] not in calibration_groups]
    calibration=[r for r in records if r['group'] in calibration_groups]
    original=ClaimHeads().fit(fitting,evidence)
    artifact=original.to_dict();parameters={}
    for claim,model in artifact['models'].items():
        rows=[r for r in calibration if r['labels'].get(claim) in (0,1)]
        y=np.array([r['labels'][claim] for r in rows],float)
        counts=[int((y==v).sum()) for v in (0,1)]
        if min(counts,default=0)<20:
            parameters[claim]={'available':False,'reason':'fewer than20 positive or20 negative calibration observations','counts':counts}
            continue
        x=np.stack([evidence[r['image_id']].embedding for r in rows])
        logit=x@np.asarray(model['coefficient'])+model['intercept']
        grid=np.exp(np.linspace(np.log(.25),np.log(4.),41))
        losses=[float(np.mean(np.logaddexp(0,logit/t)-y*logit/t)) for t in grid]
        temperature=float(grid[int(np.argmin(losses))])
        model['coefficient']=(np.asarray(model['coefficient'])/temperature).tolist()
        model['intercept']/=temperature
        parameters[claim]={'available':True,'method':'temperature_scaling','temperature':temperature,'counts':counts}
    artifact['calibrated']=parameters
    return ClaimHeads.from_dict(artifact),parameters,{'fit_groups':sorted({r['group'] for r in fitting}),
                                                    'calibration_groups':sorted(calibration_groups)}
