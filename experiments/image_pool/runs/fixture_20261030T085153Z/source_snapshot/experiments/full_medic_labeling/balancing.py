"""Resample an already isolated fit fold, never a complete dataset."""
import numpy as np

def resample_fit_fold(targets,method,seed=29):
    y=np.asarray(targets)
    if y.ndim!=1 or set(y.tolist())!=set(range(4)):
        raise ValueError('four-class fit fold required')
    if method in ('original','class_weighted'):return np.arange(len(y))
    if method not in ('random_oversampling','moderate_undersampling'):
        raise ValueError('unknown balance method')
    rng=np.random.default_rng(seed);members=[np.flatnonzero(y==k) for k in range(4)]
    counts=[len(v) for v in members]
    if method=='random_oversampling':
        # Retain every original row, then repeat minority examples.
        indices=np.concatenate([np.concatenate([v,rng.choice(v,max(counts)-len(v),replace=True)]) for v in members])
    else:
        cap=max(1,int(np.median(counts)))
        indices=np.concatenate([rng.choice(v,min(len(v),cap),replace=False) for v in members])
    return rng.permutation(indices)
