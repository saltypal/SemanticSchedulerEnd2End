"""Provisional CLIP SPI: development-only selection, untouched test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

LEVELS = ['Low', 'Medium', 'High', 'Critical']

def train(features, records, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records_path = Path(records)
    records = json.loads(records_path.read_text())
    with np.load(features, allow_pickle=False) as bank:
        ids = bank['image_ids'].tolist()
        x = bank['embeddings'].copy()
    if len(ids) != len(set(ids)) or not np.isfinite(x).all():
        raise ValueError('invalid feature bank')
    lookup = {r['image_id']: r for r in records}
    if set(ids) != set(lookup) or len(lookup) != len(records):
        raise ValueError('feature/label identities differ')
    rows = [lookup[i] for i in ids]
    y = np.array([LEVELS.index(r['level']) for r in rows])
    splits = np.array([r['split'] for r in rows])
    masks = {s: splits == s for s in ('train', 'validation', 'test')}
    groups = {s: {r['group'] for r in rows if r['split'] == s} for s in masks}
    if any(groups[a] & groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('group leakage')
    if any(set(y[m]) != set(range(4)) for m in masks.values()):
        raise ValueError('four classes required in every role')
    candidates = []
    for c in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=29)
        model.fit(x[masks['train']], y[masks['train']])
        value = f1_score(y[masks['validation']], model.predict(x[masks['validation']]), average='macro')
        print('DEVELOPMENT', c, value, 'iterations', model.n_iter_.tolist(), flush=True)
        if np.any(model.n_iter_ >= 2000): raise RuntimeError('classifier did not converge')
        candidates.append((value, -c, model))
    value, _, model = max(candidates, key=lambda row: row[:2])
    report = classification_report(y[masks['test']], model.predict(x[masks['test']]),
                                   labels=list(range(4)), target_names=LEVELS, output_dict=True, zero_division=0)
    joblib.dump(model, output/'head.joblib')
    portable = {'coef':model.coef_.tolist(), 'intercept':model.intercept_.tolist(),
                'classes':model.classes_.tolist(), 'n_iter':model.n_iter_.tolist(), 'C':model.C}
    (output/'head.json').write_text(json.dumps(portable))
    receipt = {'scope':'provisional CLIP classifier; not completed full encoder study',
        'selection':'validation macro F1 only; tie prefers smaller C', 'development_macro_f1':value,
        'test':report, 'counts':{s:int(m.sum()) for s,m in masks.items()},
        'training_ids':[ids[i] for i in np.flatnonzero(masks['train'])],
        'record_sha256':hashlib.sha256(records_path.read_bytes()).hexdigest(),
        'labels':'mixed-protocol VLM pseudo-labels; not independent urgency truth'}
    (output/'training_receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--records',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();train(a.features,a.records,a.output)
