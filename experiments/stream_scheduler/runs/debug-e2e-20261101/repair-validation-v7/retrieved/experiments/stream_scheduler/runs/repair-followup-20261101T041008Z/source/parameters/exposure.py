"""Exclude known development exposure, including pixel-near duplicate components."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from PIL import Image
from .artifacts import write_json
from .contracts import digest


def fingerprint(path):
    with Image.open(path) as image:
        array=np.asarray(image.convert('L').resize((9,8)),dtype=float)
    bits=array[:,1:]>array[:,:-1]
    return {'dhash':str(sum(int(value)<<i for i,value in enumerate(bits.ravel()))),
            'std':float(array.std())}


def build_ledger(repository,output):
    repository=Path(repository).resolve();original=repository.parent/'SemanticSchedulerEnd2End'
    inputs=[];entries=[];missing=[]
    for manifest in sorted((original/'results/ladi-curation').glob('**/*.csv')):
        with manifest.open(encoding='utf-8-sig',newline='') as handle:
            rows=list(csv.DictReader(handle))
        if not rows or 'source_sha256' not in rows[0] or 'image_path' not in rows[0]:continue
        inputs.append({'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
        for row in rows:
            path=manifest.parent/row['image_path'].replace('\\','/')
            entry={'image_id':row['source_sha256'],'source':'ladi','source_row':int(row['source_row_index'])}
            if path.is_file():entry.update(fingerprint(path))
            else:missing.append(str(path))
            entries.append(entry)
    pilot=repository/'parameters/results/local/20261025T082051Z/data/records.json'
    if not pilot.is_file() or not inputs:raise ValueError('known development source manifests missing')
    inputs.append({'path':str(pilot),'sha256':hashlib.sha256(pilot.read_bytes()).hexdigest()})
    for row in json.loads(pilot.read_text()):
        entry={key:row.get(key) for key in ('image_id','source','source_row')}
        path=Path(row['image_path'])
        if not path.is_file():raise ValueError('pilot image unavailable for exposure fingerprint')
        entry.update(fingerprint(path));entries.append(entry)
    historical=repository/'results/disaster_decoder_v3/20261022_stageA_v3/extraction/features.jsonl'
    if historical.is_file():
        inputs.append({'path':str(historical),'sha256':hashlib.sha256(historical.read_bytes()).hexdigest()})
        for line in historical.read_text().splitlines():
            row=json.loads(line)
            for key in ('image_sha256','image_id'):
                if row.get(key):entries.append({'image_id':row[key]})
    unique={digest(entry):entry for entry in entries}
    payload={'schema':'better_exposure_v1','scope':'known local LADI v1/v2 manifests, V3 features and 180-parent pilot; not all external pretraining',
             'inputs':inputs,'entries':list(unique.values()),'missing_local_images':missing,
             'near_duplicate_rule':'dHash Hamming<=2 with both grayscale std>10; propagate through current duplicate components'}
    payload['sha256']=digest(payload);write_json(output,payload)
    return payload


def apply_ledger(records,ledger):
    payload={k:v for k,v in ledger.items() if k!='sha256'}
    if ledger.get('schema')!='better_exposure_v1' or ledger.get('sha256')!=digest(payload):
        raise ValueError('invalid exposure ledger identity')
    entries=ledger['entries'];ids={e['image_id'] for e in entries}
    rows={(e.get('source'),e.get('source_row')) for e in entries if e.get('source_row') is not None}
    hashes=[(int(e['dhash']),e['std']) for e in entries if 'dhash' in e]
    exposed=set();reasons={}
    for row in records:
        matched=[]
        if row['image_id'] in ids:matched.append('content/source identity')
        if (row['source'],row.get('source_row')) in rows:matched.append('published source row')
        fp=fingerprint(row['image_path'])
        if any(min(fp['std'],std)>10 and (int(fp['dhash'])^h).bit_count()<=2 for h,std in hashes):
            matched.append('near-pixel match')
        if matched:exposed.add(row['group']);reasons[row['image_id']]=matched
    for row in records:row['development_exposed']=row['group'] in exposed
    return {'ledger_sha256':ledger['sha256'],'direct_matches':reasons,'excluded_components':len(exposed),
            'excluded_parents':sum(r['development_exposed'] for r in records),'scope':ledger['scope']}
