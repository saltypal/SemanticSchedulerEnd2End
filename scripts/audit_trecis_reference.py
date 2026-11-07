"""Bounded official priority-label preflight. No tweet text, media or model inference."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import platform
import urllib.request

REVISION='f480bde06dbd7bdd64bd84595cef79bc556c7666'
URL=f'https://raw.githubusercontent.com/trecis/trecis.github.io/{REVISION}/data/2021a/TRECIS-2021A-crisis.labels.prelim.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    print('START official TREC-IS label schema check',URL,flush=True)
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'SemanticScheduler-reference-audit'}),timeout=40) as response:
        raw=response.read(1_000_001)
    if len(raw)>1_000_000:raise ValueError('annotation preflight byte cap exceeded')
    data=json.loads(raw);records=[];events=[]
    def visit(value,event=None):
        if isinstance(value,dict):
            if 'postID' in value and 'postPriority' in value:
                records.append((event,value));return
            event=value.get('eventid',value.get('eventID',event))
            for key,item in value.items():
                visit(item,key if isinstance(item,dict) and 'postID' not in item and event is None else event)
        elif isinstance(value,list):
            for item in value:visit(item,event)
    visit(data)
    if not records:raise ValueError('official label schema not recognized; no records invented')
    counts=Counter(str(row['postPriority']) for _,row in records)
    if set(counts)-{'Critical','High','Medium','Low'}:raise ValueError('unknown official priority')
    keys=sorted(set().union(*(row.keys() for _,row in records)))
    receipt={'url':URL,'revision':REVISION,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'reference_records':len(records),
        'priority_counts':dict(counts),'record_fields':keys,'top_level_type':type(data).__name__,
        'display_alias':{'Low':'Normal'},'original_labels_preserved':True,
        'priority_target':'post priority, not independently verified image-only urgency',
        'preliminary_labels':True,'numeric_deadlines_available':any('deadline' in key.lower() for key in keys),
        'image_priority_field_available':any('image' in key.lower() and 'priority' in key.lower() for key in keys),
        'image_to_post_join_verified':False,'image_bytes_retrieved':False,
        'runtime':platform.python_version(),'scope':'One pinned preliminary annotation file; counts are not full TREC-IS dataset statistics'}
    (a.output/'reference_labels.prelim.json').write_bytes(raw)
    (a.output/'reference_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('END TREC-IS schema check',json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
