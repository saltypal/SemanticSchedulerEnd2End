"""Cloud-only full-source download attempt with disk, progress and receipts."""
import hashlib
import json
import shutil
import time
from pathlib import Path
import requests

def save(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def run(root, seconds=3600):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();results=[]
    from huggingface_hub import HfApi, hf_hub_url
    api=HfApi();info=api.dataset_info('MITLL/LADI-v2-dataset',files_metadata=True)
    sources=[{'source':'CrisisMMD','name':'CrisisMMD_v2.0.tar.gz','url':'https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz','expected_bytes':None},
             {'source':'MEDIC','name':'MEDIC.tar.gz','url':'https://crisisnlp.qcri.org/data/MEDIC/MEDIC.tar.gz','expected_bytes':None}]
    for f in info.siblings:
        if f.rfilename.startswith('data/'):
            sources.append({'source':'LADI-v2','name':f.rfilename,'url':hf_hub_url(info.id,f.rfilename,repo_type='dataset',revision=info.sha),'expected_bytes':f.size})
    save(root/'inventory.json',{'ladi_revision':info.sha,'sources':sources,'scope':'all listed LADI data files plus official complete archives; no extraction'})
    try:
        for entry in sources:
            if time.monotonic()-start>=seconds:
                results.append({**entry,'status':'time_budget_exhausted'});break
            dest=root/'downloads'/entry['source']/entry['name'];dest.parent.mkdir(parents=True,exist_ok=True)
            row={**entry,'status':'running','bytes':0};results.append(row);save(root/'downloads.json',results)
            print('DOWNLOAD START',entry['source'],entry['name'],flush=True)
            try:
                with requests.get(entry['url'],stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    expected=entry['expected_bytes'] or int(response.headers.get('Content-Length',0)) or None
                    free=shutil.disk_usage(root).free;row.update(expected_bytes=expected,free_bytes_before=free)
                    if expected and expected+3_000_000_000>free:
                        row['status']='insufficient_disk';save(root/'downloads.json',results);continue
                    digest=hashlib.sha256();last=time.monotonic();part=dest.with_suffix(dest.suffix+'.part')
                    with part.open('wb') as out:
                        for chunk in response.iter_content(4*1024*1024):
                            if not chunk:continue
                            if time.monotonic()-start>=seconds:raise TimeoutError('download execution budget reached')
                            if shutil.disk_usage(root).free<len(chunk)+3_000_000_000:raise OSError('3GB disk reserve reached')
                            out.write(chunk);digest.update(chunk);row['bytes']+=len(chunk)
                            if time.monotonic()-last>=15:
                                save(root/'downloads.json',results);print('DOWNLOAD PROGRESS',entry['source'],row['bytes'],expected,flush=True);last=time.monotonic()
                    if expected and row['bytes']!=expected:raise ValueError('download length mismatch')
                    if response.headers.get('Content-Type','').startswith('text/html'):raise ValueError('received HTML instead of dataset')
                    part.replace(dest);row.update(status='complete',sha256=digest.hexdigest(),path=str(dest))
            except Exception as error:
                row.update(status='failed',error=type(error).__name__,message=str(error))
            save(root/'downloads.json',results);print('DOWNLOAD END',row['status'],row['bytes'],flush=True)
    finally:
        save(root/'summary.json',{'seconds':time.monotonic()-start,'complete_files':sum(r['status']=='complete' for r in results),
             'planned_files':len(sources),'all_complete':len(results)==len(sources) and all(r['status']=='complete' for r in results),
             'downloaded_bytes':sum(r['bytes'] for r in results if 'bytes' in r),'free_bytes':shutil.disk_usage(root).free,
             'persistence':'Kaggle working files; durable notebook output must be verified separately'})
    return results
