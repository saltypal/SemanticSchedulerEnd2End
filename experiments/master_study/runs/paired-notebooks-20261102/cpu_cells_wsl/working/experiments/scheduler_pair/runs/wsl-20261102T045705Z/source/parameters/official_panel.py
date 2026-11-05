"""Bounded public LADI test-source panel, independent of runtime predictions."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode,urlsplit,urlunsplit
from urllib.request import Request,urlopen
from .artifacts import write_json
from .exposure import fingerprint

DATASET='MITLL/LADI-v2-dataset'


def stratum(labels):
    presence,affected=labels.get('buildings_any'),labels.get('buildings_affected_or_greater')
    if type(presence) is not bool or type(affected) is not bool:return 'unknown'
    if not presence and affected:return 'inconsistent'
    return 'absent' if not presence else 'affected' if affected else 'present_unaffected'


def near_match(a,b):
    return min(a['std'],b['std'])>10 and (int(a['dhash'])^int(b['dhash'])).bit_count()<=2


def fetch_page(offset,length=100):
    query=urlencode({'dataset':DATASET,'config':'default','split':'test','offset':offset,'length':length})
    with urlopen(Request('https://datasets-server.huggingface.co/rows?'+query,
                        headers={'User-Agent':'parameters-research'}),timeout=45) as response:
        return json.load(response)


def prepare(output,known_manifest,known_ledger,per_stratum=4,maximum_rows=200,maximum_bytes=100_000_000):
    from huggingface_hub import HfApi
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir()
    revision=HfApi().dataset_info(DATASET).sha
    known=json.loads(Path(known_manifest).read_text())
    ledger=json.loads(Path(known_ledger).read_text())
    ids={r['image_id'] for r in known}|{r['image_id'] for r in ledger['entries']}
    fingerprints=[r for r in ledger['entries'] if 'dhash' in r and 'std' in r]
    old_test=Path(__file__).parent/'results/ops/20261025T142100Z/actual_extraction_diagnostic/official_ladi_test'
    old_labels=old_test/'evaluator_labels.jsonl'
    if old_labels.is_file():
        for line in old_labels.read_text().splitlines():
            row=json.loads(line);ids.add(row['image_sha256'])
            path=old_test/row['image_path'].replace('\\','/')
            if path.is_file():fingerprints.append(fingerprint(path))
    records=[];exclusions=[];bytes_used=0;counts={k:0 for k in ('absent','present_unaffected','affected')}
    pages=[]
    try:
        for offset in range(20,20+maximum_rows,100):
            page=fetch_page(offset,min(100,20+maximum_rows-offset))
            pages.append({'offset':offset,'rows':len(page['rows']),'partial':page.get('partial'),
                          'reported_total':page.get('num_rows_total')})
            for entry in page['rows']:
                row=entry['row'];labels={k:v for k,v in row.items() if k!='image'}
                kind=stratum(labels);index=entry['row_idx']
                if kind not in counts or counts[kind]>=per_stratum:continue
                parsed=urlsplit(row['image']['src'])
                if parsed.hostname!='datasets-server.huggingface.co' or '/'+revision+'/' not in parsed.path:
                    raise ValueError('dataset viewer image revision or host differs from pinned metadata')
                url=urlunsplit((parsed.scheme,parsed.netloc,parsed.path,'',''))
                with urlopen(Request(row['image']['src'],headers={'User-Agent':'parameters-research'}),timeout=45) as response:
                    data=response.read(min(20_000_001,maximum_bytes-bytes_used+1))
                bytes_used+=len(data)
                if len(data)>20_000_000 or bytes_used>maximum_bytes:raise RuntimeError('panel download byte cap')
                identity=hashlib.sha256(data).hexdigest();path=output/'images'/(identity+'.jpg');path.write_bytes(data)
                fp=fingerprint(path)
                if identity in ids or any(near_match(fp,prior) for prior in fingerprints):
                    exclusions.append({'source_row':index,'image_id':identity,'reason':'known or selected pixel duplicate'})
                    path.unlink();continue
                record={'image_id':identity,'image_path':'images/'+path.name,'source':'ladi',
                    'group':'exact_sha256:'+identity,'native_labels':labels,'source_row':index,
                    'source_split':'test','dataset_revision':revision,'stratum':kind,'asset_url_public':url,
                    'fingerprint':fp,'role':'development_pilot','bytes':len(data)}
                records.append(record);ids.add(identity);fingerprints.append(fp);counts[kind]+=1
                write_json(output/'records.json',records)
                print('PANEL',index,kind,counts,'bytes',bytes_used,flush=True)
                if all(v==per_stratum for v in counts.values()):break
            if all(v==per_stratum for v in counts.values()):break
        complete=all(v==per_stratum for v in counts.values())
        status={'status':'complete' if complete else 'incomplete','dataset':DATASET,'revision':revision,
                'counts':counts,'download_bytes':bytes_used,'pages':pages,'exclusions':exclusions,
                'scope':'balanced diagnostic test-source panel; new to this study, not locked final evaluation',
                'producer_training_exposure':'LADI reference classifier trained official train; version overlap not independently certified',
                'group_limit':'pixel duplicate groups; event/site independence unknown'}
        write_json(output/'status.json',status)
        if not complete:raise RuntimeError('insufficient strata within declared metadata bound')
    except Exception as error:
        write_json(output/'failure.json',{'error':type(error).__name__,'message':str(error),'counts':counts,'download_bytes':bytes_used})
        raise
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--known-manifest',required=True);parser.add_argument('--known-ledger',required=True)
    args=parser.parse_args();print(prepare(args.output,args.known_manifest,args.known_ledger))
