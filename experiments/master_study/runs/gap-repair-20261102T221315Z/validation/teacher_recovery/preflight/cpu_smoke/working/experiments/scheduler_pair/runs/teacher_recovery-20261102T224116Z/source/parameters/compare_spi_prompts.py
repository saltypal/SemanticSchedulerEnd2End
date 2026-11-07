"""Compare teacher prompt versions on matched exposed development images."""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
from .artifacts import Run,write_json


def compare(old_path,new_path,output):
    run=Run(output,{'seconds':120,'scope':'paired teacher outputs on exposed development images'})
    try:
        old=json.loads(Path(old_path).read_text(encoding='utf-8'))
        new=json.loads(Path(new_path).read_text(encoding='utf-8'))
        lookup={r['image_id']:r for r in old if r.get('split')!='test'}
        pairs=[]
        for row in new:
            if row.get('split')=='test' or row['image_id'] not in lookup:continue
            previous=lookup[row['image_id']]
            if row['image_sha256']!=previous['image_sha256']:raise ValueError('matched teacher image bytes changed')
            pairs.append({'image_id':row['image_id'],'source':row['source'],'split':row['split'],
                'old_level':previous.get('level'),'new_level':row.get('level'),
                'old_evidence':previous.get('evidence'),'new_evidence':row.get('evidence')})
        matrix=Counter((r['old_level'],r['new_level']) for r in pairs)
        examples=[r for r in pairs if r['old_level']!=r['new_level']][:20]
        labels=('Low','Medium','High','Critical')
        result={'pairs':len(pairs),'old_distribution':{name:sum(r['old_level']==name for r in pairs) for name in labels},
            'new_distribution':{name:sum(r['new_level']==name for r in pairs) for name in labels},
            'new_unavailable':sum(r['new_level'] is None for r in pairs),
            'transitions':[{'old':key[0],'new':key[1],'count':value} for key,value in sorted(matrix.items(),key=lambda x:str(x[0]))],
            'examples':examples,'scope':'same exposed image bytes, two VLM prompts; teacher agreement, not independent urgency truth',
            'preflight_first24_source_counts':dict(Counter(r['source'] for r in pairs[:24]))}
        write_json(run.path/'metrics/prompt_comparison.json',result)
        write_json(run.path/'status.json',{'status':'complete','pairs':len(pairs)})
        print('Compared',len(pairs),'matched development teacher responses',flush=True)
        return result
    except Exception as error:
        write_json(run.path/'status.json',{'status':'failed','reason':type(error).__name__+': '+str(error)})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(compare(a.old,a.new,a.output),indent=2))
