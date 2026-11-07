"""Summarize observed detector behavior without inventing detection truth."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw


def summarize(folders,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];panels=[]
    for folder in map(Path,folders):
        records=json.loads((folder/'results.json').read_text())
        rows.extend(records)
        for result in records:
            source=next((p for p in folder.glob('*.json') if p.name not in ('results.json','manifest.json') and
                         json.loads(p.read_text()).get('image_sha256')==result.get('image_sha256') and
                         json.loads(p.read_text()).get('horizontal_flip')==result.get('horizontal_flip')),None)
            if source and source.with_suffix('.png').exists():panels.append((result,source.with_suffix('.png')))
    lines=['# Local YOLO-World / Grounding DINO experiment','',
        'Real cached images; descriptive outputs only. No box ground truth, mAP or urgency accuracy.',
        'Each image ran in a separate process: load and first inference are separate, not warm steady-state timings.',
        'Same four explicit nouns: person, car, building, bridge. This is a local vocabulary variant, not the runtime16-category default.',
        '', '| Model | Image ID prefix | Flip | Status | Instances | Categories | Largest area | Union area | First extraction incl. image I/O (s) |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        v=r.get('values',{})
        lines.append('| '+' | '.join(str(x) for x in (r['model'],r['image_id'][:10],r.get('horizontal_flip',False),
            r.get('status',r['process_status']),v.get('instances'),v.get('categories'),
            v.get('largest_box_fraction'),v.get('union_box_coverage'),r.get('first_inference_seconds')))+' |')
    lines.extend(['','Mission views are computed from the same boxes, without another inference. They measure the extent of selected detected nouns, not mission urgency.',
        'Grounding DINO phrases may combine nouns; ambiguous phrases are not silently mapped into one target category.',
        'A missed detection cannot establish absence. Scores and thresholds differ by producer and are not comparable probabilities.',
        'Native preprocessing budgets differ (YOLO640 versus DINO native defaults); this is not a matched-FLOP comparison.',
        'Flip consistency is a sensitivity diagnostic, not correctness. Different counts alone cannot select the better detector.'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if panels:
        canvas=Image.new('RGB',(960,290*((len(panels)+2)//3)),(245,245,245));draw=ImageDraw.Draw(canvas)
        for i,(r,path) in enumerate(panels):
            with Image.open(path) as image:picture=image.convert('RGB');picture.thumbnail((310,250))
            x=(i%3)*320;y=(i//3)*290
            canvas.paste(picture,(x,y+30));draw.text((x+3,y+3),f"{r['model']} {r['image_id'][:8]} flip={r.get('horizontal_flip')}",fill='black')
        canvas.save(output/'overlays.png')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(summarize(a.inputs,a.output))
