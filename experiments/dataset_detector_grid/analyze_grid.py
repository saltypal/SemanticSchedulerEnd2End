"""Analyze saved grid behavior. Does not infer accuracy without box truth."""
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
from parameters.pilot_parameters import VOCABULARY,canonical_label
from parameters.scene_measurements import measure_detections

def summarize(setting,path):
    rows=setting['rows'];times=np.array([r['seconds'] for r in rows]);counts=[];categories=[];cover=[];largest=[];ambiguous=0;boxes=0;invalid=0;nested=0
    for row in rows:
        audit=row['audit'];detections=audit['detections'];boxes+=len(detections)
        valid=[d for d in detections if canonical_label(d['label']) in VOCABULARY]
        ambiguous+=len(detections)-len(valid);invalid+=audit.get('rejected_boxes',0)
        scene=measure_detections(valid,audit['width'],audit['height'])
        counts.append(scene.instances);categories.append(scene.categories);cover.append(scene.union_box_coverage);largest.append(scene.largest_box_fraction or 0.)
        for a,b in itertools.combinations(valid,2):
            if canonical_label(a['label'])!=canonical_label(b['label']):continue
            x,y=a['box'],b['box'];intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
            small=min((x[2]-x[0])*(x[3]-x[1]),(y[2]-y[0])*(y[3]-y[1]))
            if small and intersection/small>.9:nested+=1
    return {'file':path.name,**setting['settings'],'images':len(rows),'complete':len(rows)==300,
        'p50_ms':float(np.median(times)*1000),'p95_ms':float(np.percentile(times,95)*1000),'mean_ms':float(times.mean()*1000),
        'mean_instances_raw':float(np.mean([r['values']['instances'] for r in rows])),
        'mean_instances_exact_vocab':float(np.mean(counts)),'median_instances_exact_vocab':float(np.median(counts)),
        'mean_categories_raw':float(np.mean([r['values']['categories'] for r in rows])),
        'mean_categories_exact_vocab':float(np.mean(categories)),'zero_detection_fraction':float(np.mean(np.array(counts)==0)),
        'mean_union_coverage':float(np.mean(cover)),'mean_largest_fraction':float(np.mean(largest)),
        'near_full_frame_fraction':float(np.mean(np.array(largest)>=.95)),
        'noncanonical_phrase_boxes':ambiguous,'total_raw_boxes':boxes,'noncanonical_phrase_fraction':ambiguous/max(boxes,1),
        'same_category_containment_pairs':nested,'invalid_boxes':invalid,
        'saturated_images':sum(r.get('saturated',False) for r in rows)}

def run(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);all_summaries={};details={}
    for model in ('yolo_world','grounding_dino'):
        paths=sorted(source.glob('**/'+model+'/setting_*.json'))
        if not paths:raise ValueError('missing grid '+model)
        if len({p.parent for p in paths})!=1:raise ValueError('ambiguous grid versions')
        summaries=[];lookup={}
        for path in paths:
            setting=json.loads(path.read_text());summary=summarize(setting,path);summaries.append(summary);lookup[path.name]=setting
        all_summaries[model]=summaries;details[model]=lookup
        keys=list(dict.fromkeys(k for r in summaries for k in r))
        with (out/(model+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(summaries)
    # Pairwise changes among complete settings, matching image identities.
    differences={}
    for model,settings in details.items():
        pairs=[]
        for (an,a),(bn,b) in itertools.combinations(settings.items(),2):
            if len(a['rows'])!=300 or len(b['rows'])!=300:continue
            changes=[k for k in a['settings'] if a['settings'][k]!=b['settings'][k]]
            if len(changes)!=1:continue
            aa={r['image_id']:r for r in a['rows']};bb={r['image_id']:r for r in b['rows']}
            if set(aa)!=set(bb):raise ValueError('grid image identities differ')
            values={}
            for field in ('instances','categories','largest_box_fraction','union_box_coverage'):
                delta=np.array([abs((aa[i]['values'][field] or 0)-(bb[i]['values'][field] or 0)) for i in aa])
                values[field]={'mean_absolute_change':float(delta.mean()),'fraction_changed':float(np.mean(delta>1e-8))}
            pairs.append({'a':an,'b':bn,'changed_setting':changes[0],'a_settings':a['settings'],'b_settings':b['settings'],'differences':values})
        differences[model]=pairs
    (out/'summary.json').write_text(json.dumps({'models':all_summaries,'single_factor_changes':differences,
        'accuracy':'not measured: no reference boxes','exact_vocab_scope':list(VOCABULARY)},indent=2))
    for model,values in all_summaries.items():
        print(model,'complete',sum(r['complete'] for r in values),'total files',len(values))
        for r in values:
            settings={k:r[k] for k in ('confidence','box_threshold','text_threshold','iou','image_size','max_detections') if k in r}
            print(r['file'],settings,'n',r['images'],'ms',round(r['p50_ms'],2),'count',round(r['mean_instances_exact_vocab'],2),
                  'zero',round(r['zero_detection_fraction'],3),'coverage',round(r['mean_union_coverage'],3),'phrase',round(r['noncanonical_phrase_fraction'],3))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
