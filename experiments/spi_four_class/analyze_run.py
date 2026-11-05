"""Recompute retrieved SPI metrics and draw evidence, without training models."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from parameters.artifacts import write_json
from parameters.spi import SPIClassifier, prepare, metrics
from parameters.spi_release import candidate_release


def analyze(campaign, features, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = json.loads((campaign / 'status.json').read_text())
    result = {'campaign_status': status, 'scope': 'development teacher agreement; no human urgency accuracy',
              'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'training_performed': False}
    candidates = campaign / 'metrics/teacher_candidates.json'
    if candidates.exists():
        result['teacher_candidates'] = json.loads(candidates.read_text())
    labels_file=campaign/'data/spi_labels.json'
    if labels_file.exists():
        labels=json.loads(labels_file.read_text())
        result['label_support']={role:dict(Counter(
            r.get('level') if r.get('available') is True else 'unavailable'
            for r in labels if r['split']==role)) for role in ('train','validation')}
        result['labelled_images']=len(labels)
    print('START retrieved SPI analysis',status,flush=True)
    validation = campaign / 'student/metrics/validation.json'
    if validation.exists():
        model = campaign / 'student/models/spi.json'
        x, ids, roles, y, identity = prepare(features, campaign / 'data/spi_labels.json')
        classifier = SPIClassifier.load(model)
        probabilities = classifier.predict_proba(x[roles['validation']],identity['feature_signature'])
        recomputed = metrics(probabilities,np.array(y['validation'],dtype=int))
        original = json.loads(validation.read_text())['student']
        for name in ('accuracy','log_loss','ordinal_mae','rps'):
            if not np.isclose(original[name],recomputed[name],rtol=1e-5,atol=1e-7):
                raise ValueError('retrieved metric mismatch: '+name)
        result['recomputed_validation']=recomputed
        result['metric_replay_matches']=True
        write_json(output/'validation_predictions.json',[
            {'image_id':ids[i],'target':classifier.levels[target],
             'prediction':classifier.levels[int(p.argmax())],
             'scores':dict(zip(classifier.levels,map(float,p)))}
            for i,target,p in zip(roles['validation'],y['validation'],probabilities)])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        confusion=np.array(recomputed['confusion_true_rows_predicted_columns'])
        fig,ax=plt.subplots(figsize=(6.8,5.5))
        image=ax.imshow(confusion,cmap='Blues')
        ax.set_xticks(range(4),classifier.levels);ax.set_yticks(range(4),classifier.levels)
        ax.set_xlabel('Student prediction');ax.set_ylabel('VLM pseudo-label reference')
        ax.set_title('Four-class SPI: exposed development validation')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(confusion[i,j]),ha='center',va='center')
        fig.colorbar(image,ax=ax,label='Images');fig.tight_layout()
        fig.savefig(output/'confusion_matrix.png',dpi=180);plt.close(fig)
        release=candidate_release(model,validation,ROOT/'parameters/config/parameter_release_v2.json',
                                  output/'release_candidate.json')
        result['release_checks']=release['release_checks']
        result['runtime_promotion_allowed']=release['runtime_promotion_allowed']
    else:
        result.update(metric_replay_matches=None,runtime_promotion_allowed=False,
                      reason='no trained student artifact; inspect failed upstream gate')
    write_json(output/'analysis.json',result)
    lines=['# Retrieved SPI result', '',
           '**Scope: exposed development data; VLM teacher agreement, not independent urgency accuracy.**','',
           'Campaign status: `'+status['status']+'`.','']
    if 'teacher_candidates' in result:
        lines.extend(['| Teacher | Protocol | Text macro F1 | Gate |',
                      '|---|---|---:|---|'])
        for row in result['teacher_candidates']:
            value=row.get('macro_f1')
            lines.append('| '+row['teacher']+' | '+row.get('protocol','not run')+' | '+
                ('unavailable' if value is None else f'{value:.4f}')+' | '+str(row['passed'])+' |')
        lines.extend(['','Text controls measure authored policy conformance, not visual accuracy.',''])
    if 'label_support' in result:
        lines.extend(['| Split | Low | Medium | High | Critical | Unavailable |',
                      '|---|---:|---:|---:|---:|---:|'])
        for role,counts in result['label_support'].items():
            lines.append('| '+role+' | '+' | '.join(str(counts.get(k,0))
                for k in ('Low','Medium','High','Critical','unavailable'))+' |')
        lines.append('')
    if 'recomputed_validation' in result:
        m=result['recomputed_validation']
        lines.extend(['| Class | Support | Precision | Recall | F1 |',
                      '|---|---:|---:|---:|---:|'])
        for name,row in m['per_class'].items():
            values=[row[k] for k in ('support','precision','recall','f1')]
            lines.append('| '+name+' | '+' | '.join('undefined' if v is None else f'{v:.4g}' for v in values)+' |')
        lines.extend(['',f"Macro F1: {m['macro_f1']:.4f}; ordinal MAE: {m['ordinal_mae']:.4f}.",
            f"Runtime development-promotion gate: {result['runtime_promotion_allowed']}.",
            'All four outputs being trained does not establish that all four have useful accuracy.'])
    else:lines.extend(['No student model was trained. No old three-class model is reported as a new four-class result.'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('END analysis',output,'promotion',result['runtime_promotion_allowed'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True);parser.add_argument('--features',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    analyze(args.campaign,args.features,args.output)
