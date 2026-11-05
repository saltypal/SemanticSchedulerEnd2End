"""Render saved results only; never refit/select models or reopen absent tests."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['Low','Medium','High','Critical']

def confusion(matrix,path,title):
    matrix=np.array(matrix);fig,ax=plt.subplots(figsize=(5,5));ax.imshow(matrix,cmap='Blues')
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center')
    ax.set_xticks(range(4),NAMES,rotation=25);ax.set_yticks(range(4),NAMES)
    ax.set_xlabel('Predicted class');ax.set_ylabel('Teacher target');ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def render(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    candidates=json.loads((root/'comparison/candidates.json').read_text())
    selected=json.loads((root/'comparison/selection_frozen.json').read_text())
    confusion(selected['development']['confusion'],out/'development_confusion.png','Development: selection-exposed')
    for row in candidates:
        if row.get('training_loss_curve'):
            fig,ax=plt.subplots();ax.plot(np.arange(1,len(row['training_loss_curve'])+1),row['training_loss_curve'])
            ax.set_xlabel('Iteration');ax.set_ylabel('Training loss');ax.set_title(row['encoder']+' / '+row['head'])
            fig.tight_layout();fig.savefig(out/(row['encoder']+'_'+row['head']+'_training_loss.png'),dpi=160);plt.close(fig)
    file=root/'release/locked_test.json'
    if not file.exists():return {'status':'development_only','reason':'locked test not opened'}
    test=json.loads(file.read_text());metrics=test['metrics']
    if 'confusion' not in metrics:return {'status':'undefined_test_metrics'}
    confusion(metrics['confusion'],out/'locked_test_confusion.png','Locked test: teacher agreement')
    labels={r['image_id']:r for r in json.loads((root/'labeling/labels.json').read_text())}
    p=np.array([r['probabilities'] for r in test['predictions']]);y=np.array([NAMES.index(labels[i]['level']) for i in test['ids']])
    confidence=p.max(1);correct=p.argmax(1)==y;centers=[];accuracy=[];counts=[]
    for lower in np.arange(10)/10:
        mask=(confidence>=lower)&(confidence<lower+.1+(1e-12 if lower>=.9 else 0))
        if mask.any():centers.append(confidence[mask].mean());accuracy.append(correct[mask].mean());counts.append(int(mask.sum()))
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].scatter(centers,accuracy,s=np.array(counts)*5);axes[0].set(xlabel='Mean confidence',ylabel='Teacher agreement',xlim=(0,1),ylim=(0,1),title='Calibration; small bins are uncertain')
    axes[1].bar(NAMES,metrics['recall']);axes[1].axhline(.7,color='red',linestyle='--');axes[1].set(ylim=(0,1),ylabel='Recall',title='Locked-test per-class recall')
    fig.tight_layout();fig.savefig(out/'locked_test_calibration_recall.png',dpi=160);plt.close(fig)
    return {'status':'rendered_saved_results','files':[p.name for p in out.glob('*.png')]}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');a=parser.parse_args();print(render(a.root))
