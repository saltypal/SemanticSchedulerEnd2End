"""Render a source-native study diagram; no generated result imagery."""
from pathlib import Path


def build(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,7.5));ax.set_xlim(0,13);ax.set_ylim(0,7.5);ax.axis('off')
    def box(x,y,w,h,text,color='#e8f0fa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#334155',linewidth=1.3))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color='#152238')
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#475569',linewidth=1.4))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=8)
    ax.text(.2,7.1,'Annotation-assisted parameter validation: implementation and evidence boundaries',fontsize=16,weight='bold')
    box(.3,5.25,2.7,1.1,'Images + fixed UE tasks\nSix UEs / five missions')
    box(4.05,5.25,3.3,1.1,'GPU 0: CLIP + YOLO-World\nPixels → predicted evidence')
    box(8.5,5.25,3.7,1.1,'Parameter JSON\nCompact vs structured candidate')
    arrow((3.08,5.8),(3.95,5.8));arrow((7.43,5.8),(8.4,5.8))
    box(4.05,3.35,3.3,1.1,'CPU: separate UE knowledge bases\nQuery → serialize → commit','#e8f5ed')
    arrow((5.7,5.15),(5.7,4.55));arrow((7.43,3.9),(10.3,5.15),'history-dependent fields')
    box(.3,1.4,3.1,1.35,'Published annotations\nPartial native facts\nEvaluator / reviewer only','#fff2d8')
    box(8.5,2.9,3.7,1.35,'GPU 1: Qwen3-VL-4B\nImage + annotation + JSON\nBinary consistency / sufficiency')
    arrow((10.35,5.15),(10.35,4.35))
    arrow((3.5,2.05),(8.4,3.45),'annotations never enter extraction')
    box(4.6,.45,3.8,1.35,'Deterministic checks + Luna audit\nKnown-fact metrics / failures\nJournal methods and measured results','#f0eafb')
    arrow((3.5,1.8),(4.5,1.15));arrow((10.35,2.8),(8.5,1.15))
    ax.text(.3,.12,'Gate: 72 reviewer-control cases before actual parameter extraction. No scheduler training in this pilot.',fontsize=9,color='#475569')
    fig.tight_layout();fig.savefig(output/'annotation_validation_workflow.png',dpi=170,bbox_inches='tight')
    fig.savefig(output/'annotation_validation_workflow.svg',bbox_inches='tight');plt.close(fig)
    return output/'annotation_validation_workflow.png'


if __name__=='__main__':
    print(build(Path(__file__).parent/'docs/figures'))
