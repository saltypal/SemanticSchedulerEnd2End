"""Explicitly authorized all-300 Gemma labeling; never trains or promotes SPI."""
import json
import shutil
from pathlib import Path
from parameters.artifacts import Run, write_json
from parameters.spi_campaign import verify_inputs
from parameters.spi_pipeline import _images, _image_path, recovery
from parameters.gemma_spi_reviewer import load_gemma
from experiments.spi_four_class.visual_reliability import assess


def execute(input_root, output, seconds=3000, runtime_receipt=None, maximum_images=300,
            teacher_loader=load_gemma, teacher_spec=None, label_mode='claims', **kwargs):
    if label_mode not in ('claims','direct'):raise ValueError('invalid label mode')
    assessor=assess
    if label_mode=='direct':
        from parameters.spi_direct_priority import assess_direct
        assessor=assess_direct
    run=Run(output,{'seconds':seconds,'purpose':'all300 VLM labeling with abstentions retained',
        'teacher_spec':teacher_spec,
        'label_mode':label_mode,
        'training_allowed':False,'runtime_promotion_allowed':False})
    reviewer=None;labels=[]
    status={'status':'incomplete','training_allowed':False,'runtime_promotion_allowed':False,
            'independent_urgency_validation':False,'flops_status':'not_instrumented'}
    try:
        if runtime_receipt:shutil.copy2(runtime_receipt,run.path/'runtime.json')
        rows,receipt=verify_inputs(Path(input_root))
        if len(rows)!=300:raise ValueError('exact approved 300-image cohort required')
        folder=_images(Path(input_root),rows,receipt,run)
        with run.stage('01_load_teacher'):
            reviewer=teacher_loader(teacher_spec or {})
            write_json(run.path/'models/teacher.json',reviewer.provenance)
        with run.stage('02_label_all300'):
            for row in rows:
                run.budget()
                result=assessor(reviewer,[_image_path(folder,row)],
                    {'instruction':'Label the urgency of the visible scene.' if label_mode=='direct' else
                     'Assess the depicted intervention need.'},row['image_id'],run.path/'raw')
                labels.append({**row,**result,'label_source':'vlm_pseudo',
                    'assessment_scope':'exposed development; not independent urgency truth'})
                write_json(run.path/'data/spi_labels.json',labels)
                print('VLM all300',len(labels),'/300',result['execution'].get('status'),result['level'],flush=True)
                if result['execution'].get('status')=='model_failed':
                    raise RuntimeError('model inference failed; preserve evidence before retry')
        status['status']='labeling_complete'
    finally:
        status['attempted_images']=len(labels)
        status['available_labels']=sum(r.get('available') is True for r in labels)
        status['levels']={name:sum(r.get('level')==name for r in labels)
                          for name in ('Low','Medium','High','Critical',None)}
        write_json(run.path/'status.json',status)
        if reviewer:reviewer.close()
        recovery(run.path)
    return run.path
