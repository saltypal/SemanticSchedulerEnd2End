"""Five under-10B VLMs on one fixed panel. No automatic winner or training."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from parameters.artifacts import Run,write_json
from parameters.spi_campaign import verify_inputs
from parameters.spi_pipeline import _images,_image_path,recovery
from parameters.spi_direct_priority import assess_direct,parse_class,PROMPT
from parameters.spi_protocol import CONTROLS,control_metrics
from parameters.spi_teachers import load_teacher,TEACHERS
from parameters.gemma_spi_reviewer import load_gemma
from parameters.native_vlm_candidates import load_candidate

SPECS=({'name':'gemma4_e4b'},TEACHERS[0],TEACHERS[1],
       {'name':'qwen25_7b'},{'name':'internvl3_8b'})


def execute(input_root,output,seconds=6400,runtime_receipt=None,**kwargs):
    run=Run(output,{'seconds':seconds,'purpose':'five under10B VLM comparison, same direct prompt',
        'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'training_allowed':False,'automatic_selection':False})
    if runtime_receipt:shutil.copy2(runtime_receipt,run.path/'runtime.json')
    status={'status':'incomplete','training_allowed':False,'runtime_promotion_allowed':False}
    reviewer=None;summary=[]
    try:
        rows,receipt=verify_inputs(Path(input_root))
        folder=_images(Path(input_root),rows,receipt,run)
        panel=json.loads(Path(__file__).with_name('grounding_panel.json').read_text())
        identities={r['image_id']:r for r in rows}
        for row in panel['rows']:
            if row['split']=='test' or any(identities[row['image_id']][k]!=row[k]
                    for k in ('image_sha256','group','split')):raise ValueError('panel identity mismatch')
        write_json(run.path/'data/panel.json',panel)
        for spec in SPECS:
            run.budget();name=spec['name'];observed=[];controls=[]
            try:
                with run.stage('load_'+name):
                    loader=load_gemma if name=='gemma4_e4b' else load_candidate if name in ('qwen25_7b','internvl3_8b') else load_teacher
                    reviewer=loader(spec)
                    write_json(run.path/'models'/f'{name}.json',reviewer.provenance)
                # Images first: detect broken visual paths before spending on text controls.
                with run.stage('images_'+name):
                    for row in panel['rows']:
                        run.budget()
                        result=assess_direct(reviewer,[_image_path(folder,row)],
                            {'instruction':'Label the urgency of the visible scene.'},row['image_id'],run.path/'raw'/name)
                        observed.append({'image_id':row['image_id'],**result})
                        write_json(run.path/'data'/f'{name}_visual.json',observed)
                        print(name,'visual',len(observed),'/24',result['execution'].get('status'),result['level'],flush=True)
                        if result['execution'].get('status')=='model_failed':raise RuntimeError('model failed; candidate stopped')
                with run.stage('text_controls_'+name):
                    for control in CONTROLS:
                        run.budget()
                        response=reviewer.review_structured([],{'hypothetical_scene':control['description']},
                            PROMPT.replace('scene shown in each image','complete hypothetical description supplied'),
                            parse_class,control['id'],run.path/'raw'/name/'controls',
                            protocol='spi_direct_scene_priority_v1/text',text_only_diagnostic=True,
                            text_only_prefix='Complete hypothetical description; no image supplied.\n')
                        parsed=response.get('parsed') or {}
                        controls.append({'id':control['id'],'expected':control['expected'],
                            'level':parsed.get('level'),'valid':response.get('status')=='complete','execution':response})
                        write_json(run.path/'metrics'/f'{name}_controls.json',controls)
                        if response.get('status')=='model_failed':raise RuntimeError('text inference failed')
                summary.append({'name':name,'status':'complete','text_controls':control_metrics(controls),
                    'visual_valid':sum(r['execution'].get('status')=='complete' for r in observed),
                    'n':len(observed),'median_wall_seconds':float(np.median([r['execution']['wall_seconds'] for r in observed])),
                    'visual_accuracy':None,'independent_reference':False})
            except Exception as error:
                summary.append({'name':name,'status':'failed','reason':type(error).__name__+': '+str(error),
                                'visual_attempted':len(observed),'controls_attempted':len(controls)})
                print('CANDIDATE FAILED',summary[-1],flush=True)
                if isinstance(error,TimeoutError):raise
            finally:
                if reviewer:reviewer.close();reviewer=None
                write_json(run.path/'metrics/comparison.json',summary)
                recovery(run.path)
        status['status']='comparison_complete' if all(r['status']=='complete' for r in summary) else 'comparison_partial'
    finally:
        if reviewer:reviewer.close()
        status.update(candidates=summary,selection='pending claim-level review; labels are not ground truth',flops_status='not_instrumented')
        write_json(run.path/'status.json',status);recovery(run.path)
    return run.path
