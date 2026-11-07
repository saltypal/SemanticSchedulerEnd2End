"""Bounded v2 SPI teacher campaign. The resulting labels are model-assessed."""
import hashlib
import json
from pathlib import Path
import threading
import time
import zipfile

from .artifacts import Run, write_json
from .contracts import digest
from .spi_campaign import verify_inputs
from .spi_labels import load_rubric, label_prompt, label_image, parse_shared_label


CONTROLS = (
    ("Low", "An ordinary intact office room with desks and no visible problem."),
    ("Low", "A clear landscape photograph of healthy trees and a calm lake."),
    ("Low", "A routine screenshot of a correctly functioning weather app."),
    ("Medium", "A small isolated pavement crack on an otherwise clear empty sidewalk, needing routine inspection."),
    ("Medium", "An otherwise working device displays a non-urgent low-storage notice."),
    ("Medium", "A minor localized roof tile defect on an intact building, appropriate for routine maintenance."),
    ("High", "A road is blocked by a large fallen tree; no person is visibly in immediate danger."),
    ("High", "A large building has collapsed into rubble; no ongoing immediate life threat is visible."),
    ("High", "Floodwater covers a main road and disrupts traffic, without anyone visibly trapped."),
    ("Critical", "A person is visibly trapped in a vehicle as flames advance toward them."),
    ("Critical", "A person is visibly struggling in rapidly rising floodwater beside an active current."),
    ("Critical", "A person hangs from a damaged high balcony edge and is visibly about to fall."),
)


def execute(input_root, output, reviewer, seconds=3200, enable_scene=False):
    """Run deterministic text controls and label the existing image cohort once."""
    input_root = Path(input_root)
    rubric = load_rubric(Path(__file__).parent / 'config/spi_rubric_v2.json')
    run = Run(output, {'seconds':seconds, 'rubric_sha256':digest(rubric),
                       'purpose':'fixed-rubric-v2-pseudo-labels', 'thinking':False,
                       'fresh_final_evaluation':False})
    labels=[]; controls=[]; worker=None; stop_scene=threading.Event()
    try:
        with run.stage('input_preflight'):
            rows, receipt=verify_inputs(input_root)
            write_json(run.path/'data/input_verification.json',receipt)
            # The locally cached extension is development-exposed. Reserve a
            # deterministic subset of *new-only* groups before any teacher
            # labels or model selection; it is a sealed development check,
            # not an independent fresh final test.
            audit_path=input_root/'selection_audit.json'
            old_count=json.loads(audit_path.read_text())['old'] if audit_path.is_file() else len(rows)
            if not isinstance(old_count,int) or not 0<=old_count<=len(rows):raise ValueError('invalid input cohort provenance')
            old_groups={row['group'] for row in rows[:old_count]}
            for row in rows[old_count:]:
                if row['group'] not in old_groups:
                    bucket=int(hashlib.sha256(row['group'].encode('utf-8')).hexdigest()[:8],16)%5
                    if bucket==0:row['split']='test'
            group_roles={}
            for row in rows:
                if row['group'] in group_roles and group_roles[row['group']]!=row['split']:
                    raise ValueError('sealed split crosses a duplicate group')
                group_roles[row['group']]=row['split']
            write_json(run.path/'data/split_manifest.json',{'old_count':old_count,
                'sealed_scope':'new-only groups; development-exposed, not fresh final',
                'assignments':[{'image_id':r['image_id'],'group':r['group'],'split':r['split']} for r in rows]})
            if receipt['layout']=='zip':
                unpacked=run.path/'data/images'; unpacked.mkdir()
                with zipfile.ZipFile(input_root/'images.zip') as archive:
                    if any(not (unpacked/name).resolve().is_relative_to(unpacked.resolve()) for name in archive.namelist()):
                        raise ValueError('unsafe input archive path')
                    archive.extractall(unpacked)
            else:
                unpacked=input_root/'images'
        if enable_scene:
            def run_scene():
                from dataclasses import asdict
                from PIL import Image
                from .scene_measurements import YOLOWorldSceneProvider
                scene_dir=run.path/'scene';scene_dir.mkdir(exist_ok=True)
                started=time.monotonic(); count=0;available=0
                try:
                    import torch
                    from ultralytics.utils.downloads import attempt_download_asset
                    torch.cuda.set_device(0)
                    checkpoint=Path(attempt_download_asset('yolov8s-worldv2.pt'))
                    producer=YOLOWorldSceneProvider(checkpoint,device='cuda:0')
                    with (scene_dir/'records.jsonl').open('w',encoding='utf-8') as handle:
                        for row in rows:
                            if stop_scene.is_set():break
                            path=unpacked/row['image_path']
                            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
                                raise ValueError('scene image checksum mismatch')
                            with Image.open(path) as image:scene=producer.measure(image)
                            if scene.status=='ok':available+=1
                            handle.write(json.dumps({'image_id':row['image_id'],'image_sha256':row['image_sha256'],
                                'scene':asdict(scene)},allow_nan=False)+'\n');handle.flush();count+=1
                            if count==1 or count%20==0:
                                write_json(scene_dir/'status.json',{'status':'running','count':count,'available':available})
                                print('GPU0 scene',count,'/',len(rows),'available',available,flush=True)
                            if count==1 and scene.status!='ok':raise RuntimeError('scene model preflight: '+str(scene.reason))
                    write_json(scene_dir/'status.json',{'status':'complete' if count==len(rows) else 'partial',
                        'count':count,'available':available,'seconds':time.monotonic()-started,'device':'cuda:0'})
                except Exception as error:
                    write_json(scene_dir/'status.json',{'status':'failed','count':count,'available':available,
                        'reason':type(error).__name__+': '+str(error),'seconds':time.monotonic()-started})
                    print('GPU0 scene failed',type(error).__name__,str(error),flush=True)
            worker=threading.Thread(target=run_scene,name='scene-gpu0',daemon=True);worker.start()
        with run.stage('text_rubric_controls'):
            # Text controls must not simultaneously demand an actual image.
            # Descriptions are hypothetical and test rubric comprehension only.
            text_rubric=dict(rubric)
            text_rubric['rules']=[r.replace('Use image and fixed rubric only',
                'Use the supplied hypothetical scene description and fixed rubric only') for r in rubric['rules']]
            prompt=(label_prompt(text_rubric).replace('ONE image using','ONE hypothetical scene description using')+
                    '\nFor this text-only engineering control, no image is provided. Treat the scene description as complete.')
            for index,(expected,description) in enumerate(CONTROLS):
                run.budget()
                result=reviewer.review_structured([], {'hypothetical_scene':description}, prompt,
                    parse_shared_label, f'text_control_{index:02d}', run.path/'raw_controls',
                    protocol=rubric['version']+'/text_control',text_only_diagnostic=True,
                    text_only_prefix='Text-only rubric control: assess the hypothetical scene description.\n')
                parsed=result.get('parsed') or result.get('review') or {}
                controls.append({'case':index,'expected':expected,'actual':parsed.get('level'),
                    'status':result.get('status'),'raw_response_ref':result.get('raw_response_ref')})
                write_json(run.path/'metrics/text_controls.json',controls)
                print('Text control',index+1,'/12',expected,parsed.get('level'),flush=True)
            correct=sum(r['expected']==r['actual'] for r in controls)
            per_class={name:sum(r['expected']==name and r['actual']==name for r in controls)
                       for name in ('Low','Medium','High','Critical')}
            gate=correct>=10 and all(n>=2 for n in per_class.values())
            write_json(run.path/'metrics/text_control_gate.json',{'correct':correct,'per_class':per_class,
                'passed':gate,'scope':'authored text fixtures, not visual performance'})
            if not gate:raise RuntimeError('text rubric control gate failed; preserve outputs and do not label cohort')
            # The shared reviewer caches its finite-logits check. Text controls
            # must not satisfy the distinct image-input preflight.
            reviewer._finite_preflight_passed=False
        with run.stage('offline_teacher'):
            for index,row in enumerate(rows):
                run.budget(); path=unpacked/row['image_path']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
                    raise ValueError('image checksum mismatch')
                label=label_image(reviewer,str(path),row['image_id'],run.path,rubric)
                labels.append({**row,**label})
                write_json(run.path/'data/spi_labels.json',labels)
                if index==0 or (index+1)%10==0:
                    print('SPI labels',index+1,'/',len(rows),'available',sum(r['available'] for r in labels),flush=True)
                if index==23:
                    complete=sum(r['execution'].get('status')=='complete' for r in labels)
                    write_json(run.path/'metrics/visual_preflight.json',{'cases':24,'complete':complete,
                        'levels':{name:sum(r.get('level')==name for r in labels) for name in ('Low','Medium','High','Critical')},
                        'scope':'technical visual preflight; distribution is not a class quota'})
                    if complete<20:raise RuntimeError('visual parsing preflight failed')
        from .spi_model_selection import support_gate,select
        gate=support_gate(labels)
        write_json(run.path/'metrics/four_class_support.json',gate)
        model_path=None
        if gate['passed']:
            try:
                select(input_root/'features.npz',run.path/'data/spi_labels.json',run.path/'student',seconds=240)
                model_path=run.path/'student/models/spi.json'
                write_json(run.path/'metrics/student_outcome.json',{'status':'trained','model':str(model_path)})
            except Exception as error:
                write_json(run.path/'metrics/student_outcome.json',{'status':'incomplete','reason':type(error).__name__+': '+str(error)})
        else:
            write_json(run.path/'metrics/student_outcome.json',{'status':'unsupported','reason':'four-class support gate failed'})
        if worker is not None:
            worker.join(timeout=60)
            scene_status=run.path/'scene/status.json'
            if not worker.is_alive() and scene_status.exists() and json.loads(scene_status.read_text()).get('status')=='complete':
                try:
                    from .parameter_vlm_review import execute as review_parameters
                    with run.stage('parameter_review'):
                        review_parameters(rows,unpacked,input_root/'features.npz',run.path/'scene/records.jsonl',
                                          run.path/'parameter_review',reviewer,model_path,run.budget)
                except Exception as error:
                    write_json(run.path/'parameter_review_error.json',{'error':type(error).__name__,'reason':str(error)})
            else:
                write_json(run.path/'parameter_review_skipped.json',{'reason':'scene output incomplete or still running'})
        write_json(run.path/'status.json',{'status':'labels_complete','labelled_images':len(labels),
            'available_labels':sum(r['available'] for r in labels),
            'claim':'model-assisted development labels; not independent urgency truth'})
    except Exception as error:
        write_json(run.path/'status.json',{'status':'incomplete','reason':str(error),
                                          'labelled_images':len(labels),'control_cases':len(controls)})
        raise
    finally:
        scene_live=False
        if worker is not None:
            # Scene extraction is independent of the VLM rubric. If the teacher
            # gate fails, give the detector a bounded opportunity to finish.
            if worker.is_alive() and len(labels)==0:
                worker.join(timeout=240)
            if worker.is_alive():
                stop_scene.set();worker.join(timeout=30)
            if worker.is_alive():
                scene_live=True
                # A live writer can change a file after hashing; exclude that
                # directory from the immutable recovery snapshot entirely.
                write_json(run.path/'scene_interrupted.json',{'reason':'scene worker did not stop within 30 seconds; scene excluded from recovery archive'})
        reviewer.close()
        files=[p for p in run.path.rglob('*') if p.is_file() and
               not p.is_relative_to(run.path/'data/images') and
               not (scene_live and p.is_relative_to(run.path/'scene')) and
               p.name not in ('spi_recovery.zip','checksums.json')]
        write_json(run.path/'checksums.json',{str(p.relative_to(run.path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        with zipfile.ZipFile(run.path/'spi_recovery.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in files+[run.path/'checksums.json']:archive.write(p,str(p.relative_to(run.path)))
    return run.path
