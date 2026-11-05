"""Four-class SPI: audited teacher pilot -> causal labels -> grouped student fit.

No model is loaded on import. All labels generated here are pseudo-labels.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import zipfile
import numpy as np

from .artifacts import Run, write_json
from .contracts import digest
from .spi_campaign import verify_inputs
from .spi_protocol import CONTROLS, VERSION, request, control_metrics
from .spi_teachers import TEACHERS, load_teacher, UnsupportedTeacher


def _images(input_root, rows, receipt, run):
    if receipt['layout'] == 'zip':
        folder = run.path / 'data/images'
        folder.mkdir()
        with zipfile.ZipFile(input_root / 'images.zip') as archive:
            if any(not (folder / name).resolve().is_relative_to(folder) for name in archive.namelist()):
                raise ValueError('unsafe image archive path')
            archive.extractall(folder)
    else:
        folder = input_root / 'images'
    return folder


def _image_path(folder, row):
    path = (folder / row['image_path']).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError('image path outside dataset')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
        raise ValueError('image hash mismatch')
    return str(path)


def recovery(path):
    """Persist partial outputs after successful runs and every failure."""
    files = [p for p in path.rglob('*') if p.is_file()
             and not p.is_relative_to(path / 'data/images')
             and p.name not in ('recovery.zip', 'checksums.json')]
    write_json(path / 'checksums.json', {str(p.relative_to(path)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with zipfile.ZipFile(path / 'recovery.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files + [path / 'checksums.json']:
            archive.write(p, str(p.relative_to(path)))


def execute(input_root, output, seconds=6400, teacher_loader=load_teacher,
            teacher_specs=TEACHERS, maximum_images=300, allow_student=True,
            visual_panel_size=80, runtime_receipt=None):
    config = {'seconds': seconds, 'maximum_images': maximum_images,
              'target_version': VERSION, 'thinking': False,
              'fresh_final_evaluation': False, 'teacher_specs': list(teacher_specs),
              'visual_panel_size':visual_panel_size,
              'purpose': 'four-class SPI development; not independent urgency truth'}
    run = Run(output, config)
    reviewer = None
    labels = []
    outcome = {'status': 'incomplete', 'four_class_model': False,
               'independent_urgency_validation': False}
    write_json(run.path / 'status.json', outcome)
    try:
        if runtime_receipt is not None:
            shutil.copy2(runtime_receipt,run.path/'runtime.json')
        with run.stage('01_input_and_reference_manifest'):
            input_root = Path(input_root)
            rows, receipt = verify_inputs(input_root)
            rows = rows[:maximum_images]
            if len({r['image_id'] for r in rows}) != len(rows):
                raise ValueError('duplicate image IDs')
            groups = {}
            for row in rows:
                if row['group'] in groups and groups[row['group']] != row['split']:
                    raise ValueError('group crosses splits')
                groups[row['group']] = row['split']
            folder = _images(input_root, rows, receipt, run)
            write_json(run.path / 'data/input_verification.json', receipt)
            write_json(run.path / 'data/split_manifest.json', {'rows': rows,
                'scope': 'existing exposed development roles; no fresh final claim'})
            write_json(run.path / 'data/authored_controls.json', CONTROLS)
            write_json(run.path / 'data/reference_layers.json', {
                'controls': 'authored text policy conformance',
                'natural_images': 'unlabeled SPI; teacher labels are pseudo',
                'native_labels': 'not supplied to the teacher',
                'limitation': 'no independent natural-image urgency gold in this input'})

        candidates = []
        selected = None
        panel_labels = {}
        # A failed Qwen9B run does not automatically justify re-running every model.
        # Challenge it only when no protocol passes the same fixed controls.
        for spec in teacher_specs:
            run.budget()
            if reviewer is not None:
                reviewer.close()
                reviewer = None
            with run.stage('02_load_' + spec['name']):
                try:
                    reviewer = teacher_loader(spec)
                except UnsupportedTeacher as error:
                    candidates.append({'teacher':spec['name'],'passed':False,
                        'status':'unsupported_runtime','reason':str(error)})
                    write_json(run.path / 'metrics/teacher_candidates.json', candidates)
                    print('Teacher unavailable', spec['name'], str(error), flush=True)
                    continue
                reviewer.provenance={**reviewer.provenance,
                    'configured_generic_mode':reviewer.provenance.get('mode'),
                    'mode':'image_only_spi_teacher_v3',
                    'purpose':'offline_pseudo_labelling_no_native_annotations'}
                write_json(run.path / 'models' / (spec['name'] + '_provenance.json'), reviewer.provenance)
            with run.stage('03_visual_transport_' + spec['name']):
                checks = []
                # Only development roles are used in selection/preflight.
                preflight_rows = [r for r in rows if r['split'] != 'test'][:10]
                if len(preflight_rows) != 10:
                    raise ValueError('ten development images required for image transport preflight')
                for i, row in enumerate(preflight_rows):
                    run.budget()
                    label = request(reviewer, [_image_path(folder, row)],
                        {'instruction': 'Assess the depicted intervention need.'},
                        'visual_transport_' + row['image_id'], run.path / 'raw' / spec['name'], 'evidence')
                    execution = label['execution']
                    checks.append({'image_id': row['image_id'],
                        'complete': execution.get('status') == 'complete',
                        'has_image': bool(execution.get('image_inputs')),
                        'level': label['level'], 'execution': execution})
                    write_json(run.path / 'metrics' / (spec['name'] + '_transport.json'), checks)
                    print('Visual preflight', spec['name'], i + 1, '/10', execution.get('status'), flush=True)
                if sum(c['complete'] and c['has_image'] for c in checks) < 10:
                    raise RuntimeError('image transport/response preflight failed; inspect receipts')
            for protocol in ('evidence', 'direct'):
                with run.stage('04_controls_' + spec['name'] + '_' + protocol):
                    results = []
                    for i, control in enumerate(CONTROLS):
                        run.budget()
                        label = request(reviewer, [], {'hypothetical_scene': control['description']},
                            protocol + '_' + control['id'], run.path / 'raw' / spec['name'], protocol)
                        results.append({'id': control['id'], 'expected': control['expected'],
                            'level': label['level'], 'valid': label['execution'].get('status') == 'complete',
                            'label': label})
                        write_json(run.path / 'metrics' / (spec['name'] + '_' + protocol + '_controls.json'), results)
                        if i == 0 or (i + 1) % 12 == 0:
                            print('Controls', spec['name'], protocol, i + 1, '/48', flush=True)
                    metric = control_metrics(results)
                    candidate = {'teacher': spec['name'], 'protocol': protocol, **metric}
                    candidates.append(candidate)
                    write_json(run.path / 'metrics/teacher_candidates.json', candidates)
                    print('Teacher candidate', candidate, flush=True)
            passing = [c for c in candidates if c['passed'] and c['teacher'] == spec['name']]
            if passing:
                panel = [r for r in rows if r['split'] != 'test'][:visual_panel_size]
                if len(panel) != visual_panel_size:
                    raise ValueError('insufficient development images for the frozen visual panel')
                for candidate in passing:
                    protocol = candidate['protocol']
                    with run.stage('04_visual_panel_' + spec['name'] + '_' + protocol):
                        observed = []
                        for i, row in enumerate(panel):
                            run.budget()
                            label = request(reviewer, [_image_path(folder, row)],
                                {'instruction':'Assess the depicted intervention need.'},
                                'panel_' + protocol + '_' + row['image_id'],
                                run.path / 'raw' / spec['name'] / 'visual_panel', protocol)
                            observed.append({**row, **label, 'teacher':reviewer.provenance})
                            write_json(run.path / 'data' / (spec['name']+'_'+protocol+'_visual_panel.json'), observed)
                            if i == 0 or (i + 1) % 10 == 0:
                                print('Visual panel',spec['name'],protocol,i+1,'/',visual_panel_size,flush=True)
                        valid = sum(r['execution'].get('status') == 'complete' for r in observed)
                        latencies = [r['execution'].get('wall_seconds',0) for r in observed]
                        candidate.update(visual_panel_n=len(observed),
                            visual_valid_fraction=valid/len(observed),
                            visual_available_fraction=sum(r['available'] for r in observed)/len(observed),
                            visual_median_seconds=float(np.median(latencies)),
                            natural_urgency_reference=False)
                        candidate['passed'] = candidate['passed'] and valid/len(observed)>=.98
                        panel_labels[protocol] = {r['image_id']:r for r in observed}
                        write_json(run.path/'metrics/teacher_candidates.json',candidates)
                passing = [c for c in passing if c['passed']]
                if passing:
                    selected = max(passing,key=lambda c:(c['macro_f1'],-c['critical_to_low_medium'],
                        -c['visual_median_seconds'],c['protocol']=='evidence'))
                    agreements = []
                    if len(passing)==2:
                        for row in panel:
                            agreements.append({'image_id':row['image_id'],
                                'direct':panel_labels['direct'][row['image_id']]['level'],
                                'evidence':panel_labels['evidence'][row['image_id']]['level']})
                    write_json(run.path/'metrics/visual_protocol_comparison.json',{
                        'paired_outputs':agreements,'selected':selected,
                        'scope':'visual validity, disagreement and cost; no independent urgency accuracy'})
                    break
        if selected is None:
            outcome.update(status='teacher_gate_failed', reason='no protocol meets four-class text controls')
            return run.path
        write_json(run.path / 'models/selected_teacher.json', {
            **selected, 'provenance': reviewer.provenance,
            'selection_scope': 'authored text controls plus visual transport; not natural urgency accuracy'})

        with run.stage('05_visual_panel_and_labeling'):
            # Test groups are deliberately not sent to the teacher during fitting.
            offered = [r for r in rows if r['split'] != 'test']
            for index, row in enumerate(offered):
                run.budget()
                label = panel_labels.get(selected['protocol'],{}).get(row['image_id'])
                if label is None:
                    label = request(reviewer, [_image_path(folder, row)],
                        {'instruction': 'Assess the depicted intervention need.'},
                        row['image_id'], run.path / 'raw/image_labels', selected['protocol'])
                labels.append({**row, **label, 'teacher': reviewer.provenance,
                    'assessment_scope': 'pseudo-label development; not independently verified urgency'})
                write_json(run.path / 'data/spi_labels.json', labels)
                if index == 0 or (index + 1) % 10 == 0:
                    print('SPI labels', index + 1, '/', len(offered),
                          {name: sum(r.get('level') == name for r in labels)
                           for name in ('Low', 'Medium', 'High', 'Critical')}, flush=True)
                if index == 79:
                    write_json(run.path / 'metrics/visual_panel.json', {
                        'n': 80, 'valid': sum(r['execution']['status'] == 'complete' for r in labels),
                        'independent_reference': False,
                        'claim': 'visual output audit and class support, not visual urgency accuracy'})
                    if sum(r['execution']['status'] == 'complete' for r in labels) < 79:
                        raise RuntimeError('visual panel response-validity gate failed')

        with run.stage('06_four_class_support'):
            from .spi_model_selection import support_gate
            gate = support_gate(labels)
            write_json(run.path / 'metrics/support_gate.json', gate)
            if not gate['passed']:
                outcome.update(status='four_class_support_failed',
                               reason='do not invent/remove classes; inspect support by class/group')
                return run.path
        if not allow_student:
            outcome.update(status='labels_ready_student_not_requested')
            return run.path
        # Free GPU before CPU training and preserve all labels even if selection fails.
        reviewer.close()
        reviewer = None
        with run.stage('07_grouped_student_selection'):
            from .spi_model_selection import select
            remaining = seconds - (time.monotonic() - run.started)
            if remaining < 300:
                raise TimeoutError('less than five minutes left for bounded CPU selection/export')
            report = select(input_root / 'features.npz', run.path / 'data/spi_labels.json',
                run.path / 'student', seconds=min(600, int(remaining - 60)))
            outcome.update(status='trained_development_model', four_class_model=True,
                model='student/models/spi.json', validation=report['student'],
                independent_urgency_validation=False)
        write_json(run.path / 'REPORT.json', outcome)
        return run.path
    except Exception as error:
        outcome.update(status='incomplete', reason=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        outcome['labelled_images'] = len(labels)
        write_json(run.path / 'status.json', outcome)
        if reviewer is not None:
            reviewer.close()
        (run.path / 'REPORT.md').write_text(
            '# Four-class SPI campaign\n\n' + json.dumps(outcome, indent=2) +
            '\n\nLabels and validation describe teacher agreement, not independent urgency accuracy.\n', encoding='utf-8')
        recovery(run.path)
