"""Development-only grounding diagnostic. Never fits or promotes a classifier."""
import hashlib
import json
import shutil
from pathlib import Path

from parameters.artifacts import Run, write_json
from parameters.spi_campaign import verify_inputs
from parameters.spi_pipeline import _images, _image_path, recovery
from parameters.spi_protocol import PROMPT, CONTROLS, parse_evidence, control_metrics
from parameters.spi_teachers import TEACHERS, load_teacher

TIGHTENED_PROMPT = PROMPT + '''
Grounding requirements:
Identify the visible abnormal condition before inferring intervention. Object
identity, turbulent water, dramatic appearance, and possible future consequences
do not establish malfunction. Immediate threat requires a visible ongoing or
imminent harmful process or a visibly occurring critical failure. Describe the
subject and observable relationship supporting that distinction in evidence.
Do not infer disease, submersion, structural failure, or exposed people without
visible support. Severe static damage is not itself a visibly ongoing failure.
Use null when the decisive distinction cannot be established. Do not assign a
class to fill a distribution, and do not assume a dangerous event from its topic.
'''


def select_panel(labels):
    """Enriched diagnosis: 8 prior Critical, 5 Medium+3 High, 8 Low."""
    counts = {'Critical': 8, 'Medium': 5, 'High': 3, 'Low': 8}
    result = []
    for level, n in counts.items():
        # Preserve original order so the documented first bridge/dam cases occur.
        selected = [row for row in labels if row.get('level') == level][:n]
        if len(selected) != n:
            raise ValueError('diagnostic panel needs declared class support: ' + level)
        result.extend(selected)
    if len({row['group'] for row in result}) != 24:
        raise ValueError('diagnostic panel must contain 24 distinct groups')
    if any(row['split'] == 'test' for row in result):
        raise ValueError('diagnostic selection excludes test roles')
    first = result[:2] + [next(row for row in result if row['level'] == 'Low')]
    return first + [row for row in result if row not in first]


def assess(reviewer, paths, payload, case_id, output, text_only=False):
    if set(payload) != ({'hypothetical_scene'} if text_only else {'instruction'}):
        raise ValueError('evaluator information is prohibited in teacher payload')
    prompt = TIGHTENED_PROMPT
    if text_only:
        prompt = prompt.replace('this image alone', 'the complete hypothetical description alone')
    execution = reviewer.review_structured(paths, payload, prompt, parse_evidence,
        case_id, output, protocol='spi_grounding_diagnostic_v1',
        text_only_diagnostic=text_only,
        text_only_prefix=('Complete hypothetical scene description; no image is supplied.\n'
                          if text_only else None))
    parsed = execution.get('parsed') or execution.get('review')
    if not isinstance(parsed, dict):
        parsed = {'level': None, 'available': False, 'evidence': '',
                  'reason': execution.get('message', 'no valid response')}
    return {**parsed, 'execution': execution,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}


def execute(input_root, output, seconds=3000, runtime_receipt=None,
            maximum_images=None, teacher_loader=load_teacher, teacher_specs=None):
    config = {'seconds': seconds, 'purpose': 'enriched visual grounding diagnosis',
              'training_allowed': False, 'runtime_promotion_allowed': False,
              'independent_urgency_reference': False, 'thinking': False}
    run = Run(output, config)
    reviewer = None
    status = {**config, 'status': 'incomplete'}
    try:
        if runtime_receipt:
            shutil.copy2(runtime_receipt, run.path / 'runtime.json')
        with run.stage('01_verify_frozen_diagnostic'):
            manifest = json.loads(Path(__file__).with_name('grounding_panel.json').read_text())
            baseline = manifest['rows']
            if len(baseline) != 24 or len({r['group'] for r in baseline}) != 24:
                raise ValueError('invalid frozen diagnostic panel')
            rows, receipt = verify_inputs(Path(input_root))
            identities = {r['image_id']: r for r in rows}
            for row in baseline:
                identity = identities[row['image_id']]
                if row['split'] == 'test' or any(identity[k] != row[k]
                        for k in ('image_sha256', 'group', 'split')):
                    raise ValueError('diagnostic input identity/split mismatch')
            folder = _images(Path(input_root), rows, receipt, run)
            write_json(run.path / 'data/panel.json', manifest)
        summary = []
        for spec in (TEACHERS[:2] if teacher_specs is None else teacher_specs):
            with run.stage('02_load_' + spec['name']):
                reviewer = teacher_loader(spec)
                write_json(run.path / 'models' / (spec['name'] + '.json'), reviewer.provenance)
            results = []
            with run.stage('03_text_conformance_' + spec['name']):
                for control in CONTROLS:
                    run.budget()
                    value = assess(reviewer, [], {'hypothetical_scene': control['description']},
                        'text_' + control['id'], run.path / 'raw' / spec['name'], True)
                    results.append({'id': control['id'], 'expected': control['expected'],
                        'level': value['level'], 'valid': value['execution'].get('status') == 'complete',
                        'label': value})
                    write_json(run.path / 'metrics' / (spec['name'] + '_controls.json'), results)
                    if value['execution'].get('status') == 'model_failed':
                        raise RuntimeError('teacher text inference failed; inspect saved request before retry')
                    if len(results) % 12 == 0:
                        print('Text controls', spec['name'], len(results), '/48', flush=True)
                metric = control_metrics(results)
            observed = []
            with run.stage('04_visual_grounding_' + spec['name']):
                for row in baseline:
                    run.budget()
                    value = assess(reviewer, [_image_path(folder, row)],
                        {'instruction': 'Assess the depicted intervention need.'},
                        'visual_' + row['image_id'], run.path / 'raw' / spec['name'])
                    observed.append({'image_id': row['image_id'], **value})
                    write_json(run.path / 'data' / (spec['name'] + '_visual.json'), observed)
                    if value['execution'].get('status') == 'model_failed':
                        raise RuntimeError('teacher visual inference failed; inspect saved request before retry')
                    print('Grounding diagnostic', spec['name'], len(observed), '/24',
                          value['execution'].get('status'), value['level'], flush=True)
                    if len(observed) == 3 and any(x['execution'].get('status') != 'complete'
                            or not x['execution'].get('image_inputs') for x in observed):
                        raise RuntimeError('three-image transport preflight failed')
            summary.append({'teacher': spec['name'], 'text_controls': metric,
                'visual_n': len(observed),
                'valid': sum(x['execution'].get('status') == 'complete' for x in observed),
                'available': sum(x['available'] for x in observed),
                'changed_from_v3': sum(x['level'] != row['level'] for x, row in zip(observed, baseline)),
                'accuracy': None, 'reason': 'no independent image urgency reference'})
            write_json(run.path / 'metrics/summary.json', summary)
            reviewer.close()
            reviewer = None
        status.update(status='diagnostic_complete', teachers=len(summary),
            next_gate='review decisive visual claims; no automatic bulk labeling or model promotion')
    except Exception as error:
        status.update(reason=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if reviewer is not None:
            reviewer.close()
        write_json(run.path / 'status.json', status)
        recovery(run.path)
    return run.path
