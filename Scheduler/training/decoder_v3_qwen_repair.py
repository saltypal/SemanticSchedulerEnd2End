"""Bounded development-only visual/structured-output diagnostic; no scheduler fitting."""
import base64
from io import BytesIO
import hashlib
import json
from pathlib import Path
import time

from Scheduler.training.decoder_v3_ollama import QWEN_MODEL, request_json
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import PROFILES
from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import parse_qwen_assessment


def assessment_schema():
    task = dict(type='object', additionalProperties=False,
        properties=dict(assessable=dict(type='boolean'),
            relevance_grade=dict(type='integer', minimum=0, maximum=3),
            evidence_grade=dict(type='integer', minimum=0, maximum=3),
            visible_evidence=dict(type='array', items=dict(type='string'), maxItems=2)),
        required=['assessable', 'relevance_grade', 'evidence_grade', 'visible_evidence'])
    return dict(type='object', additionalProperties=False,
        properties=dict(tasks=dict(type='object', additionalProperties=False,
            properties={p.task: task for p in PROFILES.values()},
            required=[p.task for p in PROFILES.values()])), required=['tasks'])


def vision_request(image, prompt, output_format=None, max_tokens=700):
    """Exactly one image per independent request; no preceding answer in context."""
    image = image.convert('RGB').copy()
    image.thumbnail((1024, 1024))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    payload = dict(model=QWEN_MODEL, messages=[dict(role='user', content=prompt,
        images=[base64.b64encode(buffer.getvalue()).decode()])], stream=False,
        think=False, keep_alive='10m',
        options=dict(temperature=0, seed=20261023, num_predict=max_tokens, num_ctx=8192))
    if output_format is not None:
        payload['format'] = output_format
    response = request_json('/api/chat', payload, timeout=240)
    return dict(text=response['message']['content'], image_size=list(image.size),
        image_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
        counters={k: response.get(k) for k in ('prompt_eval_count', 'eval_count', 'total_duration')})


REPAIRED_PROMPT = '''Inspect this aerial image. Assess ONLY what is visually supported.
Separate absence of visible damage from inability to assess: an identifiable undamaged
building or road can be assessable=true with evidence_grade=0. An image with no target,
obscured target, or inadequate visual detail can be assessable=false. Never force a
positive damage finding. Aerial appearance cannot establish casualties, urgency,
location, true safety, or subtle structural integrity. Mention concrete visible cues.
For building_damage: evidence_grade 0=no damage visible, 1=uncertain, 2=visible affected
condition, 3=clear minor-or-greater damage. For road_damage: 0=no damage visible,
1/2=uncertain/partial evidence, 3=clear visible road damage. For structural_flooding:
0=no inundation visible, 1/2=water near structures or uncertain inundation,
3=clear water inundation involving structures. Relevance 0=none,1=context,2=partial,
3=direct. Include all three task keys. Return the supplied JSON schema only.'''


def run_diagnostic(reviewer, images, output, max_seconds=600):
    """Retain prose and structured responses; no automatic claim of semantic success."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'responses.jsonl').exists():
        raise FileExistsError('choose a new diagnostic output directory')
    started, rows = time.monotonic(), []
    try:
        for identifier, image, kind in images:
            if time.monotonic()-started > max_seconds:
                break
            row = dict(image_id=identifier, kind=kind, producer=reviewer.provenance)
            try:
                row['description'] = vision_request(image,
                    'Describe the visible scene in three short factual sentences. '
                    'Mention visible objects and colors. Do not infer emergency priority.', max_tokens=180)
                row['structured'] = vision_request(image, REPAIRED_PROMPT, assessment_schema())
                row['assessment'] = parse_qwen_assessment(row['structured']['text'])
                row['status'] = 'schema_valid_pending_visual_review'
            except Exception as error:
                row['status'], row['error'] = 'failed', repr(error)
            rows.append(row)
            with (output/'responses.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row)+'\n')
            print(identifier, row['status'], flush=True)
    finally:
        summary = dict(attempted=len(rows), elapsed_seconds=time.monotonic()-started,
            assessable_counts={p.task: sum(bool(r.get('assessment', {}).get('tasks', {})
                .get(p.task, {}).get('assessable')) for r in rows if r['kind']=='development_image')
                for p in PROFILES.values()},
            semantic_quality='NOT established by JSON validity or assessability counts',
            ground_truth_created=False, scheduler_retrained=False)
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
