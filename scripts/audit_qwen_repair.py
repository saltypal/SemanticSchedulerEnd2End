"""Persist native-label comparisons and hashes without turning VLM responses into gold."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT/'results/disaster_decoder_v3/20261023_qwen_repair_v1'
    responses = [json.loads(s) for s in (run/'diagnostic/responses.jsonl').read_text().splitlines()]
    manifest = json.loads((ROOT/'results/disaster_decoder_v3/20261022_stageA_v3/extraction/manifest.json').read_text())
    native = {row['image_id']: row for row in manifest}
    comparisons = []
    for row in responses:
        if row['kind'] != 'development_image':
            continue
        source = native[row['image_id']]
        assert source['role'] == 'A'
        comparisons.append(dict(image_id=row['image_id'], role=source['role'],
            native_labels={name: source['labels'][name] for name in
                ('buildings_minor_or_greater', 'roads_damage', 'flooding_structures')},
            qwen_tasks=row.get('assessment', {}).get('tasks', {})))
    control = next(row for row in responses if row['kind'] == 'vision_transport_control')
    description = control['description']['text'].lower()
    audit = dict(scope='five-case diagnostic, not accuracy validation',
        visual_control_mentions_expected_objects=all(word in description for word in ('red', 'square', 'blue', 'circle')),
        development_only=True, native_comparisons=comparisons,
        acceptance='NOT accepted as a semantic priority label generator',
        reason='All-abstention behavior improved, but one native building-positive image was scored zero; no positive road/flood cases in this pilot.',
        source_hashes={p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in run.rglob('*') if p.is_file() and p.suffix != '.pyc'})
    output = run/'semantic_audit.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(audit, indent=2))
    print(output)


if __name__ == '__main__':
    main()
