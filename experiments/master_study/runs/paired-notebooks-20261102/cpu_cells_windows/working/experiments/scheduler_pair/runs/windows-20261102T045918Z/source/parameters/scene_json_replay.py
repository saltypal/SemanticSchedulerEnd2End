"""Convert retained runtime caches into v2 JSON; never run a model or use labels."""
from pathlib import Path
import argparse
import hashlib
import json
from .artifacts import write_json
from .contracts import canonical
from .extractor import VisualEvidence
from .scene_json import build_scene_document, restore_bundle, validate_document
from .ue_profiles import load_ue_profiles


def replay(source, output, ontology=None):
    source, output = Path(source), Path(output)
    if output.exists(): raise FileExistsError('use a fresh conversion output directory')
    feature_path = source/'extraction/features.jsonl'
    trace_path = source/'extraction/trace.json'
    # Verify only consumed source bytes; evaluator case files are never read.
    checks = json.loads((source/'checksums.json').read_text())
    for path in (feature_path, trace_path):
        if checks.get(path.relative_to(source).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('runtime cache identity mismatch')
    evidence = {row['image_id']: VisualEvidence.from_dict(row['evidence'])
                for row in map(json.loads, feature_path.read_text().splitlines())}
    missions, _ = load_ue_profiles()
    rows = json.loads(trace_path.read_text())
    output.mkdir(parents=True)
    sizes = []; conflicts = 0; comparison_count = 0
    for row in rows:
        bundle = restore_bundle(row['bundle'])
        document, audit = build_scene_document(bundle, evidence[row['image_id']], missions[bundle.mission_id],
            source_id=row['sender'], image_id=row['image_id'], ontology=ontology)
        validate_document(document)
        stem = row['observation_id']
        if Path(stem).name != stem: raise ValueError('unsafe observation identity')
        write_json(output/'documents'/(stem+'.json'), document)
        write_json(output/'audit'/(stem+'.json'), audit)
        for claim in document['scene']['claims']:
            if claim['name'] in row['candidate']['claims']:
                assert claim['value'] is row['candidate']['claims'][claim['name']]
                comparison_count += 1
        conflicts += len(document['diagnostics'])
        sizes.append(dict(observation=stem, old_json_bytes=len(canonical(row['candidate']).encode()),
                          new_json_bytes=len(canonical(document).encode()),
                          audit_bytes=len(canonical(audit).encode())))
    report = dict(status='complete', schema='semantic_scene_evidence_v2', observations=len(rows),
        unique_images=len(evidence), model_inference_run=False, evaluator_files_consumed=False,
        retained_prediction_checks=comparison_count, diagnostic_count=conflicts,
        source_files={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (feature_path, trace_path)},
        bytes=sizes, byte_scope='uncompressed canonical UTF-8 JSON, not binary header or latent payload',
        scope='schema conversion and information preservation; no new accuracy or reviewer result')
    write_json(output/'REPORT.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--ontology')
    args = parser.parse_args()
    ontology = json.loads(Path(args.ontology).read_text()) if args.ontology else None
    print(json.dumps(replay(args.source, args.output, ontology), indent=2))
