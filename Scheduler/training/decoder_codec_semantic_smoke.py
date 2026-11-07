"""Source-only predicted headers and separately model-assessed reconstructions."""
import argparse
import json
from pathlib import Path
import traceback

from Scheduler.training.decoder_v3_study import write_json
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterExtractor, ParameterHeader, TransmitterMemory, get_profile, sigmoid
from Archived.owners_before_simplification_20261027.SemParam.decoder_context_profiles import STUDY_PROFILES


def make_headers(source_evidence, profile_id=2001):
    profile = get_profile(profile_id, STUDY_PROFILES)
    memories = {f'ue{i}': TransmitterMemory(f'ue{i}', profile_id, registry=STUDY_PROFILES) for i in range(4)}
    result = []
    for i, evidence in enumerate(source_evidence):
        source = f'ue{i % 4}'
        memory = memories[source]
        bundle = ParameterExtractor().extract(evidence, profile, {}, memory)
        header = ParameterHeader(profile_id, i, memory.version, 0, bundle.values, bundle.available, STUDY_PROFILES)
        wire = header.to_bytes()
        memory.commit(evidence)
        result.append(dict(image_id=evidence.image_id, source_id=source, session_id='real_codec_preflight',
                           sequence=i, header_hex=wire.hex(), values=bundle.values, available=bundle.available,
                           missing_reasons=bundle.reasons,
                           header_scope='source_image_frozen_LADI_partial_parameters_S_I_only'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('received-manifest', 'decode-metrics', 'native-records', 'model-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profile', type=int, choices=(2001, 2002), default=2001)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from Archived.owners_before_simplification_20261027.SemParam.decoder_native_classifier import LocalNativeClassifier
        classifier = LocalNativeClassifier(args.model_dir)
        received = json.loads(args.received_manifest.read_text())
        decoded = {row['image_id']: row for row in json.loads(args.decode_metrics.read_text())}
        native = {row['image_id']: row for row in json.loads(args.native_records.read_text())}
        if not received.get('complete') or not received['records']:
            raise ValueError('complete received cache required')
        profile = get_profile(args.profile, STUDY_PROFILES)
        evidence, assessment = [], []
        for i, record in enumerate(received['records']):
            identity = record['image_id']
            print(f'Frozen source/reconstruction classifier {i+1}/{len(received["records"])}', flush=True)
            source = args.received_manifest.parent/record['reference_image']
            predicted = classifier.predict(source, identity)
            evidence.append(predicted)
            reconstructed = classifier.predict(decoded[identity]['reconstruction'], identity)
            probability = float(sigmoid([reconstructed.logits[profile.critical_claim]])[0])
            assessment.append(dict(image_id=identity, native_positive=native[identity]['labels'][profile.task],
                                   reconstructed_probability=probability, model_assessed_present=probability >= .5,
                                   scope='frozen_classifier_assessed_not_independent_fact_verification'))
            write_json(args.output/'assessment_partial.json', assessment)
        write_json(args.output/'headers.json', make_headers(evidence, args.profile))
        write_json(args.output/'assessment.json', assessment)
        write_json(args.output/'source_predictions.json', [row.to_dict() for row in evidence])
        write_json(args.output/'status.json', dict(status='complete', count=len(evidence), profile=args.profile,
                                                  classifier_sha256=classifier.checkpoint_sha256,
                                                  human_review=False, only_S_I_produced=True))
    except Exception as error:
        write_json(args.output/'failure.json', dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
