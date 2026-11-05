"""Bridge opaque received payloads, supplied transmitter headers and real decoding."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.decoder_service import DecoderService
from Scheduler.training.decoder_v3_study import write_json
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterHeader
from Archived.owners_before_simplification_20261027.SemParam.decoder_context_profiles import STUDY_PROFILES


def replay_received_policy(adapter, received_manifest, headers, policy, cache_root, output,
                           checkpoint_sha, profile_id, device='cpu',
                           header_scope='supplied_transmitter_headers_unverified_producer'):
    """Fact truth never crosses this interface; semantic scoring is external.

    Timeline includes measured dispatch+decode seconds, excludes artifact I/O.
    Failed attempts use measured wall elapsed including their error/I/O path.
    Header entries explicitly supply image_id, header_hex, source_id, session_id,
    sequence. They are not derived from native evaluator labels here.
    """
    from Codec.integration.decoder_payload_cache import decode_cached
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('codec policy replay output must be new or empty')
    records = received_manifest['records']
    if (not received_manifest.get('complete') or not records
            or len({row['image_id'] for row in records}) != len(records)):
        raise ValueError('complete unique received payload manifest required')
    by_image = {row['image_id']: row for row in headers}
    if len(by_image) != len(headers) or set(by_image) != {row['image_id'] for row in records}:
        raise ValueError('exactly one supplied transmitter header per payload required')
    policy.reset()
    service = DecoderService(policy, profile_id, STUDY_PROFILES)
    for record in records:
        supplied = by_image[record['image_id']]
        wire = bytes.fromhex(supplied['header_hex'])
        header = ParameterHeader.from_bytes(wire, STUDY_PROFILES)
        service.enqueue(ReceivedMessage(supplied['source_id'], supplied['session_id'], header.message_token,
                                        wire, record, True, 0., supplied['sequence']))
    output.mkdir(parents=True, exist_ok=True)
    trace, clock = [], 0.
    while len(service):
        started = time.perf_counter()
        job = service.dispatch(clock)
        dispatch_seconds = time.perf_counter()-started
        record = job.message.payload_ref
        image_name = hashlib.sha256(record['image_id'].encode()).hexdigest()
        decode_started = time.perf_counter()
        try:
            metrics = decode_cached(adapter, record, cache_root, output/f'{image_name}.png',
                                     checkpoint_sha, device)
            clock += dispatch_seconds+metrics['decode_seconds']
            completion = service.complete(job.message.message_id, 'succeeded', clock)
            entry = dict(image_id=record['image_id'], metrics=metrics,
                         decision=asdict(job.decision), completion=asdict(completion),
                         dispatch_seconds=dispatch_seconds)
        except Exception as error:
            clock += dispatch_seconds+time.perf_counter()-decode_started
            completion = service.complete(job.message.message_id, 'failed', clock)
            entry = dict(image_id=record['image_id'], metrics=None, decision=asdict(job.decision),
                         completion=asdict(completion), dispatch_seconds=dispatch_seconds,
                         error_type=type(error).__name__, error=str(error))
        trace.append(entry)
        write_json(output/'trace.json', trace)
    result = dict(status='complete', header_scope=header_scope, trace=trace,
                  source_payloads={r['image_id']: r['payload_sha256'] for r in records},
                  timeline='measured_dispatch_plus_decode_excludes_successful_artifact_IO',
                  semantic_fact_preservation='not_assessed',
                  execution_scopes=sorted({r['execution_scope'] for r in records}),
                  actual_codec_returned_successfully=any(row['metrics'] is not None and
                      row['metrics']['execution_scope'] == 'measured_checkpoint_inference' for row in trace),
                  succeeded=sum(row['completion']['outcome'] == 'succeeded' for row in trace),
                  failed=sum(row['completion']['outcome'] == 'failed' for row in trace))
    write_json(output/'summary.json', result)
    return result


def main():
    import argparse
    import json
    import traceback
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('received-manifest', 'headers', 'output', 'model-dir', 'upstream-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--policy-artifact', type=Path)
    parser.add_argument('--profile', type=int, choices=(2001, 2002), default=2001)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from Codec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
        from Codec.integration.decoder_payload_cache import sha256_file
        from Scheduler.contextual_decoder_scheduler import StaticPolicyAdapter, load_contextual_policy
        from Scheduler.decoder_scheduler import SemanticScheduler
        print('Strict load codec for common received-payload policy replay', flush=True)
        adapter = load_stage1_swinjscc(args.model_dir, device=args.device, upstream_root=args.upstream_root)
        for module in adapter._require_built():
            module.eval()
        checkpoint_sha = sha256_file(args.model_dir/'stage1_swinjscc_full.pt')
        received = json.loads(args.received_manifest.read_text())
        headers = json.loads(args.headers.read_text())
        policies = {'FIFO': StaticPolicyAdapter(SemanticScheduler('fifo', args.profile)),
                    'I': StaticPolicyAdapter(SemanticScheduler('importance', args.profile))}
        if args.policy_artifact is not None:
            policies['learned'] = load_contextual_policy(args.policy_artifact)
        results = {}
        for name, policy in policies.items():
            print(f'Replay {name}: {len(received["records"])} cached received payloads', flush=True)
            results[name] = replay_received_policy(adapter, received, headers, policy,
                                                    args.received_manifest.parent, args.output/name,
                                                    checkpoint_sha, args.profile, args.device,
                                                    header_scope='source_frozen_classifier_partial_parameters')
        hashes = [result['source_payloads'] for result in results.values()]
        if any(value != hashes[0] for value in hashes[1:]):
            raise RuntimeError('policies did not use the same received payloads')
        write_json(args.output/'summary.json', dict(status='complete', results=results,
                                                   actual_codec_executed=any(result['actual_codec_returned_successfully']
                                                                             for result in results.values()),
                                                   shared_payload_hashes_verified=True,
                                                   semantic_fact_preservation='not_independently_assessed'))
    except Exception as error:
        write_json(args.output/'failure.json', dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
