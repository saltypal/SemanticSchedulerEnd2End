"""Readable terminal replay, or live image extraction, using canonical V3 modules."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import (
    ImageEvidence, PARAMETER_NAMES, ParameterExtractor, ParameterHeader,
    PROFILES, TemperatureCalibration, TransmitterMemory, get_profile,
)
from Scheduler.decoder_scheduler import SchedulerCandidate, load_scheduler
from Evaluation.decoder_queue import DecoderQueue, ReceivedMessage

STUDY = ROOT / 'results/disaster_decoder_v3/20261023_stageB_v1/full/study'
CACHE = ROOT / 'results/disaster_decoder_v3/20261022_stageA_v3/extraction/features.jsonl'


def cached_evidence(path, count):
    """First held-out rows in file order; no labels, predictions or score filtering."""
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('role') == 'test':
                rows.append(ImageEvidence.from_dict(row))
                if len(rows) == count:
                    return rows
    raise ValueError(f'cache contains fewer than {count} test observations')


def run_demo(evidence, task, artifact, calibration, output, mode, add_repeat=True):
    profile = get_profile(task)
    scheduler = load_scheduler(artifact, PROFILES)
    receiver = DecoderQueue(scheduler, profile)
    memory = TransmitterMemory('ue-demo', profile.profile_id)
    extractor = ParameterExtractor()
    observations = list(evidence)
    if not observations:
        raise ValueError('at least one observation is required')
    if add_repeat:
        observations.append(observations[0])
    print(f'\nTASK: {profile.task} | MODEL: {scheduler.kind}')
    print(f'INPUT: {mode}')
    print('Freshness unavailable: no trusted capture timestamp is invented.')
    print('Payloads are observation references, NOT encoded latents. No decoding occurs.')
    print('Scores express ordering, not a probability or a causal explanation.\n')
    records, candidates = [], []
    for index, item in enumerate(observations):
        bundle = extractor.extract(item, profile, {}, memory, calibration)
        prior_version = memory.version
        header = ParameterHeader(profile.profile_id, index + 1, prior_version, 0,
                                 bundle.values, bundle.available)
        wire = header.to_bytes()
        parsed = ParameterHeader.from_bytes(wire)
        memory.commit(item)
        message = ReceivedMessage('ue-demo', 'terminal-demo', index + 1, wire,
                                  {'observation_id': item.image_id}, True, 0., index)
        receiver.enqueue(message)
        candidates.append(SchedulerCandidate.from_header(
            message.message_id, message.source_id, 0., index, parsed))
        fields = ' '.join(f'{name}={value:.3f}' if valid else f'{name}=NA'
                         for name, value, valid in zip(PARAMETER_NAMES, parsed.values, parsed.available))
        print(f'M{index+1} image={item.image_id[:14]} prior-history={prior_version} {fields}')
        print(f'   header={wire.hex()} ({len(wire)} bytes, CRC validated)')
        records.append(dict(message_id=message.message_id, image_id=item.image_id,
            prior_memory_version=prior_version, header_hex=wire.hex(),
            values=dict(zip(PARAMETER_NAMES, parsed.values)),
            available=dict(zip(PARAMETER_NAMES, parsed.available)),
            missing_reasons=bundle.reasons))
    ranking = scheduler.rank(tuple(candidates))
    print('\nHEADER-ONLY RANKING (all messages already arrived):')
    for rank, decision in enumerate(ranking, 1):
        token = json.loads(decision.message_id)[2]
        score = 'NA' if decision.score is None else f'{decision.score:.6f}'
        print(f'  {rank}. M{token} score={score}')
    dispatch = []
    while len(receiver):
        decision, message = receiver.dispatch(float(len(dispatch)))
        dispatch.append(dict(message_id=decision.message_id, score=decision.score,
                             observation_id=message.payload_ref['observation_id']))
    assert [d.message_id for d in ranking] == [d['message_id'] for d in dispatch]
    print('DISPATCH: ' + ' -> '.join(f'M{json.loads(d["message_id"])[2]}' for d in dispatch))
    print('Queue drained; no image, label or embedding was supplied to the policy.')
    print('This demonstrates behavior, NOT independent semantic correctness or codec performance.')
    artifact = Path(artifact)
    result = dict(task=profile.task, mode=mode, artifact=str(artifact.resolve()),
        artifact_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(artifact.iterdir()) if p.is_file()},
        codec_executed=False, repeat_inserted=add_repeat, headers=records, dispatch=dispatch)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'SAVED: {output.resolve()}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=[p.task for p in PROFILES.values()], default='building_damage')
    parser.add_argument('--model', choices=['ranknet', 'cahr', 'selected'], default='ranknet')
    parser.add_argument('--images', nargs='+', type=Path, help='Live extraction; downloads frozen models if absent')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--no-repeat', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 32:
        parser.error('--count must be 1..32')
    task_dir = STUDY / args.task
    selection = json.loads((task_dir / 'selection.json').read_text())
    if args.model == 'selected':
        artifact = task_dir / 'selected_scheduler'
    else:
        method = 'neural_ranknet' if args.model == 'ranknet' else 'neural_coverage'
        remote = selection['methods'][method]['artifact']
        artifact = STUDY / remote.split('/study/', 1)[1]
    calibration = TemperatureCalibration(**json.loads((task_dir / 'calibration.json').read_text()))
    if args.images:
        from PIL import Image
        from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import FrozenImageExtractor
        model = FrozenImageExtractor(device=args.device)
        evidence = []
        for path in args.images:
            identifier = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                evidence.append(model.extract(image.convert('RGB'), identifier))
        mode = 'live image-model extraction; parameters calculated now'
    else:
        evidence = cached_evidence(args.cache, args.count)
        mode = 'saved real-image model outputs; parameters/history/header calculated now (NOT live vision)'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'results/disaster_decoder_v3/terminal_demo' / f'{stamp}.json'
    run_demo(evidence, args.task, artifact, calibration, output, mode, not args.no_repeat)


if __name__ == '__main__':
    main()
