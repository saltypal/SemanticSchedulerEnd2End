"""Completion-aware CPU study on explicitly development-exposed real features.

No new data, labels, physical facts or codec success are invented. The native
cache arm is a singleton-observation control, not natural cross-view semantics.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.decoder_service import DecoderService
from Evaluation.decoder_semantic_benchmark import coverage_metrics, marginal_gain
from Scheduler.training import decoder_v3_study as legacy
from Scheduler.training.decoder_context_training import TrainingDecision, fit_policy, subset_candidates, choose_config
from Scheduler.contextual_decoder_scheduler import DecisionContext, StaticPolicyAdapter, OverlapPenaltyPolicy
from Scheduler.decoder_scheduler import SemanticScheduler, SchedulerCandidate, tie_key
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterExtractor, ParameterHeader, TransmitterMemory, get_profile
from Archived.owners_before_simplification_20261027.SemParam.decoder_context_profiles import STUDY_PROFILES, KNOWN_PROFILES, TRANSFER_PROFILES


@dataclass(frozen=True)
class NativeEpisode:
    episode_id: str
    pool: str
    task_profile: int
    stratum: str
    messages: tuple
    facts: dict
    weights: dict


class SignaturePolicy:
    profile_id = None

    def select(self, context):
        def signature(row):
            fields = (0, 2, 3, 5, 6)
            return tuple((row.available[i], row.values[i] if row.available[i] else None) for i in fields)
        served = {signature(row) for row in context.successful_header_history}
        choices = sorted(context.ready_candidates,
                         key=lambda row: (signature(row) in served, -row.values[3], tie_key(row)))
        if not choices:
            return None
        from Scheduler.decoder_scheduler import Decision
        return Decision(choices[0].message_id, 'header_signature', None)

    def reset(self):
        pass


def eligible(records, role):
    # Source native labels for known profiles only are consumed here.
    unique = {}
    for row in sorted(records, key=lambda row: row['image_id']):
        if row['role'] == role:
            unique.setdefault(row['group'], row)
    return list(unique.values())


def make_episode(pool, profile_id, calibration, seed, pool_name, stratum='no_repeat'):
    if len(pool) < 32:
        raise ValueError('native episode requires 32 distinct source groups')
    profile = get_profile(profile_id, STUDY_PROFILES)
    rng = np.random.default_rng(seed)
    chosen = [pool[i] for i in rng.permutation(len(pool))[:32]]
    if len({row['group'] for row in chosen}) != 32:
        raise ValueError('duplicate groups in episode source pool')
    if stratum not in legacy.STRATA:
        raise ValueError('unsupported repeat stratum')
    count = 8 if stratum == 'no_repeat' else 6 if stratum.endswith('25') else 4
    originals = [chosen[i*count:(i+1)*count] for i in range(4)]
    observations = [list(rows) for rows in originals]
    for source in range(4):
        other = (source+1) % 4 if stratum.startswith('cross') else source
        observations[source].extend(originals[other][i % count] for i in range(8-count))
    episode_id = f'{profile.task}:{pool_name}:{stratum}:{seed}'
    entries = []
    for source in range(4):
        memory = TransmitterMemory(f'ue{source}', profile_id, registry=STUDY_PROFILES)
        for token, row in enumerate(observations[source]):
            evidence = row['evidence']
            bundle = ParameterExtractor().extract(evidence, profile, {}, memory, calibration)
            header = ParameterHeader(profile_id, token, memory.version, 0,
                                     bundle.values, bundle.available, STUDY_PROFILES)
            wire = header.to_bytes()
            memory.commit(evidence)
            # Exactly one native boolean for these additive binary profiles.
            label = row['labels'].get(profile.label_names[0])
            if type(label) not in (bool, int) or label not in (0, 1):
                raise ValueError('missing/unsupported native binary truth')
            fact = (row['image_id'],) if label else ()
            entries.append((source, token, wire, fact))
    messages, facts, weights = [], {}, {}
    for sequence, index in enumerate(rng.permutation(32)):
        source, token, wire, fact = entries[index]
        message = ReceivedMessage(f'ue{source}', episode_id, token, wire, index, True, 0., sequence)
        messages.append(message)
        facts[message.message_id] = fact
        weights.update({f: 1. for f in fact})
    return NativeEpisode(episode_id, pool_name, profile_id, stratum, tuple(messages), facts, weights)


def bank(pool, profile_ids, calibrations, seed=0, repeats=1, pool_prefix='development'):
    output = []
    for start in range(0, len(pool)-31, 32):
        block = pool[start:start+32]
        for profile_id in profile_ids:
            for index, stratum in enumerate(legacy.STRATA):
                for repeat in range(repeats):
                    output.append(make_episode(block, profile_id, calibrations[profile_id],
                                               seed+1000*start+100*index+repeat,
                                               f'{pool_prefix}-{start//32}', stratum))
    if not output:
        raise ValueError('insufficient groups for a complete evaluation pool')
    return output


def candidates(episode):
    return tuple(SchedulerCandidate.from_header(m.message_id, m.source_id, m.arrival_time, m.sequence,
                                                ParameterHeader.from_bytes(m.header_bytes, STUDY_PROFILES))
                 for m in episode.messages)


def training_decisions(episodes, seed=0, pointwise=False, overlap=False):
    output = []
    rng = np.random.default_rng(seed)
    for number, episode in enumerate(episodes):
        if not episode.weights:
            continue
        ready = list(candidates(episode))
        if overlap:
            # Directional overlap is defined only for candidates with useful facts.
            pairs = [(a, b) for a in ready for b in ready if a.message_id != b.message_id
                     and episode.facts[a.message_id]]
            for index in rng.permutation(len(pairs))[:64]:
                a, b = pairs[index]
                source = set(episode.facts[a.message_id])
                target = len(source & set(episode.facts[b.message_id]))/len(source)
                output.append(TrainingDecision(DecisionContext((a,), (b,), 1, 2, episode.task_profile),
                                               (target,), 1/max(1, min(64, len(pairs)))))
            continue
        history = []
        for step in range(10):
            context = DecisionContext(tuple(ready), tuple(history), step, len(episode.messages), episode.task_profile)
            delivered = [] if pointwise else [row.message_id for row in history]
            targets = tuple(marginal_gain(row.message_id, delivered, episode.facts, episode.weights) for row in ready)
            output.append(TrainingDecision(context, targets, .1))
            if number % 2:
                selected = ready[int(rng.integers(len(ready)))]
            else:
                selected = min(ready, key=lambda row: (-row.values[3], tie_key(row)))
            ready.remove(selected)
            history.append(selected)
    return output


def replay(policy, episode):
    policy.reset()
    service = DecoderService(policy, episode.task_profile, STUDY_PROFILES)
    for message in episode.messages:
        service.enqueue(message)
    order, trace = [], []
    while len(service):
        now = float(len(order))
        dispatched = service.dispatch(now)
        identity = dispatched.message.message_id
        order.append(identity)
        service.complete(identity, 'succeeded', now+1.)
        trace.append(dict(message_id=identity, score=dispatched.decision.score,
                          fallback=dispatched.decision.fallback, completed_at=now+1.))
    metrics = coverage_metrics(order, episode.facts, episode.weights, 10)
    return dict(episode=episode.episode_id, pool=episode.pool, profile=episode.task_profile,
                stratum=episode.stratum, **metrics, order=order, trace=trace,
                execution_scope='ideal_decoder_abstract_service_ticks')


def mean_score(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['A_K'] is not None:
            groups[(row['profile'], row['pool'])][row['stratum']].append(row['A_K'])
    task_values = defaultdict(list)
    for (profile, _), strata in groups.items():
        task_values[profile].append(float(np.mean([np.mean(values) for values in strata.values()])))
    return None if not task_values else float(np.mean([np.mean(values) for values in task_values.values()]))


def grouped_folds(pool, count=3, seed=0):
    groups = sorted({row['group'] for row in pool})
    if len(groups) < count*32:
        raise ValueError('insufficient independent groups for complete held-fold queues')
    shuffled = np.random.default_rng(seed).permutation(groups)
    return [set(shuffled[i::count]) for i in range(count)]


def select_linear(pool, calibrations, config, progress):
    folds = grouped_folds(pool, 3, config['seed'])
    data = []
    for i, held in enumerate(folds):
        fit_pool = [row for row in pool if row['group'] not in held]
        validation_pool = [row for row in pool if row['group'] in held]
        fit_episodes = [make_episode(fit_pool, p, calibrations[p], config['seed']+j,
                                    f'inner-fit-{i}', legacy.STRATA[j % 5])
                        for p in KNOWN_PROFILES for j in range(config['train_episodes'])]
        decisions = training_decisions(fit_episodes, config['seed'])
        evaluation = bank(validation_pool, KNOWN_PROFILES, calibrations, config['seed'], pool_prefix=f'inner-held-{i}')
        data.append((decisions, evaluation))
    grid, fingerprints = [], {}
    from Scheduler.training.decoder_context_training import matrices, fit_medians
    for fields in subset_candidates(config.get('subset_limit')):
        # Fingerprint across ALL folds. Constant/missing fields may make subsets identical.
        hashes = []
        for decisions, _ in data:
            matrix, _, _ = matrices(decisions, fit_medians(decisions), fields)
            hashes.append(hashlib.sha256(matrix.tobytes()).hexdigest())
        key = tuple(hashes)
        if key in fingerprints:
            for original in fingerprints[key]:
                grid.append({**original, 'fields': fields, 'equivalent_to': original['fields']})
            continue
        local = []
        for alpha in (.1, 1., 10.):
            scores = []
            for decisions, evaluation in data:
                model = fit_policy(decisions, fields, alpha=alpha)
                score = mean_score([replay(model, episode) for episode in evaluation])
                if score is None:
                    raise ValueError('undefined held-fold selection objective')
                scores.append(score)
            row = dict(fields=fields, alpha=alpha, mean_A_K=float(np.mean(scores)), fold_scores=scores)
            local.append(row)
            grid.append(row)
        fingerprints[key] = local
        progress(f'inner subsets {len(grid)//3}/{len(subset_candidates(config.get("subset_limit")))}')
    return choose_config(grid), grid


def run(cache, output, train_episodes=4, subset_limit=3, seed=20261024, nested=False):
    from Scheduler.training.decoder_parameter_audit import git
    from Evaluation.decoder_datasets import sha256
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be new or empty')
    if train_episodes < 1:
        raise ValueError('positive training episode count required')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = dict(train_episodes=train_episodes, subset_limit=subset_limit, seed=seed, nested=nested)
    def save(name, value):
        legacy.write_json(output/name, value)
    def progress(stage):
        state = dict(stage=stage, elapsed_seconds=time.perf_counter()-started)
        print(json.dumps(state), flush=True)
        save('progress.json', state)
    try:
        progress('verify cache and exclude historic test before experiment use')
        records = legacy.load_cache(cache)
        quality = legacy.verify_extraction_cache(cache, records)
        records = [row for row in records if row['role'] in ('A', 'B', 'C')]
        source_paths = [Path(__file__), Path('Scheduler/training/decoder_context_training.py'),
                        Path('Scheduler/contextual_decoder_scheduler.py'), Path('Evaluation/decoder_service.py'),
                        Path('Evaluation/decoder_queue.py'), Path('Archived/owners_before_simplification_20261027/SemParam/decoder_parameters.py')]
        save('manifest.json', dict(revision=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain'),
                                   interpreter=sys.executable, python=sys.version, platform=platform.platform(),
                                   numpy=np.__version__, config=config, quality=quality,
                                   source_hashes={str(p): sha256(p) for p in source_paths},
                                   cache_hashes={name: sha256(Path(cache)/name) for name in ('features.jsonl', 'labels.jsonl')},
                                   data_scope='exposed_LADI_singleton_control_no_new_final_holdout'))
        calibrations = {p: legacy.calibrate(records, get_profile(p, STUDY_PROFILES)) for p in KNOWN_PROFILES}
        pool = eligible(records, 'A')
        if nested:
            outer_rows = []
            for number, held in enumerate(grouped_folds(pool, 3, seed)):
                fit_pool = [row for row in pool if row['group'] not in held]
                held_pool = [row for row in pool if row['group'] in held]
                choice, grid = select_linear(fit_pool, calibrations, config, progress)
                train = [make_episode(fit_pool, p, calibrations[p], seed+i, f'outer-fit-{number}', legacy.STRATA[i % 5])
                         for p in KNOWN_PROFILES for i in range(train_episodes)]
                model = fit_policy(training_decisions(train, seed), choice['fields'], alpha=choice['alpha'])
                evaluated = [replay(model, e) for e in bank(held_pool, KNOWN_PROFILES, calibrations, seed,
                                                          pool_prefix=f'outer-held-{number}')]
                outer_rows.append(dict(fold=number, selected=choice, mean_A_K=mean_score(evaluated),
                                       groups=sorted(held), rows=evaluated, grid=grid))
                save('nested_outer.json', outer_rows)
        choice, grid = select_linear(pool, calibrations, config, progress)
        save('selection.json', dict(selected=choice, grid=grid, selection_role='A_grouped_inner_only',
                                     calibration_role='fixed_external_B_disjoint_from_A_and_C',
                                     final_test_scope='none; C is exposed development'))
        episodes = [make_episode(pool, p, calibrations[p], seed+i, 'final-A-fit', legacy.STRATA[i % 5])
                    for p in KNOWN_PROFILES for i in range(train_episodes)]
        decisions = training_decisions(episodes, seed)
        standalone = training_decisions(episodes, seed, pointwise=True)
        models = {
            'contextual_linear': fit_policy(decisions, choice['fields'], alpha=choice['alpha']),
            'pointwise_linear': fit_policy(standalone, (3,), alpha=10., pointwise=True),
            'contextual_tree': fit_policy(decisions, choice['fields'], kind='boosted_trees', seed=seed),
            'conditioned_linear': fit_policy(decisions, choice['fields'], alpha=10., task_ids=KNOWN_PROFILES),
            'masks_only': fit_policy(decisions, masks_only=True),
        }
        for p in KNOWN_PROFILES:
            models[f'specialist_{p}'] = fit_policy([d for d in decisions if d.context.task_profile == p],
                                                 choice['fields'], alpha=10.)
        overlap = fit_policy(training_decisions(episodes, seed, overlap=True), kind='linear')
        model_dir = output/'models'
        model_dir.mkdir()
        for name, model in {**models, 'overlap_predictor': overlap}.items():
            legacy.write_json(model_dir/f'{name}.json', model.artifact)
        frozen_hashes = {p.name: sha256(p) for p in model_dir.glob('*.json')}
        save('model_freeze.json', frozen_hashes)
        evaluation = bank(eligible(records, 'C'), KNOWN_PROFILES, calibrations, seed, pool_prefix='exposed-C')
        policies = {**models,
                    **{name: StaticPolicyAdapter(SemanticScheduler(kind, seed=seed)) for name, kind in
                       [('FIFO', 'fifo'), ('RR', 'round_robin'), ('I', 'importance'), ('random', 'random')]},
                    'header_signature': SignaturePolicy(),
                    'overlap_penalty': OverlapPenaltyPolicy(models['pointwise_linear'], overlap)}
        rows = []
        for name, model in policies.items():
            progress(f'evaluate {name} on exposed C')
            for episode in evaluation:
                if name.startswith('specialist_') and name != f'specialist_{episode.task_profile}':
                    continue
                rows.append(dict(policy=name, **replay(model, episode)))
        save('known_task_development.json', rows)
        # Held-task truth is accessed only after model artifacts/choices freeze.
        progress('frozen scheduler structural-flooding transfer; no target calibration')
        transfer_cal = {p: legacy.calibrate([], get_profile(p, STUDY_PROFILES)) for p in TRANSFER_PROFILES}
        transfer = bank(eligible(records, 'C'), TRANSFER_PROFILES, transfer_cal, seed, pool_prefix='transfer-C')
        transfer_rows = []
        for name in ('contextual_linear', 'pointwise_linear', 'FIFO', 'RR', 'I'):
            for episode in transfer:
                transfer_rows.append(dict(policy=name, **replay(policies[name], episode)))
        save('unseen_scheduler_task_development.json', transfer_rows)
        if frozen_hashes != {p.name: sha256(p) for p in model_dir.glob('*.json')}:
            raise RuntimeError('model artifacts changed during held-task evaluation')
        summary = dict(status='complete', scope='native_label_control_development_only',
                       subset_scope='all255' if subset_limit is None else f'restricted_{subset_limit}',
                       selected=choice, nested=nested, actual_codec_executed=False,
                       known={name: mean_score([row for row in rows if row['policy'] == name]) for name in policies},
                       transfer={name: mean_score([row for row in transfer_rows if row['policy'] == name])
                                 for name in ('contextual_linear', 'pointwise_linear', 'FIFO', 'RR', 'I')},
                       elapsed_seconds=time.perf_counter()-started)
        save('summary.json', summary)
        progress('complete')
        return summary
    except Exception as error:
        save('failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-episodes', type=int, default=4)
    parser.add_argument('--subset-limit', type=int, default=3)
    parser.add_argument('--all-subsets', action='store_true')
    parser.add_argument('--nested', action='store_true')
    parser.add_argument('--seed', type=int, default=20261024)
    args = vars(parser.parse_args())
    if args.pop('all_subsets'):
        args['subset_limit'] = None
    run(**args)


if __name__ == '__main__':
    main()
