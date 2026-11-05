"""CPU-only, one-family development pilot. Never exports a production scheduler.

C has historical development exposure. A grouped inner search is selection evidence,
not nested final evaluation. Native image IDs are singleton observation units.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
import traceback

import numpy as np
import sklearn
from sklearn.linear_model import Ridge

from Scheduler.training import decoder_v3_study as v3
from Scheduler.training import decoder_parameter_audit as audit
from Scheduler.decoder_scheduler import SemanticScheduler, feature_matrix, tie_key
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import TransmitterMemory, ParameterHeader, ImageEvidence, ParameterExtractor, TemperatureCalibration

ALPHAS = (.1, 1., 10.)
METRICS = ('A10', 'coverage10', 'high_recall10', 'redundancy10')


def metric_reference(order, units, utilities, k=10, success=None):
    """Independent set-based reference; failed decodes consume a slot, gain nothing."""
    if sorted(order) != list(range(len(units))) or not 1 <= k <= len(units):
        raise ValueError('full permutation and valid cutoff required')
    weights = {}
    for unit, weight in zip(units, utilities):
        if not math.isfinite(weight) or weight < 0 or (unit in weights and weights[unit] != weight):
            raise ValueError('inconsistent or invalid singleton weight')
        weights[unit] = weight
    total = math.fsum(weights.values())
    success = [True] * len(units) if success is None else success
    if len(success) != len(units):
        raise ValueError('one success flag per job required')
    covered = set()
    curve = []
    rewards = []
    for index in order[:k]:
        before = math.fsum(weights[u] for u in covered)
        if success[index]:
            covered.add(units[index])
        after = math.fsum(weights[u] for u in covered)
        rewards.append(None if total == 0 else (after-before)/total)
        curve.append(None if total == 0 else after/total)
    high = {u for u, weight in weights.items() if weight == 3}
    return dict(A10=None if total == 0 else math.fsum(curve)/k,
                coverage10=curve[-1], high_recall10=None if not high else len(high & covered)/len(high),
                redundancy10=1-len(covered)/k, curve=curve, rewards=rewards)


def singleton_optimum(units, utilities, k=10):
    """Exact because each job covers one unit, all costs equal, weights nonnegative."""
    weights = dict(zip(units, utilities))
    total = math.fsum(weights.values())
    if total == 0:
        return None
    descending = sorted(weights.values(), reverse=True)
    return [math.fsum(descending[:min(i, len(descending))])/total for i in range(1, k+1)]


def oracle_order(episode):
    remaining = set(range(len(episode.units)))
    seen = set()
    order = []
    while remaining:
        chosen = min(remaining, key=lambda i: (-(0 if episode.units[i] in seen else episode.utilities[i]),
                                               tie_key(episode.candidates[i])))
        remaining.remove(chosen)
        seen.add(episode.units[chosen])
        order.append(chosen)
    return order


def multif_act_example():
    """Synthetic multi-fact headroom, deliberately outside the singleton benchmark."""
    jobs = [set('a'+str(i) for i in range(1,4)) | set('b'+str(i) for i in range(1,4)),
            set('a'+str(i) for i in range(1,6)), set('b'+str(i) for i in range(1,6))]
    # Ten facts, each weight .1. A first covers .6; then B/C adds .2.
    # A10-like two-slot mean: greedy (.6+.8)/2=.70; optimum (.5+1)/2=.75.
    score = lambda order: sum(len(set().union(*(jobs[i] for i in order[:k])))/10 for k in (1,2))/2
    return dict(scope='synthetic_multi_fact_only', greedy=score((0,1)),
                optimal=max(score(order) for order in itertools.permutations(range(3), 2)))


def retained_development(records):
    """Filter before inspecting native truth, calibrating or constructing queues."""
    if any(row['role'] not in ('A', 'B', 'C', 'test') for row in records):
        raise ValueError('unexpected role')
    return [row for row in records if row['role'] in ('A', 'B', 'C')]


def group_folds(records, folds, seed):
    if any(row['role'] != 'A' for row in records):
        raise ValueError('selection requires A only')
    groups = sorted({row['group'] for row in records})
    if len(groups) < folds * 32:
        raise ValueError('insufficient A duplicate groups for held-fold 32-job pools')
    shuffled = np.random.default_rng(seed).permutation(groups)
    return [set(shuffled[i::folds]) for i in range(folds)]


def subsets(limit=None):
    result = [tuple(i for i in range(8) if bits & (1 << i)) for bits in range(1,256)]
    return result if limit is None else result[:limit]


def selection_key(row):
    return (-row['score'], len(row['fields']), -row['alpha'], tuple(row['fields']))


def choose(rows):
    best = max(row['score'] for row in rows)
    near = [row for row in rows if best-row['score'] <= 1e-12]
    return min(near,key=lambda row:(len(row['fields']),-row['alpha'],tuple(row['fields'])))


def fit_arrays(episodes):
    medians = v3.fit_medians(episodes)
    candidates = [candidate for episode in episodes for candidate in episode.candidates]
    units = [unit for episode in episodes for unit in episode.units]
    counts = Counter(units)
    weights = np.asarray([1/counts[unit] for unit in units])
    matrix = feature_matrix(candidates, medians)
    mean = np.average(matrix, axis=0, weights=weights)
    # Explicit square deviations avoid cancellation/negative variance near constants.
    variance = np.average((matrix-mean)**2, axis=0, weights=weights)
    scale = np.where((variance > 0) & (np.ptp(matrix,axis=0) != 0), np.sqrt(variance), 1.)
    target = np.asarray([utility/3 for episode in episodes for utility in episode.utilities])
    return medians, mean, scale, (matrix-mean)/scale, target, weights


def columns(fields):
    return list(fields) + [field+8 for field in fields]


def equal_pool_mean(rows, metric='A10'):
    values = audit.pool_means(rows, metric)
    return None if not values else float(np.mean(list(values.values())))


def model_rows(episodes, predictor, model, task):
    rows = []
    for episode in episodes:
        order = predictor(episode)
        metrics = metric_reference(order, episode.units, episode.utilities)
        rows.append(dict(task=task, model=model, pool=episode.pool_id, stratum=episode.stratum,
                         episode=episode.episode_id, **{key: metrics[key] for key in METRICS}))
    return rows


def fitted_predictor(arrays, fields, alpha):
    medians, mean, scale, matrix, target, weights = arrays
    indices = columns(fields)
    model = Ridge(alpha=alpha).fit(matrix[:,indices], target, sample_weight=weights)
    def predict(episode):
        transformed = (feature_matrix(episode.candidates, medians)-mean)/scale
        return audit.order_scores(episode.candidates, model.predict(transformed[:,indices]))
    return predict


def bank(records, profile, calibration, repeats, role):
    return v3.evaluation_bank(records, role, profile, calibration, repeats)


def feature_search(records, profile, calibration, config, progress):
    a = [row for row in records if row['role'] == 'A']
    folds = group_folds(a, config['folds'], config['seed'])
    cached = []
    memberships = []
    for number, held in enumerate(folds):
        train_records = [row for row in a if row['group'] not in held]
        held_records = [row for row in a if row['group'] in held]
        pool = v3.eligible_records(train_records, 'A', profile)
        train = [v3.make_episode(pool, profile, v3.STRATA[i % 5], config['seed']+i,
                                 calibration, f'inner-train-{number}') for i in range(config['train_episodes'])]
        evaluation = bank(held_records, profile, calibration, config['repeats'], 'A')
        for episode_index, episode in enumerate(evaluation):
            evaluation[episode_index] = replace(episode, pool_id=f'fold{number}-{episode.pool_id}')
        used = {u for episode in evaluation for u in episode.units}
        memberships.append(dict(fold=number, train_groups=sorted({r['group'] for r in train_records}),
                                held_groups=sorted(held), unused_held_images=sorted(r['image_id'] for r in held_records if r['image_id'] not in used)))
        cached.append((fit_arrays(train), evaluation))
    grid = []
    fold_rows = []
    candidates = subsets(config['subset_limit'])
    # I/all controls retain their own selected alphas even in restricted smoke.
    candidates = list(dict.fromkeys(candidates + [(3,), tuple(range(8))]))
    for index, fields in enumerate(candidates):
        for alpha in ALPHAS:
            scores = []
            pool_counts = []
            for number, (arrays, evaluation) in enumerate(cached):
                rows = model_rows(evaluation, fitted_predictor(arrays, fields, alpha), 'inner', profile.task)
                score = equal_pool_mean(rows)
                if score is None:
                    raise ValueError('undefined inner-fold objective')
                scores.append(score)
                pool_counts.append(len(audit.pool_means(rows)))
                fold_rows.append(dict(fields=','.join(map(str,fields)), alpha=alpha, fold=number, A10=score, pools=pool_counts[-1],
                                      undefined_episodes=sum(row['A10'] is None for row in rows)))
            grid.append(dict(fields=fields, alpha=alpha, score=float(np.average(scores,weights=pool_counts))))
        if index % 16 == 0:
            progress(f'{profile.task}: subsets {index+1}/{len(candidates)}')
    allowed = set(subsets(config['subset_limit']))
    selected = choose([row for row in grid if row['fields'] in allowed])
    controls = {name: choose([row for row in grid if row['fields'] == fields])
                for name, fields in [('I-only', (3,)), ('all-fields', tuple(range(8)))]}
    return selected, controls, grid, fold_rows, memberships


def mechanics(episodes):
    memory = TransmitterMemory('fixture', 1001)
    before = memory.snapshot()
    empty = memory.score_history([1.,0.], [])['N']
    assert memory.snapshot() == before
    memory.commit(dict(image_id='fixture', embedding=[1.,0.], concepts=[]))
    same = memory.score_history([1.,0.], [])['N']
    orthogonal = memory.score_history([0.,1.], [])['N']
    assert (empty, same, orthogonal) == (1.,0.,.5)
    header = ParameterHeader(1001, 0, 0, 1, tuple(np.linspace(0,1,8)), (True,False)*4)
    parsed = ParameterHeader.from_bytes(header.to_bytes())
    assert parsed.available == header.available
    assert all(abs(a-b) <= .5/255+1e-12 for a,b,m in zip(header.values,parsed.values,header.available) if m)
    profile = v3.PROFILES[1001]
    absent = ImageEvidence('confident-absence', logits={name:-12. for name in profile.label_names})
    bundle = ParameterExtractor().extract(absent, profile, {}, TransmitterMemory('absent',1001),
                                          TemperatureCalibration(1001,1.,True))
    assert bundle.available[2] and bundle.values[2] > .99 and bundle.values[3] < .01
    episode = episodes[0]
    fifo = v3.replay_stream(SemanticScheduler('fifo'), episode, no_backlog=True)
    importance = v3.replay_stream(SemanticScheduler('importance'), episode, no_backlog=True)
    assert fifo == importance
    candidates = [candidate for e in episodes for candidate in e.candidates]
    health = {}
    for index, name in enumerate(('R','S','Q','I','F','P','C','N')):
        values = [c.values[index] for c in candidates if c.available[index]]
        health[name] = dict(available=len(values), total=len(candidates),
                            occupied_8bit_levels=sorted({int(np.floor(255*value+.5)) for value in values}),
                            occupied_deciles=sorted({min(9,int(value*10)) for value in values}))
    return dict(kb_empty_N=empty, kb_same_N=same, kb_orthogonal_N=orthogonal,
                header_half_lsb=True, confident_absence_Q=bundle.values[2], confident_absence_I=bundle.values[3], no_backlog_invariance=True, health=health)


def run(cache, output, seed=20261024, folds=3, train_episodes=24, repeats=1, subset_limit=None):
    output, cache = Path(output), Path(cache)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    if folds != 3 or train_episodes < 1 or repeats < 1 or (subset_limit is not None and not 1 <= subset_limit <= 255):
        raise ValueError('requires folds=3, positive episodes/repeats and subset-limit in 1..255')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = dict(seed=seed, folds=folds, train_episodes=train_episodes, repeats=repeats, subset_limit=subset_limit)
    def progress(stage):
        state = dict(stage=stage, elapsed_seconds=time.perf_counter()-started)
        print(json.dumps(state), flush=True)
        v3.write_json(output/'progress.json', state)
    try:
        sources = [Path(__file__), Path(v3.__file__), Path(audit.__file__),
                   Path('Scheduler/decoder_scheduler.py'), Path('Archived/owners_before_simplification_20261027/SemParam/decoder_parameters.py')]
        provenance = dict(config=config, cache=str(cache.resolve()), interpreter=sys.executable, python=sys.version,
                          platform=platform.platform(), numpy=np.__version__, sklearn=sklearn.__version__,
                          revision=audit.git('rev-parse','HEAD'), dirty=audit.git('status','--porcelain'),
                          source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                          cache_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file()})
        v3.write_json(output/'provenance.json', provenance)
        progress('integrity read, then test exclusion')
        raw = v3.load_cache(cache)
        quality = v3.verify_extraction_cache(cache, raw)
        records = retained_development(raw)
        del raw
        rows, summaries = [], {}
        for profile in v3.PROFILES.values():
            progress(f'{profile.task}: B calibration')
            calibration = v3.calibrate(records, profile)
            selected, controls, grid, fold_rows, memberships = feature_search(records, profile, calibration, config, progress)
            task_dir = output/profile.task
            task_dir.mkdir()
            v3.write_json(task_dir/'grid.json', grid)
            audit.write_csv(task_dir/'fold_scores.csv', fold_rows)
            v3.write_json(task_dir/'fold_membership.json', memberships)
            fold_choices = []
            for number in range(folds):
                choices = [dict(fields=tuple(int(i) for i in r['fields'].split(',')),alpha=r['alpha'],score=r['A10'])
                           for r in fold_rows if r['fold']==number and tuple(int(i) for i in r['fields'].split(',')) in set(subsets(subset_limit))]
                fold_choices.append(choose(choices))
            best_score = max(r['score'] for r in grid if r['fields'] in set(subsets(subset_limit)))
            near = [r for r in grid if r['fields'] in set(subsets(subset_limit)) and best_score-r['score'] <= .001]
            v3.write_json(task_dir/'selection.json', dict(selected=selected, controls=controls,
                exploratory_fold_choices=fold_choices, exploratory_near_best_tolerance=.001,
                near_best_field_inclusion={str(i):sum(i in r['fields'] for r in near)/len(near) for i in range(8)},
                fit_count_inner=len(fold_rows),undefined_fold_scores=sum(r['A10'] is None for r in fold_rows)))
            # C construction and scoring occur only after the A-only selection freezes.
            evaluation = bank(records, profile, calibration, repeats, 'C')
            task_rows = []
            policies = {'FIFO':SemanticScheduler('fifo'), 'RR':SemanticScheduler('round_robin'),
                        'I':SemanticScheduler('importance')}
            policies.update({f'random-{s}':SemanticScheduler('random',seed=s) for s in (seed,seed+1,seed+2)})
            for name, policy in policies.items():
                task_rows.extend(model_rows(evaluation, lambda e,p=policy:v3.episode_order(p,e), name, profile.task))
            task_rows.extend(model_rows(evaluation, lambda e:audit.order_scores(e.candidates,[-c.values[3] if c.available[3] else -.5 for c in e.candidates]), 'reversed-I', profile.task))
            def shuffled(e):
                permutation = np.random.default_rng(seed+int(hashlib.sha256(e.episode_id.encode()).hexdigest()[:8],16)).permutation(len(e.candidates))
                candidates = tuple(replace(c, values=e.candidates[j].values, available=e.candidates[j].available)
                                   for c,j in zip(e.candidates,permutation))
                return v3.episode_order(SemanticScheduler('importance'), replace(e,candidates=candidates))
            for name,predictor in [('shuffled-semantic-vector',shuffled), ('oracle-utility',lambda e:audit.order_scores(e.candidates,e.utilities)),
                                   ('oracle-unit-greedy',oracle_order), ('signature_order',lambda e:audit.signature_order(e.candidates))]:
                task_rows.extend(model_rows(evaluation,predictor,name,profile.task))
            optimum_rows = []
            for episode in evaluation:
                optimum = singleton_optimum(episode.units,episode.utilities)
                observed = metric_reference(oracle_order(episode),episode.units,episode.utilities)['curve']
                assert (optimum is None and all(x is None for x in observed)) or np.allclose(optimum,observed)
                optimum_rows.append(dict(episode=episode.episode_id,curve=optimum))
            v3.write_json(task_dir/'analytic_optimum.json', optimum_rows)
            pool = v3.eligible_records(records,'A',profile)
            train = [v3.make_episode(pool,profile,v3.STRATA[i%5],seed+i,calibration) for i in range(train_episodes)]
            arrays = fit_arrays(train)
            for name, choice in dict(selected=selected, **controls).items():
                task_rows.extend(model_rows(evaluation,fitted_predictor(arrays,choice['fields'],choice['alpha']),name,profile.task))
            grouped = defaultdict(list)
            for row in task_rows:
                grouped[row['model']].append(row)
            summaries[profile.task] = dict(selection=selected, mechanics=mechanics(evaluation),
                unused_C_images=sorted(r['image_id'] for r in v3.eligible_records(records,'C',profile) if r['image_id'] not in {u for e in evaluation for u in e.units}),
                aggregates={name:{metric:equal_pool_mean(items,metric) for metric in METRICS} for name,items in grouped.items()},
                per_stratum={name:{stratum:{metric:equal_pool_mean([r for r in items if r['stratum']==stratum],metric) for metric in METRICS} for stratum in v3.STRATA} for name,items in grouped.items()},
                undefined_episodes=sorted({r['episode'] for r in task_rows if r['A10'] is None}))
            rows.extend(task_rows)
            audit.write_csv(output/'per_episode.csv',rows)
            v3.write_json(output/'partial_summary.json',summaries)
        result = dict(status='complete',scope='restricted_smoke' if subset_limit else 'one_family_development_pilot_not_nested_final',
                      C='historically exposed development',test='excluded before any label use beyond cache integrity',
                      tasks=summaries,quality=quality,synthetic_multi_fact=multif_act_example())
        v3.write_json(output/'summary.json',result)
        progress('complete')
        return result
    except Exception as error:
        v3.write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--train-episodes',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--subset-limit',type=int)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
