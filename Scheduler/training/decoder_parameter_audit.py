"""CPU-only exploratory audit of existing V3 headers; never a fitted scheduler export."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from Scheduler.training import decoder_v3_study as v3
from Scheduler.decoder_scheduler import PARAMETER_NAMES, feature_matrix, tie_key


def variants():
    fields = tuple(range(8))
    return {'I': (3,), 'all': fields, **{f'I+{n}': (3, i) for i, n in enumerate(PARAMETER_NAMES) if i != 3},
            **{f'all-minus-{n}': tuple(i for i in fields if i != j) for j, n in enumerate(PARAMETER_NAMES)},
            'masks-only': ()}


def design(candidates, medians, fields, masks_only=False):
    matrix = feature_matrix(candidates, medians)
    indices = list(range(8, 16)) if masks_only else list(fields) + [i + 8 for i in fields]
    return matrix[:, indices]


def order_scores(candidates, scores):
    return sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), tie_key(candidates[i])))


def signature(candidate, fields=(0, 2, 3, 5, 6)):
    if sum(candidate.available[i] for i in fields) < 2:
        return None
    return (candidate.profile_id, tuple(candidate.available[i] for i in fields),
            tuple(int(np.floor(candidate.values[i] * 255 + .5)) if candidate.available[i] else None for i in fields))


def signature_order(candidates):
    """Header-only deferral; all jobs retained. Missing signatures are never merged."""
    ranked = order_scores(candidates, [c.values[3] if c.available[3] else .5 for c in candidates])
    seen, first, deferred = set(), [], []
    for i in ranked:
        key = signature(candidates[i])
        if key is not None and key in seen:
            deferred.append(i)
        else:
            first.append(i)
            if key is not None:
                seen.add(key)
    return first + deferred


def pool_means(rows, metric='A10'):
    """Equal repetitions within stratum, then equal strata within independent pool."""
    strata = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            strata[(row['pool'], row['stratum'])].append(row[metric])
    pools = defaultdict(list)
    for (pool, _), values in strata.items():
        pools[pool].append(float(np.mean(values)))
    return {pool: float(np.mean(values)) for pool, values in pools.items()}


def comparison(left, right, bootstrap, seed):
    a, b = pool_means(left), pool_means(right)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError('no useful paired evaluation pools')
    differences = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(len(keys), size=(bootstrap, len(keys)))].mean(axis=1)
    return dict(pools=len(keys), delta_A10=float(differences.mean()),
                ci_low=float(np.quantile(samples, .025)), ci_high=float(np.quantile(samples, .975)))


def feature_audit(episodes):
    candidates = [c for e in episodes for c in e.candidates]
    values = np.array([c.values for c in candidates])
    masks = np.array([c.available for c in candidates])
    stats = {}
    for i, name in enumerate(PARAMETER_NAMES):
        x = values[masks[:, i], i]
        hist, edges = np.histogram(x, bins=np.linspace(0, 1, 11))
        stats[name] = dict(total=len(candidates), available=len(x), available_fraction=float(masks[:, i].mean()),
                           zero=int((x == 0).sum()), one=int((x == 1).sum()),
                           zero_fraction=float((x == 0).mean()) if len(x) else None,
                           min=float(x.min()) if len(x) else None, max=float(x.max()) if len(x) else None,
                           quartiles=np.quantile(x, [.25, .5, .75]).tolist() if len(x) else None,
                           empirically_testable=bool(len(x) and np.ptp(x) > 0),
                           distinct_quantized_levels=len(np.unique(np.floor(x * 255 + .5))),
                           histogram=hist.tolist(), edges=edges.tolist())
    correlations = []
    for i in range(8):
        for j in range(i + 1, 8):
            keep = masks[:, i] & masks[:, j]
            a, b = values[keep, i], values[keep, j]
            defined = len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0
            correlations.append(dict(left=PARAMETER_NAMES[i], right=PARAMETER_NAMES[j], n=len(a),
                                     pearson=float(np.corrcoef(a, b)[0, 1]) if defined else None,
                                     spearman=float(spearmanr(a, b).statistic) if defined else None))
    return stats, correlations


def collisions(episodes, fields):
    groups = defaultdict(list)
    for e in episodes:
        for c, unit, utility in zip(e.candidates, e.units, e.utilities):
            key = signature(c, fields)
            if key is not None:
                groups[key].append((unit, utility))
    return dict(signatures=len(groups), repeated_candidate_groups=sum(len(v) > 1 for v in groups.values()),
                multiple_original_units=sum(len(set(u for u, _ in v)) > 1 for v in groups.values()),
                conflicting_utility=sum(len(set(y for _, y in v)) > 1 for v in groups.values()),
                intended_same_unit_only=sum(len(v) > 1 and len(set(u for u, _ in v)) == 1 for v in groups.values()),
                scope='C candidate occurrences, repetitions included; cross-episode aliasing is descriptive')


def false_deferrals(episodes):
    false, intended, total = 0, 0, 0
    for e in episodes:
        seen = {}
        ranked = order_scores(e.candidates, [c.values[3] if c.available[3] else .5 for c in e.candidates])
        for i in ranked:
            key = signature(e.candidates[i])
            if key is None:
                continue
            if key in seen:
                total += 1
                if e.units[i] in seen[key]: intended += 1
                else: false += 1
            if key not in seen:
                seen[key] = {e.units[i]}
    return dict(total=total, different_original_unit=false, already_represented_unit=intended,
                scope='evaluator-only identities; deferral is not dropping or proof of physical fact loss')


def novelty_diagnostics(episodes, seed, stratify=True):
    scores, truth = [], []
    for e in episodes:
        episode_seed = int.from_bytes(hashlib.sha256(f'{seed}:{e.episode_id}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(episode_seed)
        prefix = set(rng.permutation(len(e.candidates))[:min(5, len(e.candidates))].tolist())
        seen = {e.units[i] for i in prefix}
        for i, c in enumerate(e.candidates):
            if i not in prefix and c.available[7]:
                scores.append(c.values[7]); truth.append(int(e.units[i] not in seen))
    x, y = np.asarray(scores), np.asarray(truth)
    result = dict(n=len(y), receiver_new=int(y.sum()), receiver_seen=int((y == 0).sum()),
                roc_auc=float(roc_auc_score(y, x)) if len(set(truth)) == 2 else None,
                low_N_receiver_new=int(((x < .5) & (y == 1)).sum()),
                high_N_receiver_seen=int(((x >= .5) & (y == 0)).sum()),
                scope='controlled repeated-observation mismatch; random five-job completed prefix; .5 illustrative untuned threshold')
    if stratify:
        result['per_stratum'] = {s: novelty_diagnostics([e for e in episodes if e.stratum == s], seed, False)
                                 for s in sorted({e.stratum for e in episodes})}
    return result


def counterexample():
    # Headers of jobs a,b are identical; the hidden prior contains a or b equiprobably.
    worlds = ((0., 1.), (1., 0.))
    return dict(world_marginal_gains=worlds, header_only_expected_first_gain=.5,
                privileged_expected_first_gain=1., bound='Any header-only randomized first decision has expectation 0.5 in these balanced worlds.',
                scope='pure synthetic identifiability illustration, not measured policy performance')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def label_support(records, profile):
    """Prevalence among eligible original-group representatives, including empty roles."""
    support, prevalence = {}, {}
    for role in ('A', 'B', 'C'):
        rows = v3.eligible_records(records, role, profile)
        support[role] = len(rows)
        prevalence[role] = {label: float(np.mean([r['labels'][label] for r in rows])) if rows else None
                            for label in profile.label_names}
    return support, prevalence


def run(cache, output, seed=20261024, repeats=2, train_episodes=64, bootstrap=1000):
    started = time.perf_counter()
    cache, output = Path(cache), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    if min(repeats, train_episodes, bootstrap) < 1:
        raise ValueError('repeats, train episodes and bootstrap must be positive')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(cache=str(cache.resolve()), output=str(output.resolve()), seed=seed, repeats=repeats,
                  train_episodes=train_episodes, bootstrap=bootstrap)
    provenance = dict(config=config, revision=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain'),
                      interpreter=sys.executable, python=sys.version, platform=platform.platform(),
                      numpy=np.__version__, sklearn=sklearn_version,
                      source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     (Path(__file__), Path(v3.__file__), Path('Scheduler/decoder_scheduler.py'), Path('Archived/owners_before_simplification_20261027/SemParam/decoder_parameters.py'))})
    v3.write_json(output/'provenance.json', provenance)
    def progress(stage):
        print(f'{stage} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        v3.write_json(output/'progress.json', dict(stage=stage, elapsed_seconds=time.perf_counter()-started))
    try:
        progress('validating cache')
        for name in ('features.jsonl', 'labels.jsonl'):
            path = cache/name
            if not path.exists() or path.read_bytes()[:100].startswith(b'version https://git-lfs'):
                raise ValueError(f'missing cache or Git LFS pointer: {path}')
            provenance.setdefault('cache_hashes', {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = v3.load_cache(cache)
        quality = v3.verify_extraction_cache(cache, records)
        if any(r['role'] not in ('A', 'B', 'C', 'test') for r in records):
            raise ValueError('unexpected cache role')
        # Exclude test records before any calibration, eligibility or label audit.
        records = [r for r in records if r['role'] in ('A', 'B', 'C')]
        provenance['roles'] = dict(Counter(r['role'] for r in records))
        v3.write_json(output/'provenance.json', provenance)
        all_rows, comparisons, feature_reports, correlation_reports, collision_reports, novelty_reports, tasks = [], [], {}, {}, {}, {}, {}
        for profile in v3.PROFILES.values():
            task = profile.task
            progress(f'{task}: calibration B and fixed A training queues')
            calibration = v3.calibrate(records, profile)
            pool = v3.eligible_records(records, 'A', profile)
            train = [v3.make_episode(pool, profile, v3.STRATA[i % len(v3.STRATA)], seed+i, calibration) for i in range(train_episodes)]
            evaluation = v3.evaluation_bank(records, 'C', profile, calibration, repeats)
            if not any(sum(dict(zip(e.units, e.utilities)).values()) > 0 for e in evaluation):
                raise ValueError(f'{task}: zero useful evaluation episodes')
            medians = v3.fit_medians(train)
            candidates = [c for e in train for c in e.candidates]
            units = [u for e in train for u in e.units]
            counts = Counter(units)
            weights = np.array([1/counts[u] for u in units])
            target = np.array([u/3 for e in train for u in e.utilities])
            feature_reports[task], correlation_reports[task] = feature_audit(evaluation)
            feature_reports[task]['unique_C_original_records'] = len(set(u for e in evaluation for u in e.units))
            collision_reports[task] = {'static': collisions(evaluation, (0, 2, 3, 5, 6)), 'full16': collisions(evaluation, tuple(range(8)))}
            collision_reports[task]['false_deferrals'] = false_deferrals(evaluation)
            novelty_reports[task] = novelty_diagnostics(evaluation, seed)
            role_support, prevalence = label_support(records, profile)
            tasks[task] = dict(A_unique_units=len(counts), C_pools=len(set(e.pool_id for e in evaluation)),
                               A_medians=medians.tolist(), calibration=v3.calibration_audit(records, profile, calibration),
                               label_role_support=role_support, label_prevalence=prevalence)
            task_rows = []
            eval_counts = Counter(u for e in evaluation for u in e.units)
            def evaluate(model_name, variant, predictor=None):
                for e in evaluation:
                    pred = predictor(e) if predictor else None
                    importance = [c.values[3] if c.available[3] else .5 for c in e.candidates]
                    if model_name == 'FIFO': order = sorted(range(len(e.candidates)), key=lambda i: tie_key(e.candidates[i]))
                    elif model_name == 'signature': order = signature_order(e.candidates)
                    elif model_name == 'oracle-greedy':
                        order, seen, remaining = [], set(), set(range(len(e.candidates)))
                        while remaining:
                            i = min(remaining, key=lambda i: (-(e.utilities[i] if e.units[i] not in seen else 0), tie_key(e.candidates[i])))
                            order.append(i); seen.add(e.units[i]); remaining.remove(i)
                    else: order = order_scores(e.candidates, pred if pred is not None else importance)
                    truth = np.array(e.utilities)/3
                    ew = np.array([1/eval_counts[u] for u in e.units])
                    clipped = np.clip(pred, 0, 1) if pred is not None else None
                    task_rows.append(dict(task=task, pool=e.pool_id, stratum=e.stratum, episode=e.episode_id,
                                          model=model_name, variant=variant, A10=v3.early_coverage(order, e.units, e.utilities),
                                          utility_MAE=float(np.average(abs(clipped-truth), weights=ew)) if clipped is not None else None,
                                          utility_MSE=float(np.average((clipped-truth)**2, weights=ew)) if clipped is not None else None))
            evaluate('I', 'direct'); evaluate('FIFO', 'direct'); evaluate('signature', 'direct')
            evaluate('oracle-utility', 'privileged', lambda e: np.array(e.utilities)/3)
            evaluate('oracle-greedy', 'privileged')
            for family in ('Ridge', 'HistGradientBoosting'):
                fitted = {}
                for name, fields in variants().items():
                    progress(f'{task}: {family}/{name} fit and evaluation')
                    x = design(candidates, medians, fields, name == 'masks-only')
                    key = (x.shape, hashlib.sha256(x.tobytes()).hexdigest())
                    if key not in fitted:
                        scaler = StandardScaler().fit(x, sample_weight=weights) if family == 'Ridge' else None
                        model = Ridge(alpha=1.) if family == 'Ridge' else HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, l2_regularization=1., random_state=seed, early_stopping=False)
                        model.fit(scaler.transform(x) if scaler else x, target, sample_weight=weights)
                        fitted[key] = model, scaler
                    model, scaler = fitted[key]
                    def predict(e, model=model, scaler=scaler, fields=fields, name=name):
                        z = design(e.candidates, medians, fields, name == 'masks-only')
                        return model.predict(scaler.transform(z) if scaler else z)
                    evaluate(family, name, predict)
                progress(f'{task}: {family} probes complete')
            grouped = defaultdict(list)
            for row in task_rows: grouped[(row['model'], row['variant'])].append(row)
            for (model, variant), rows in grouped.items():
                references = [('I', 'direct')]
                if model in ('Ridge', 'HistGradientBoosting'):
                    references.append((model, 'I'))
                for reference in dict.fromkeys(references):
                    comparisons.append(dict(task=task, model=model, variant=variant, reference='/'.join(reference),
                                            **comparison(rows, grouped[reference], bootstrap, seed)))
                if variant.startswith('all-minus'):
                    comparisons.append(dict(task=task, model=model, variant='all', reference=f'{model}/{variant}',
                                            **comparison(grouped[(model, 'all')], rows, bootstrap, seed)))
            tasks[task]['metrics'] = {f'{model}/{variant}': {
                metric: float(np.mean(list(pool_means(rows, metric).values()))) if pool_means(rows, metric) else None
                for metric in ('A10', 'utility_MAE', 'utility_MSE')}
                for (model, variant), rows in grouped.items()}
            tasks[task]['per_stratum_A10'] = {f'{model}/{variant}': {
                stratum: float(np.mean(list(pool_means([r for r in rows if r['stratum'] == stratum]).values())))
                if pool_means([r for r in rows if r['stratum'] == stratum]) else None for stratum in v3.STRATA}
                for (model, variant), rows in grouped.items()}
            all_rows.extend(task_rows)
            write_csv(output/'per_episode.csv', all_rows)
            write_csv(output/'model_comparisons.csv', comparisons)
        for name, data in [('feature_stats', feature_reports), ('correlation', correlation_reports),
                           ('header_collisions', collision_reports), ('novelty_diagnostics', novelty_reports)]:
            v3.write_json(output/f'{name}.json', data)
        summary = dict(status='complete', tasks=tasks, extraction_quality=quality, synthetic_counterexample=counterexample(),
                       elapsed_seconds=time.perf_counter()-started,
                       undefined_episodes=len({(r['task'], r['episode']) for r in all_rows if r['A10'] is None}),
                       interpretation='C is historically exposed diagnostic evaluation, not a new final holdout. Exploratory paired pool bootstrap, no multiplicity correction.')
        v3.write_json(output/'summary.json', summary)
        (output/'REPORT.md').write_text(report(summary, comparisons, provenance), encoding='utf-8')
        progress('complete')
        return summary
    except Exception as error:
        v3.write_json(output/'failure.json', dict(type=type(error).__name__, message=str(error), elapsed_seconds=time.perf_counter()-started))
        raise


def report(summary, comparisons, provenance):
    lines = ['# V3 parameter diagnostic audit', '', summary['interpretation'], '',
             'A alone fits medians, scalers and fixed utility regressors; B alone fits original Q calibration. C evaluates all declared probes. Test records are excluded before calibration and episode construction.',
             'Each selected field contributes value and validity separately. Missing values use A medians. Ridge alpha=1; histogram boosting uses 80 iterations, seven leaves, L2=1, no early stopping. Repeated A image rows have inverse occurrence weights. Ranking uses raw predictions and original tie keys; regression errors clip to [0,1].',
             'A10 is original unique-image utility coverage averaged over first ten decodes. Undefined zero-utility episodes remain missing. Repetitions average within strata and strata within disjoint C source-group pools; bootstrap resamples those pools. Small pool counts limit confidence intervals.',
             'R/S/Q/I/F/P/C/N are existing transmitter fields in the unchanged 32-byte header. F cannot be assessed when absent. P and C are scene proxies without independent pixel annotation truth. Correlation alone does not justify dropping or keeping a field. Masks-only is a missingness diagnostic, not semantic information.',
             'Static signatures use profile, validity and quantized R/Q/I/P/C only; full16 collision statistics additionally include S/F/N. Same-unit repeated copies are distinguished from different-original collisions. Signature deferral retains every candidate. Oracle orders use evaluator-only original image identity/native utility and are privileged controlled-observation references, not deployed policies or physical-fact truth.',
             'Novelty compares unchanged transmitter N to receiver-new identity after random completed prefixes. This diagnoses controlled observation mismatch, not real-world fact novelty. The balanced synthetic example is an information bound only.', '',
             '| Task | Model/variant | Reference | Paired ΔA10 | Pool bootstrap 95% interval | Pools |',
             '|---|---|---|---:|---|---:|']
    for row in comparisons:
        lines.append(f"| {row['task']} | {row['model']}/{row['variant']} | {row['reference']} | {row['delta_A10']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['pools']} |")
    lines += ['', 'No best-model selection, tuning, novelty claim, or journal-validity claim follows from this diagnostic.',
              '', 'Exact config/environment/hashes: `provenance.json`. Episode evidence: `per_episode.csv`; paired comparisons: `model_comparisons.csv`; descriptive artifacts: `feature_stats.json`, `correlation.json`, `header_collisions.json`, `novelty_diagnostics.json`. Progress and failures persist incrementally.',
              '', '```json', json.dumps(provenance['config'], indent=2), '```']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261024); parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--train-episodes', type=int, default=64); parser.add_argument('--bootstrap', type=int, default=1000)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
