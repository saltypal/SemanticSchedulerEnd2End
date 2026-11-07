"""Post-run integrity checks and prespecified supplementary analyses.

This module never fits/selects a scheduler. It reproduces saved predictions,
evaluates privileged references, and reports fixed-model utility sensitivity.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import time

import numpy as np

from Scheduler.training.decoder_v3_study import (Episode, PROFILES, read_jsonl, write_json,
    load_cache, calibrate, eligible_records, early_coverage, episode_order,
    pool_means, paired_bootstrap)
from Scheduler.decoder_scheduler import SchedulerCandidate, load_scheduler
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterHeader, ParameterExtractor, TransmitterMemory


def greedy_coverage_order(units,utilities):
    remaining,seen,order = set(range(len(units))),set(),[]
    while remaining:
        selected = max(remaining,key=lambda index:(utilities[index] if units[index] not in seen else 0.,-index))
        remaining.remove(selected); seen.add(units[selected]); order.append(selected)
    return order


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def artifact_path(run,stored):
    relative = PurePosixPath(stored).relative_to('/kaggle/working')
    destination = Path(run).joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(Path(run).resolve()):
        raise ValueError('artifact path escapes downloaded run')
    return destination


def aggregate_curves(rows):
    """Match primary A10 weighting: repetitions -> strata -> source pools."""
    from collections import defaultdict
    strata,pools = defaultdict(list),defaultdict(list)
    for row in rows:
        if row['curve'] is not None:
            strata[(row['pool_id'],row['stratum'])].append(row['curve'])
    for (pool,stratum),curves in strata.items():
        pools[pool].append(np.mean(curves,axis=0))
    return np.mean([np.mean(curves,axis=0) for curves in pools.values()],axis=0) if pools else None


def make_report_figures(run,output):
    """Report-only correction; never select plotted competitors on test scores."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run,output = Path(run),Path(output)
    tasks = [profile.task for profile in PROFILES.values()]
    palette = dict(fifo='#777777',importance='#e69f00',neural_coverage='#0072b2',
                   neural_ranknet='#009e73',coverage_tree='#cc79a7',regressor='#a65628')
    labels = dict(fifo='FIFO',importance='Importance only',neural_coverage='CAHR',
                  neural_ranknet='Ordinary neural RankNet')
    figure,axes = plt.subplots(1,3,figsize=(18,4))
    for axis,task in zip(axes,tasks):
        folder = run/'study'/task
        results = json.loads((folder/'evaluation/episode_metrics.json').read_text())
        comparator = json.loads((folder/'selection.json').read_text())['strongest_baseline']
        for name in dict.fromkeys(('fifo','importance','neural_coverage','neural_ranknet',comparator)):
            curve = aggregate_curves(results[name])
            if curve is not None:
                if not np.isclose(curve.mean(),np.mean(list(pool_means(results[name]).values())),atol=1e-12):
                    raise ValueError('plotted curve does not agree with primary pool-weighted A10')
                axis.plot(range(1,11),curve,label=labels.get(name,'C-selected: '+name),
                          color=palette.get(name),linewidth=1.7)
        axis.set(title=task.replace('_',' '),xlabel='Messages serviced',ylabel='Pool-weighted observation coverage',ylim=(0,1))
        axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output/'figure1_coverage.png',dpi=200); plt.close(figure)
    figure,axes = plt.subplots(1,3,figsize=(18,4))
    for axis,task in zip(axes,tasks):
        folder = run/'study'/task
        paired = json.loads((folder/'evaluation/paired_by_stratum.json').read_text())
        for index,(stratum,row) in enumerate(paired['comparisons'].items()):
            if row['ci95'] is None: continue
            low,high = row['ci95']; value = row['mean_difference']
            axis.errorbar(value,index,xerr=[[max(0,value-low)],[max(0,high-value)]],fmt='o',color='#0072b2')
        axis.set_yticks(range(5),list(paired['comparisons']),fontsize=8)
        axis.axvline(0,color='black',linewidth=.8)
        axis.set(title=task.replace('_',' '),xlabel='CAHR minus '+paired['baseline']+' (A10)')
    figure.tight_layout(); figure.savefig(output/'figure2_paired_comparison.png',dpi=200); plt.close(figure)
    write_json(output/'figure_provenance.json',dict(weighting='repetitions within strata, then strata within source pools; equals primary A10',
        curves='FIFO, importance-only, CAHR, ordinary neural RankNet, C-selected comparator; no test-based curve selection',
        model_or_prediction_changes=False,reason='align report curves with primary table weighting'))


def parameter_latency(cache,output):
    """CPU header math on cached evidence, not end-to-end GPU extraction time."""
    records = load_cache(cache)
    measurements = {}
    for profile in PROFILES.values():
        rows = eligible_records(records,'A',profile)
        calibration = calibrate(records,profile)
        memories = [TransmitterMemory(f'ue{index}',profile.profile_id) for index in range(4)]
        extractor = ParameterExtractor()
        seconds = []
        for index,row in enumerate(rows):
            memory = memories[index%4]
            start = time.perf_counter()
            bundle = extractor.extract(row['evidence'],profile,{},memory,calibration)
            wire = ParameterHeader(profile.profile_id,index,memory.version,0,
                                   bundle.values,bundle.available).to_bytes()
            ParameterHeader.from_bytes(wire)
            seconds.append(time.perf_counter()-start)
            concepts = {name for name in profile.eligible_concepts if row['evidence'].logits[name]>=0}
            memory.commit(dict(embedding=row['evidence'].embedding,concepts=concepts,image_id=row['image_id']))
        measurements[profile.task] = dict(images=len(rows),median_ms=1000*float(np.median(seconds)),
            p95_ms=1000*float(np.quantile(seconds,.95)),memory_capacity=64,
            scope='cached-evidence parameter math plus serialization/parsing; excludes model inference and memory commit')
    write_json(Path(output)/'parameter_cpu_latency.json',dict(measurements=measurements,
        platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
        warning='Different host from Kaggle GPU profiles; do not add these timings into an end-to-end latency claim.'))


def analyze_run(run,cache,output):
    run,cache,output = Path(run),Path(cache),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    status = json.loads((run/'study/status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('completed study required')
    data_audit = json.loads((run/'study/data_audit.json').read_text())
    for name,digest in data_audit['cache_hashes'].items():
        if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('analysis cache differs from executed training input')
    verified,oracles,sensitivity,auxiliary,availability,main_comparisons = {},{},{},{},{},{}
    with gzip.open(output/'verified_scores.jsonl.gz','wt',encoding='utf-8') as score_log:
        for profile in PROFILES.values():
            folder = run/'study'/profile.task
            selection = json.loads((folder/'selection.json').read_text())
            results = json.loads((folder/'evaluation/episode_metrics.json').read_text())
            runtime = read_gzip(folder/'evaluation/runtime_episodes.jsonl.gz')
            truths = {row['episode_id']:row for row in read_gzip(folder/'evaluation/evaluator_episodes.jsonl.gz')}
            episodes = []
            for row in runtime:
                truth = truths[row['episode_id']]
                candidates = tuple(SchedulerCandidate(**candidate) for candidate in row['candidates'])
                metadata = results['neural_coverage'][len(episodes)]
                if metadata['episode_id']!=row['episode_id']:
                    raise ValueError('saved evaluation episode alignment differs')
                for candidate,wire in zip(candidates,row['header_hex']):
                    header = ParameterHeader.from_bytes(bytes.fromhex(wire))
                    if header.values!=candidate.values or header.available!=candidate.available or header.profile_id!=candidate.profile_id:
                        raise ValueError('runtime candidate differs from serialized header')
                episodes.append(Episode(row['episode_id'],metadata['pool_id'],profile.task,metadata['stratum'],
                    candidates,tuple(truth['units']),tuple(truth['utilities']),tuple(row['header_hex'])))
            if len(results)!=13 or any(len(rows)!=len(episodes) for rows in results.values()):
                raise ValueError('incomplete matched-method comparison')
            checked = 0
            for name,configuration in selection['methods'].items():
                model = load_scheduler(artifact_path(run,configuration['artifact']),{profile.profile_id:profile})
                for episode,stored in zip(episodes,results[name]):
                    order = episode_order(model,episode)
                    if order!=stored['order']:
                        raise ValueError(f'exported-model prediction mismatch: {profile.task}/{name}/{episode.episode_id}')
                    scores = model.scores(episode.candidates)
                    if not np.isfinite(scores).all():
                        raise ValueError('nonfinite exported-model score')
                    score_log.write(json.dumps(dict(task=profile.task,method=name,episode_id=episode.episode_id,
                        message_ids=[candidate.message_id for candidate in episode.candidates],scores=scores.tolist(),order=order))+'\n')
                    checked += 1
            verified[profile.task] = dict(episode_method_predictions_reproduced=checked,
                methods=len(results),episodes=len(episodes),selected_on_C=selection['selected'],
                baseline_selected_on_C=selection['strongest_baseline'])
            main_comparisons[profile.task] = {name:paired_bootstrap(pool_means(results['neural_coverage']),pool_means(rows))
                                              for name,rows in results.items() if name!='neural_coverage'}
            flags = np.asarray([candidate.available for episode in episodes for candidate in episode.candidates],float)
            availability[profile.task] = dict(zip(('R','S','Q','I','F','P','C','N'),flags.mean(axis=0).tolist()))
            oracle_rows = {'true_item_utility':[],'greedy_marginal_coverage':[]}
            for episode in episodes:
                orders = dict(true_item_utility=sorted(range(len(episode.units)),key=lambda i:(-episode.utilities[i],i)),
                              greedy_marginal_coverage=greedy_coverage_order(episode.units,episode.utilities))
                for name,order in orders.items():
                    oracle_rows[name].append(dict(pool_id=episode.pool_id,stratum=episode.stratum,
                        A10=early_coverage(order,episode.units,episode.utilities)))
            oracles[profile.task] = {name:dict(pools=len(pool_means(rows)),
                A10=float(np.mean(list(pool_means(rows).values())))) for name,rows in oracle_rows.items()}
            sensitivity[profile.task] = {}
            for higher in (2.,3.,5.):
                by_method = {}
                for name,rows in results.items():
                    adjusted = [dict(pool_id=q.pool_id,stratum=q.stratum,
                        A10=early_coverage(stored['order'],q.units,tuple(higher if w==3 else w for w in q.utilities)))
                        for q,stored in zip(episodes,rows)]
                    means = pool_means(adjusted)
                    by_method[name] = float(np.mean(list(means.values()))) if means else None
                sensitivity[profile.task][str(higher)] = by_method
            if profile.task=='building_damage':
                auxiliary['ablations'] = {}
                for path in sorted((folder/'ablations').glob('*/test.json')):
                    rows = json.loads(path.read_text())
                    auxiliary['ablations'][path.parent.name] = paired_bootstrap(pool_means(rows),pool_means(results['neural_coverage']))
                if len(auxiliary['ablations'])!=9:
                    raise ValueError('expected nine retrained ablations')
                auxiliary['seeds'] = {}
                for seed in (1,2):
                    main = json.loads((folder/'seeds/neural_coverage'/str(seed)/'test.json').read_text())
                    baseline = json.loads((folder/'seeds'/selection['strongest_baseline']/str(seed)/'test.json').read_text())
                    auxiliary['seeds'][str(seed)] = paired_bootstrap(pool_means(main),pool_means(baseline))
            for rows in read_gzip(folder/'evaluation/streaming.jsonl.gz'):
                events = rows['events']
                if len(events)!=len({event['message_id'] for event in events}) or len(events)!=32:
                    raise ValueError('stream replay did not service each offered message exactly once')
                if any(event['start']<event['arrival'] or event['wait']<0 for event in events):
                    raise ValueError('stream selected a future message or negative waiting time')
                if any(right['start']<left['completion'] for left,right in zip(events,events[1:])):
                    raise ValueError('overlapping service in single-worker stream')
                if rows['scenario']=='no_backlog' and any(event['wait']!=0 for event in events):
                    raise ValueError('no-backlog control introduced waiting')
    rows = list(csv.DictReader((run/'study/comparison.csv').open()))
    macro = []
    for name in sorted({row['method'] for row in rows}):
        subset = [row for row in rows if row['method']==name]
        macro.append(dict(method=name,**{metric:float(np.mean([float(row[metric]) for row in subset]))
            for metric in ('A10','coverage5','coverage10','ndcg10','high_recall10','redundancy10')}))
    write_json(output/'audit.json',dict(status='passed',predictions=verified,
        input_run=str(run.resolve()),input_cache=str(cache.resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='post-run integrity and prespecified fixed-model analyses; no retraining or test-based model selection'))
    write_json(output/'oracle_diagnostics.json',dict(scope='privileged, not deployable',tasks=oracles))
    write_json(output/'utility_sensitivity.json',dict(scope='fixed models and unchanged headers; reference high-grade utility changed, not retraining',tasks=sensitivity))
    write_json(output/'building_auxiliary_comparisons.json',auxiliary)
    write_json(output/'header_availability.json',availability)
    write_json(output/'main_vs_all_baselines.json',main_comparisons)
    with (output/'macro_comparison.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(macro[0])); writer.writeheader(); writer.writerows(macro)
    parameter_latency(cache,output)
    make_report_figures(run,output)
    return verified


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--cache',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(analyze_run(args.run,args.cache,args.output))
