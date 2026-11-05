"""Observable local fixture pilot; reuses the canonical eleven learner arms."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from Evaluation.stream_simulator import (StreamConfig, StreamEnvironment, build_stream,
                                         priority_metrics, REGISTRY_HASH, FIELDS, PROFILES)
from Scheduler.scene_policy import ScenePolicy, canonical, digest
from Scheduler.decoder_scheduler import Decision
from Scheduler.training.scene_learners import (ARMS, Example, fit_supervised, train_dqn, train_ppo)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(canonical(value) + '\n')


class SPIPolicy:
    def reset(self): pass
    def select(self, context):
        if not context.ready_candidates: return None
        def key(c):
            priority = int(np.argmax(c.values[:4])) if any(c.available[:4]) else 4
            return priority, c.arrival_time, c.sequence, c.message_id
        c = min(context.ready_candidates, key=key)
        return Decision(c.message_id, 'spi_fifo_unknown_last', 0.)


def collect(episodes, role='train'):
    examples = []
    for index, episode in enumerate(episodes):
        env = StreamEnvironment(episode)
        policy = ScenePolicy(kind='random', seed=index)
        while not env.finished:
            context, gains = env.observation(), env.marginal_gains()
            pairs = tuple((i, j, 1.) for i in range(len(gains)) for j in range(len(gains)) if gains[i] > gains[j])
            examples.append(Example(context, gains, f'{role}-{index}', 'synthetic_world_reference', pairs, 'seeded_random'))
            env.step(policy.select(context).message_id)
    return examples


def evaluate(policy, episode, manifest, output=None, name='policy'):
    env = StreamEnvironment(episode)
    policy.reset()
    latencies, probes = [], []
    while not env.finished:
        start = time.perf_counter_ns()
        context = env.observation()
        decision = policy.select(context)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if decision is None: raise RuntimeError('policy returned no action with ready jobs')
        transition = env.step(decision.message_id)
        latencies.append(elapsed_ms)
        if output:
            append(output / f'{name}.jsonl', {'observation': asdict(context),
                'decision': asdict(decision), 'selection_ms': elapsed_ms,
                'outcome': env.events[-1], 'training_reward': transition.reward,
                'next_ready_ids': [c.message_id for c in transition.next_observation.ready_candidates]})
    metrics = env.metrics()
    metrics['priority'] = priority_metrics(episode, manifest, env.events, manifest['config']['delay_cap'])
    selected = {e['message_id'] for e in env.events}
    metrics['accounting'] = {'offered': len(episode.messages), 'selected': len(selected),
        'succeeded': sum(e['success'] for e in env.events),
        'failed': sum(not e['success'] and not e['unfinished'] for e in env.events),
        'in_flight_at_horizon': sum(e['unfinished'] for e in env.events),
        'queued_at_horizon': len(env.controller), 'not_arrived': len(env.pending)}
    a = metrics['accounting']
    assert a['offered'] == a['succeeded'] + a['failed'] + a['in_flight_at_horizon'] + a['queued_at_horizon'] + a['not_arrived']
    if metrics['J_time'] is not None:
        assert np.isclose(metrics['reward_sum'], metrics['J_time'])
    else:
        assert metrics['offered_weight'] == 0 and metrics['reward_sum'] == 0
    metrics['selection_ms'] = {'p50': float(np.median(latencies)), 'p95': float(np.quantile(latencies, .95)),
        'n': len(latencies), 'scope': 'CPU context build + selection; unbatched; overhead not charged to synthetic service time'}
    metrics['fallback_count'] = getattr(policy, 'fallback_count', 0)
    metrics['events_sha256'] = digest(env.events)
    metrics['input_stream_sha256'] = manifest['sha256']
    return metrics


def choice_probes(policy, examples):
    # Identical held-out snapshots/history for every policy; not its own rollouts.
    results = []
    for e in examples:
        if max(e.gains, default=0) <= 0: continue
        if getattr(policy, 'artifact', None):
            scores = policy.scores(e.context)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], e.context.ready_candidates[i].arrival_time))
        else:
            policy.reset()
            chosen = policy.select(e.context).message_id
            first = next(i for i, c in enumerate(e.context.ready_candidates) if c.message_id == chosen)
            # Non-scoring references expose only a next choice, not an invented full order.
            order = [first]
        row = {}
        for k in (1, 3, 5):
            if len(order) < min(k, len(e.gains)): continue
            gain = lambda ids: sum(e.gains[i] / np.log2(r + 2) for r, i in enumerate(ids[:k]))
            ideal = gain(sorted(range(len(e.gains)), key=lambda i: -e.gains[i]))
            row[f'ndcg@{k}'] = float(gain(order) / ideal)
        row['best_choice'] = float(e.gains[order[0]] == max(e.gains))
        results.append(row)
    keys = {k for r in results for k in r}
    return {k: {'mean': float(np.mean([r[k] for r in results if k in r])),
                 'defined': sum(k in r for r in results)} for k in sorted(keys)}


def compute_profile(artifact, candidates=1):
    model = artifact['model']
    if model['kind'] != 'dense':
        return {'parameters': None, 'forward_flops': None, 'scope': 'tree traversal; dense FLOPs not applicable'}
    parameters = sum(np.size(l['weight']) + len(l['bias']) for l in model['layers'])
    mac = sum(np.size(l['weight']) for l in model['layers']) * candidates
    return {'parameters': int(parameters), 'forward_flops': int(2 * mac), 'macs': int(mac),
            'candidates': candidates, 'scope': 'dense matrix products only; 1 MAC=2 FLOPs; excludes preprocessing/bias/ReLU and runtime overhead'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--rl-steps', type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.episodes <= 3 or args.epochs < 1 or args.rl_steps < 16:
        parser.error('pilot requires 2..3 episodes, positive epochs and >=16 RL decisions')
    out = args.output.resolve()
    if out.exists(): raise ValueError('unique new run directory required')
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(out / 'run.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    import torch
    torch.set_num_threads(1)
    source_root = Path(__file__).resolve().parents[2]
    import importlib.metadata as metadata
    files = ['Evaluation/stream_simulator.py', 'Evaluation/stream_io.py', 'Evaluation/scene_environment.py',
             'Evaluation/decoder_service.py', 'Evaluation/spi_evaluation.py', 'Evaluation/decoder_queue.py',
             'Scheduler/scene_policy.py', 'Scheduler/training/stream_pilot.py', 'Scheduler/training/scene_learners.py']
    identity = {'scope': 'synthetic plumbing/training pilot; NOT real image or VLM evidence',
        'interpreter': sys.executable, 'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {k: metadata.version(k) for k in ('numpy', 'torch', 'lightgbm')},
        'device': 'cpu', 'threads': 1, 'config': vars(args) | {'output': str(out)},
        'source_sha256': {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in files},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(),
        'dirty_state': subprocess.check_output(['git', 'status', '--short'], cwd=source_root, text=True),
        'registry': {'sha256': REGISTRY_HASH, 'fields': FIELDS},
        'spi_classifier': 'excluded: synthetic supplied class or explicit unavailable',
        'teacher': 'not loaded; three real teacher arms remain blocked pending qualification'}
    save(out / 'manifest.json', identity)
    for filename in files:
        dest = out / 'source' / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source_root / filename).read_bytes())
    save(out / 'profiles.json', [dict(id=p[0], device_description=p[1], mission=p[2]) for p in PROFILES])
    save(out / 'status.json', {'stage': 'generating', 'status': 'running'})
    logging.info('Run started at %s; CPU fixture only', out)
    # Three held-in worlds for training; evaluation worlds have different IDs/seeds.
    training_data = [build_stream(StreamConfig(seed=110+i, max_ues=4, messages=28, capture_window=7.)) for i in range(3)]
    train = [e for e, _ in training_data]
    for i, (_, manifest) in enumerate(training_data): save(out / f'data/training_{i}.json', manifest)
    examples = collect(train)
    validation, validation_manifest = build_stream(StreamConfig(seed=910, max_ues=12, messages=24, capture_window=6.))
    save(out / 'data/validation.json', validation_manifest)
    probes = collect([validation], role='validation')
    save(out / 'data/training_examples.json', [asdict(e) for e in examples])
    save(out / 'data/validation_examples.json', [asdict(e) for e in probes])
    evaluations = []
    for i, population in enumerate((4, 60, 100)[:args.episodes]):
        episode, manifest = build_stream(StreamConfig(seed=510+i, max_ues=population, messages=32+i*16,
            capture_window=8., failure_probability=.1, service_log_sigma=.25, correlated_outage=i == 2))
        save(out / f'data/evaluation_{i}.json', manifest)
        evaluations.append((episode, manifest))
    evaluation_probes = collect([e for e, _ in evaluations], role='pilot_evaluation')
    save(out / 'data/pilot_evaluation_examples.json', [asdict(e) for e in evaluation_probes])
    reports, policies = {}, {'fifo': ScenePolicy(), 'rr': ScenePolicy(kind='rr'),
        'random': ScenePolicy(kind='random', seed=7), 'spi': SPIPolicy(),
        'relevance': ScenePolicy(kind='single', field=4), 'novelty': ScenePolicy(kind='single', field=5)}
    validation_score = lambda policy: evaluate(policy, validation, validation_manifest)['J_time']
    for arm in ARMS:
        logging.info('Starting learner %s', arm)
        save(out / 'status.json', {'stage': arm, 'status': 'running'})
        start = time.monotonic()
        if arm in ('rl_vlm_f', 'vlm_replay', 'visionprefer'):
            reports[arm] = {'status': 'blocked', 'reason': 'No qualified real VLM labels; fixture labels are not impersonated as Qwen feedback'}
            save(out / 'models' / arm / 'status.json', reports[arm])
            continue
        def progress(state, history, name=arm):
            dest = out / 'models' / name
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(state, dest / 'resume.pt')
            save(dest / 'learning.json', history)
        try:
            if arm == 'masked_ppo':
                model, history, state = train_ppo(lambda i: StreamEnvironment(train[i % len(train)]), examples,
                    steps=args.rl_steps, rollout_steps=min(32, args.rl_steps), save_progress=progress,
                    validation_score=validation_score, eval_interval=32, experiment_id=digest(identity['source_sha256']))
            elif arm in ('masked_double_dqn', 'td_per'):
                model, history, state = train_dqn(lambda i: StreamEnvironment(train[i % len(train)]), examples,
                    algorithm=arm, steps=args.rl_steps, save_progress=progress,
                    validation_score=validation_score, eval_interval=32, experiment_id=digest(identity['source_sha256']))
            else:
                model, history, state = fit_supervised(examples, arm, epochs=args.epochs, fixture=True,
                    save_progress=progress, validation_score=validation_score)
            if model is None: raise RuntimeError('training returned no artifact')
            progress(state, history)
            save(out / 'models' / arm / 'portable.json', model)
            policies[arm] = ScenePolicy(json.loads((out / 'models' / arm / 'portable.json').read_text()))
            reports[arm] = {'status': 'fixture_trained', 'seconds': time.monotonic()-start,
                'learning_records': len(history), 'compute': compute_profile(model),
                'checkpoint': f'models/{arm}/resume.pt', 'model': f'models/{arm}/portable.json'}
        except Exception as exc:
            logging.exception('Learner failed: %s', arm)
            reports[arm] = {'status': 'failed', 'error': repr(exc), 'seconds': time.monotonic()-start}
        save(out / 'learner_status.json', reports)
    results = []
    for name, policy in policies.items():
        logging.info('Evaluating %s on %d held-out synthetic episodes', name, len(evaluations))
        for i, (episode, manifest) in enumerate(evaluations):
            metrics = evaluate(policy, episode, manifest, out / f'logs/episode_{i}', name)
            save(out / f'metrics/{name}_{i}.json', metrics)
            results.append({'policy': name, 'episode': i, 'max_ues': manifest['config']['max_ues'],
                'J_time': metrics['J_time'], 'critical': metrics['priority']['by_reference_class']['critical'],
                'accounting': metrics['accounting'], 'selection_ms': metrics['selection_ms']})
        save(out / f'metrics/{name}_validation_probes.json', choice_probes(policy, probes))
        save(out / f'metrics/{name}_shared_evaluation_probes.json', choice_probes(policy, evaluation_probes))
    save(out / 'learner_status.json', reports)
    save(out / 'comparison.json', results)
    lines = ['# Local synthetic stream pilot', '',
        'Synthetic parameters/world labels only. No SPI classifier, RGB inference, Qwen or codec was run.',
        'These short training updates test execution; they do not establish convergence or emergency importance.', '',
        '| Policy | Episode | Max UE pool | J_time | Critical capped delay |', '|---|---:|---:|---:|---:|']
    for row in results:
        lines.append(f"| {row['policy']} | {row['episode']} | {row['max_ues']} | {row['J_time']:.4f} | {row['critical']['mean_capped_completion_delay']} |")
    lines += ['', '## Learners', ''] + [f"- {name}: {r['status']}" for name, r in reports.items()]
    lines += ['', '## Limits', '', 'The profile/trajectory stream is a constructed fixture, not coherent real camera imagery.',
        'Timing uses synthetic service units; measured selection milliseconds are host CPU measurements only.',
        'Header bytes are compact JSON interchange bytes, not the current 35-byte wire encoding.',
        'No claimed FLOPs for teacher/extractor; dense matrix MAC estimates exclude preprocessing and activations.',
        'Real VLM-dependent arms are explicitly blocked until qualified feedback exists.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(out / 'status.json', {'status': 'complete_with_blocked_teacher_arms', 'episodes': len(evaluations),
         'learner_failures': [k for k,v in reports.items() if v['status']=='failed'], 'scope': identity['scope']})
    logging.info('Complete: %s', out / 'REPORT.md')


if __name__ == '__main__': main()
