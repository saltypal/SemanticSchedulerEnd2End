"""Replay supplied parameter records; no SPI classifier or image inference."""
import argparse
import json
from pathlib import Path
from Evaluation.stream_io import load_offered_stream
from Scheduler.scene_policy import ScenePolicy
from Scheduler.training.stream_pilot import SPIPolicy, evaluate, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--policy', choices=['fifo', 'rr', 'random', 'spi'], default='fifo')
    parser.add_argument('--observation-end', type=float, required=True)
    parser.add_argument('--delay-cap', type=float, default=10.)
    parser.add_argument('--failure-probability', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('new unique output directory required')
    args.output.mkdir(parents=True)
    save(args.output/'status.json', {'status':'running', 'stage':'load_external_records'})
    print('Loading external parameters, with no classifier import', flush=True)
    try:
        episode, manifest = load_offered_stream(args.runtime, args.references, seed=args.seed,
            failure_probability=args.failure_probability, delay_cap=args.delay_cap,
            observation_end=args.observation_end)
        policy = (ScenePolicy(json.loads(args.model.read_text())) if args.model else
                  SPIPolicy() if args.policy == 'spi' else ScenePolicy(kind=args.policy, seed=args.seed))
        save(args.output/'manifest.json', manifest)
        metrics = evaluate(policy, episode, manifest, args.output/'logs', 'external')
        save(args.output/'metrics.json', metrics)
        save(args.output/'status.json', {'status':'complete', 'scope':episode.scope,
            'has_references':bool(args.references), 'service':'synthetic; no codec or SPI inference'})
        print('Complete: '+str(args.output.resolve()), flush=True)
    except Exception as exc:
        save(args.output/'status.json', {'status':'failed', 'error':repr(exc)})
        raise


if __name__ == '__main__': main()
