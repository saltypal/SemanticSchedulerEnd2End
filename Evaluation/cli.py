"""Lazy module dispatch. Scheduler-only execution needs no neural framework."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from Evaluation.configuration import MODULES, resolve_config
from Evaluation.runs import Run

def execute(config, run):
    module, action = config['module'], config['action']
    if module in ('Evaluation','Scheduler'):
        from Evaluation.scheduling.contracts import ResourceBudget
        from Evaluation.scheduling.simulation import simulate
        from Scheduler.baselines.fixtures import demonstration_jobs
        from Scheduler.baselines.policies import POLICIES, create_policy
        settings = config['scheduler']
        budget = ResourceBudget(settings['resource_units'], config['transport']['resource_unit'], settings['max_jobs_per_slot'])
        names = POLICIES if config['implementation'] == 'all' else (config['implementation'],)
        jobs = demonstration_jobs()
        results = [simulate(jobs, create_policy(name), budget, slot_ms=settings['slot_ms'], horizon_ms=settings['horizon_ms']) for name in names]
        from Evaluation.runs import write_json
        write_json(run.path/'jobs.json', [asdict(job) for job in jobs])
        write_json(run.path/'traces.json', results)
        if action in ('demo','evaluate'):
            from Scheduler.evaluation.report import plot_comparison
            plot_comparison(results, run.path/'policy_comparison.png')
        return {'validation':'synthetic scheduling only; no classifier or wireless quality claims',
                'policies':[{key:value for key,value in result.items() if key not in ('jobs','decisions')} for result in results]}
    if module == 'Codec':
        from Codec.runtime import execute as execute_codec
        return execute_codec(config, run)
    raise ValueError('Use python -m parameters.parameter_cli for image parameter extraction')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--module', choices=MODULES)
    parser.add_argument('--action')
    parser.add_argument('--implementation')
    parser.add_argument('--profile')
    parser.add_argument('--device')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path)
    arguments = vars(parser.parse_args(argv))
    config_path = arguments.pop('config')
    output = arguments.pop('output')
    if output is not None:
        arguments['paths'] = {'results': str(output.resolve())}
    try:
        config = resolve_config(config_path, arguments)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        parser.error(str(error))
    run = Run(config)
    try:
        result = execute(config, run)
        run.finish(result)
    except Exception as error:
        run.fail(error)
        raise
    print(json.dumps({'run_id':run.id,'output':str(run.path),'result':result}, indent=2, default=str))

    if result.get('failed',0):
        raise SystemExit(2)
