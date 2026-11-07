"""Validated configuration: defaults < profile < root < explicit CLI."""
from copy import deepcopy
import os
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('Codec', 'Scheduler', 'Evaluation')
ACTIONS = {
    'Codec': ('validate', 'availability', 'benchmark', 'smoke', 'train'),
    'Scheduler': ('validate', 'demo', 'evaluate'),
    'Evaluation': ('validate', 'demo', 'evaluate'),
}

def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else deepcopy(value)
    return result

def read_yaml(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a YAML mapping: {path}')
    return value

def resolve_config(path=None, overrides=None):
    path = Path(path or ROOT / 'config.yaml').resolve()
    root = read_yaml(path)
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    module = overrides.get('module', root.get('module', 'Evaluation'))
    if module not in MODULES:
        raise ValueError(f'Unknown module: {module}')
    defaults = read_yaml(ROOT / module / 'config/defaults.yaml')
    profiles = root.pop('profiles', {})
    profile = overrides.get('profile', root.get('profile', 'default'))
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f'Unknown configuration profile: {profile}')
    config = merge(merge(merge(defaults, profiles[profile]), root), overrides)
    config['module'] = module
    config['profile'] = profile
    unknown = set(config) - {'module','action','profile','implementation','seed','device','paths','transport','scheduler','benchmark','semantic','dependencies','training'}
    if unknown:
        raise ValueError(f'Unknown root configuration keys: {sorted(unknown)}')
    if config.get('action') not in ACTIONS[module]:
        raise ValueError(f'Unsupported action {config.get("action")!r} for {module}')
    if type(config.get('seed')) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if not isinstance(config.get('device'), str) or config['device'] not in ('auto','cpu','cuda','cuda:0','cuda:1'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:0, cuda:1; TPU backend is not implemented')
    for name in ('paths', 'transport', 'scheduler', 'benchmark', 'semantic', 'dependencies', 'training'):
        if name in config and not isinstance(config[name], dict):
            raise ValueError(f'{name} must be a mapping')
    paths = config.get('paths', {})
    if not paths.get('results'):
        raise ValueError('paths.results is required')
    for key, value in paths.items():
        if value is not None:
            expanded = os.path.expandvars(os.path.expanduser(str(value)))
            candidate = Path(expanded)
            paths[key] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    channel = config['transport'].get('name', 'awgn')
    if channel not in ('awgn', 'rayleigh_perfect_csi'):
        raise ValueError('Implemented transport names: awgn, rayleigh_perfect_csi; Rician is deferred')
    config['transport']['name'] = channel
    if module in ('Scheduler','Evaluation'):
        from Evaluation.scheduling.contracts import ResourceBudget
        from Scheduler.baselines.policies import POLICIES
        if config['implementation'] not in (*POLICIES, 'all'):
            raise ValueError('Unknown scheduler implementation')
        settings = config['scheduler']
        for key in ('slot_ms','horizon_ms'):
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'scheduler.{key} must be positive and finite')
        ResourceBudget(settings['resource_units'], config['transport']['resource_unit'], settings['max_jobs_per_slot'])
    if module == 'Codec':
        if config['implementation'] not in ('raw','jpeg','swinjscc','deepjscc','ntscc','external_worker'):
            raise ValueError('Unknown codec implementation')
        benchmark = config['benchmark']
        if type(benchmark['timing_repetitions']) is not int or benchmark['timing_repetitions'] < 1:
            raise ValueError('timing_repetitions must be positive')
        if type(benchmark['jpeg_quality']) is not int or not 1 <= benchmark['jpeg_quality'] <= 95:
            raise ValueError('jpeg_quality must be in [1,95]')
        if type(benchmark['compute_lpips']) is not bool:
            raise ValueError('compute_lpips must be boolean')
        if isinstance(benchmark['snr_db'], bool) or not isinstance(benchmark['snr_db'],(int,float)) or not math.isfinite(benchmark['snr_db']):
            raise ValueError('snr_db must be finite')
        if type(benchmark['rate']) is not int or benchmark['rate'] <= 0:
            raise ValueError('rate must be a positive integer')
    for dependency in config.get('dependencies', {}).values():
        if not isinstance(dependency, dict):
            raise ValueError('Each dependency must be a mapping')
        for key in ('python','source_root','checkpoint','experiment_root','team_root',
                    'deep_checkpoint','ntscc_checkpoint','cifar_root','calibration_cache',
                    'deep_source_root','ntscc_source_root'):
            if key in dependency:
                candidate = Path(os.path.expandvars(os.path.expanduser(str(dependency[key]))))
                dependency[key] = str((candidate if candidate.is_absolute() else path.parent/candidate).resolve())
    if module == 'Codec' and config['action'] == 'train':
        if config['implementation'] != 'swinjscc':
            raise ValueError('Portable training currently implements SwinJSCC only')
        for name in ('train','validation','upstream_swin'):
            if not paths.get(name) or not Path(paths[name]).is_dir():
                raise ValueError(f'paths.{name} is required and must exist before training')
        for name in ('base_epochs','sa_ra_epochs','batch_size_per_device','patience','crop_size'):
            value = config['training'][name]
            if type(value) is not int or value <= 0:
                raise ValueError(f'training.{name} must be a positive integer')
        if config['training']['crop_size'] % 128:
            raise ValueError('Swin training crop_size must be a multiple of 128')
    return config
