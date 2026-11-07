"""Portable Base -> SA+RA training using reusable model modules."""
from dataclasses import asdict
from pathlib import Path
from Codec.Models.SwinJSCC.training_utils import detect_training_plan, transfer_compatible_weights
from Codec.Models.SwinJSCC.trainer import build_base_then_sara, train_phase, TrainingPhaseConfig
from Codec.Models.SwinJSCC.swin_config import SwinJSCCConfig
from Codec.Models.SwinJSCC.checkpoint_manager import export_stage1_artifacts
from Codec.Preprocessing.image_loader import dataset_provenance
from Evaluation.runs import seed_models

def train(config, run):
    settings = config['training']
    for key in ('train','validation'):
        if not config['paths'].get(key) or not Path(config['paths'][key]).is_dir():
            raise ValueError(f'paths.{key} must point to real training images')
    seed_models(config['seed'])
    plan = detect_training_plan(config['device'])
    architecture = SwinJSCCConfig()
    base, adaptive = build_base_then_sara(architecture,config['paths']['upstream_swin'],device=plan.device)
    results = {}
    for name, adapter in (('base',base),('sa_ra',adaptive)):
        if name == 'sa_ra':
            results['transfer'] = transfer_compatible_weights(base.build_training_module(),adaptive.build_training_module())
        phase = TrainingPhaseConfig(name=name,epochs=settings[name+'_epochs'],batch_size_per_gpu=settings['batch_size_per_device'],
            crop_size=settings['crop_size'],seed=config['seed'],early_stopping_patience=settings['patience'])
        result = train_phase(adapter,phase,config['paths']['train'],config['paths']['validation'],architecture.snr_db_grid,
            architecture.rate_grid,plan,run.path/'checkpoints',resume=False)
        results[name] = asdict(result)
    data = {key:dataset_provenance(config['paths'][key]) for key in ('train','validation')}
    export_stage1_artifacts(adaptive,run.path/'models/SwinJSCC',data,results,upstream_relative_path=config['paths']['upstream_swin'])
    return {'training':results,'hardware':asdict(plan),'data':data}
