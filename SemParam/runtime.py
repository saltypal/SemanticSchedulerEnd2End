"""Image-only semantic extraction. Codec loading is never required here."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any
from INFRA.Artifacts import TaskContext
from SemParam.catalog import SEMANTIC_PARAMETER_ORDER, SemanticPipelineSettings
from SemParam.observation_builder import parse_semantic_timestamp

def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Stage 1B configuration requires mapping: {name}")
    return dict(value)


def task_context_from_mapping(values: dict[str, Any]) -> TaskContext:
    """Create an explicit task policy from a YAML-compatible mapping."""

    return TaskContext(
        task_id=str(values["task_id"]),
        task_text=str(values["task_text"]),
        concept_importance={str(name): float(weight) for name, weight in dict(values.get("concept_importance", {})).items()},
        deadline_at=parse_semantic_timestamp(values.get("deadline_at")),
        urgency_horizon_seconds=float(values.get("urgency_horizon_seconds", 60.0)),
        context_text=str(values["context_text"]) if values.get("context_text") else None,
        metadata=dict(values.get("metadata", {})),
    )


def settings_from_config(config: dict[str, Any]) -> SemanticPipelineSettings:
    semantic = _as_mapping(config.get("semantic"), "semantic")
    enabled = tuple(str(name) for name in semantic.get("enabled_parameters", SEMANTIC_PARAMETER_ORDER))
    return SemanticPipelineSettings(
        concept_prompts={str(name): str(prompt) for name, prompt in dict(semantic["concept_prompts"]).items()},
        enabled_parameters=enabled,
        freshness_tau_seconds=float(semantic.get("freshness_tau_seconds", 10.0)),
        clip_temperature=float(semantic.get("clip_temperature", 0.07)),
        version=str(semantic.get("version", "stage1b-transformer/v1")),
        metadata={"config_stage": config.get("stage")},
    )



def run_semantic_demo(*args, **kwargs):
    """Compatibility entry point for the combined historical showcase."""
    from INFRA.experiments.semantic_showcase import run_semantic_demo as run
    return run(*args, **kwargs)

def load_semantic_image_artifact(path):
    from INFRA.experiments.semantic_showcase import load_semantic_image_artifact as load
    return load(path)

def execute(config, run):
    if config['action'] == 'train':
        from SemParam.training import train
        return train(Path(config['paths']['semantic_training']),Path(config['paths']['model_dir']),config['device'],
                     run.path,config['semantic']['resolved_settings'],Path(config['paths']['upstream_swin']))
    legacy = config['semantic']['resolved_settings']
    settings = settings_from_config(legacy)
    if config['action'] == 'validate':
        from SemParam.semantic_header import SemanticHeaderCodec
        return {'status':'semantic_contract_validated', 'parameters':list(settings.enabled_parameters),
                'codec_checkpoint_required':False, 'extraction_accuracy':'not evaluated'}
    from INFRA.runs import seed_models, select_device
    from SemParam.clip_backbone import CLIPBackbone
    from SemParam.KnowledgeBases import NetworkXKnowledgeBase
    from SemParam.pipeline import TransformerParameterPipeline
    from SemParam.semantic_header import SemanticHeaderCodec
    seed_models(config['seed'])
    device = select_device(config['device'])
    clip_settings = legacy['semantic']['clip']
    pipeline = TransformerParameterPipeline(settings, CLIPBackbone(
        model_name=clip_settings['model_name'], device=device,
        local_files_only=config['semantic']['local_files_only']), NetworkXKnowledgeBase())
    task = task_context_from_mapping(legacy['demo']['task'])
    now = parse_semantic_timestamp(legacy['demo']['reference_now'])
    rows = []
    header = SemanticHeaderCodec(settings.catalog)
    for index, entry in enumerate(legacy['demo']['images'], 1):
        path = Path(config['paths']['images']) / entry['filename']
        deadline = parse_semantic_timestamp(entry.get('deadline_at'))
        started = perf_counter_ns()
        bundle = pipeline.extract_from_path(path, replace(task, deadline_at=deadline or task.deadline_at), now,
            captured_at=entry.get('captured_at'), metadata={'demo_manifest':True})
        elapsed = (perf_counter_ns()-started)/1e6
        rows.append({'image_id':'image-' + str(bundle.metadata['image_hash'])[:16], 'source_path':str(path),
                     'extraction_time_ms':elapsed, 'bundle':bundle.to_dict()})
    from SemParam.evaluation.coverage import evaluate
    return {'status':'image_extraction_complete', 'device':device, 'images':rows,
            'model':clip_settings['model_name'], 'model_revision':getattr(pipeline.observation_builder.backend._model.config,'_commit_hash',None), 'codec_checkpoint_required':False,
            'evaluation': evaluate([row['bundle'] for row in rows]),
            'note':'Demonstration evidence, not classifier accuracy or validated urgency labels.'}
