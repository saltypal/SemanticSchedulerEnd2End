"""Generate thin, repository-backed Stage 1B/Stage 1C notebooks."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def _base_setup_cell() -> str:
    return """from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / 'pyproject.toml').is_file() and (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError('Run this notebook from the SemanticSchedulerEnd2End repository or a child directory.')


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
SOURCE_ROOT = PROJECT_ROOT / 'src'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unavailable'


RUN_CONTEXT = {
    'project_root': str(PROJECT_ROOT),
    'git_commit': git_commit(),
    'python': platform.python_version(),
}
print(json.dumps(RUN_CONTEXT, indent=2))
"""


def build_training_notebook() -> dict:
    md = lambda source: cell("markdown", source)
    py = lambda source: cell("code", source)
    cells = [
        md(
            """# Stage 1C prototype: local semantic-parameter codebook training

This notebook creates a **demo-only** semantic parameter codebook from separate local images. It uses CLIP only as an offline teacher. Runtime codebook inference consumes SwinJSCC semantic representations, never CLIP embeddings.

It is not a research-quality training split and must not use `Test/images`."""
        ),
        md("## 1. Setup and reproducibility"),
        py(_base_setup_cell()),
        md("## 2. Configuration and strict input separation"),
        py(
            """from utils.config_loader import load_yaml_config
from EncDecPipeline.Preprocessing.image_loader import discover_images, file_sha256

CONFIG_PATH = PROJECT_ROOT / 'config' / 'stage1c_parameter_codebook.yaml'
config = load_yaml_config(CONFIG_PATH)
codebook_config = config['parameter_codebook']
TRAIN_IMAGE_ROOT = PROJECT_ROOT / codebook_config['training_image_root']
TEST_IMAGE_ROOT = PROJECT_ROOT / config['demo']['image_root']
MODEL_DIR = PROJECT_ROOT / 'models' / 'SwinJSCC'
ARTIFACT_PATH = PROJECT_ROOT / codebook_config['artifact_path']

training_images = discover_images(TRAIN_IMAGE_ROOT)
test_images = discover_images(TEST_IMAGE_ROOT)
if len(training_images) < int(codebook_config['minimum_distinct_images']):
    raise RuntimeError(
        f\"Need at least {codebook_config['minimum_distinct_images']} distinct local training images; found {len(training_images)}.\"
    )
training_hashes = [file_sha256(path) for path in training_images]
test_hashes = {file_sha256(path) for path in test_images}
if len(set(training_hashes)) != len(training_hashes):
    raise RuntimeError('Training image SHA-256 hashes are not unique.')
overlap = sorted(set(training_hashes).intersection(test_hashes))
if overlap:
    raise RuntimeError(f'Train/test image leakage detected by SHA-256: {overlap}')
if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f'Missing manifest-validated Stage 1A model directory: {MODEL_DIR}')

RUN_CONTEXT.update(
    {
        'config': str(CONFIG_PATH),
        'training_image_root': str(TRAIN_IMAGE_ROOT),
        'training_image_count': len(training_images),
        'test_image_root': str(TEST_IMAGE_ROOT),
        'artifact_path': str(ARTIFACT_PATH),
        'demo_only': bool(codebook_config['demo_only']),
    }
)
print(json.dumps(RUN_CONTEXT, indent=2))
"""
        ),
        md("## 3. Load the offline Transformer teacher and SwinJSCC representation hook"),
        py(
            """from TransformerPipeline.KnowledgeBases import NetworkXKnowledgeBase
from TransformerPipeline.clip_backbone import CLIPBackbone
from TransformerPipeline.pipeline import TransformerParameterPipeline
from TransformerPipeline.runtime import load_semantic_image_artifact, settings_from_config, task_context_from_mapping
from TransformerPipeline.observation_builder import parse_semantic_timestamp
from EncDecPipeline.Models.SwinJSCC.model_loader import load_stage1_swinjscc
from CodebookParam.swin_representation import capture_pre_modulation_representation

settings = settings_from_config(config)
task = task_context_from_mapping(config['demo']['task'])
reference_now = parse_semantic_timestamp(config['demo']['reference_now'])
device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
clip_config = config['semantic']['clip']
teacher = TransformerParameterPipeline(
    settings=settings,
    embedding_backend=CLIPBackbone(
        model_name=clip_config['model_name'],
        device=device,
        local_files_only=bool(clip_config.get('local_files_only', False)),
    ),
    knowledge_base=NetworkXKnowledgeBase(),
)
adapter = load_stage1_swinjscc(MODEL_DIR, device=device)
encoder, decoder = adapter._require_built()
encoder.eval()
decoder.eval()
print({'device': device, 'teacher': teacher.name, 'task_profile_id': task.task_id})
"""
        ),
        md("## 4. Form offline Swin/CLIP teacher pairs"),
        py(
            """from CodebookParam.semantic_codebook import CONTENT_PARAMETER_NAMES

representations = []
target_rows = []
for image_path, image_hash in zip(training_images, training_hashes, strict=True):
    bundle = teacher.extract_from_path(
        image_path,
        task,
        reference_now,
        captured_at=reference_now,
        metadata={'parameter_codebook_training': True},
    )
    targets = {}
    for parameter_name in CONTENT_PARAMETER_NAMES:
        parameter = bundle.get(parameter_name)
        if not parameter.available or parameter.value is None:
            raise RuntimeError(f'Offline teacher did not produce {parameter_name} for {image_path}.')
        targets[parameter_name] = parameter.value
    image_artifact = load_semantic_image_artifact(image_path)
    capture = capture_pre_modulation_representation(
        adapter,
        image_artifact,
        snr_db=float(config['semantic']['snr_db']),
        rate=int(config['semantic']['rate']),
    )
    representations.append(capture.representation.detach().cpu().numpy())
    target_rows.append(targets)

print({'paired_images': len(representations), 'target_fields': CONTENT_PARAMETER_NAMES})
"""
        ),
        md("## 5. Train and save the deterministic demo-only codebook"),
        py(
            """from CodebookParam.semantic_codebook import SemanticParameterCodebook

semantic_codebook = SemanticParameterCodebook.train(
    representations,
    target_rows,
    task_profile_id=task.task_id,
    training_image_hashes=training_hashes,
    random_state=int(codebook_config['random_state']),
    codebook_size=int(codebook_config['codebook_size']),
    demo_only=bool(codebook_config['demo_only']),
    metadata={'run_context': RUN_CONTEXT},
)
saved_path = semantic_codebook.save(ARTIFACT_PATH)
summary_path = PROJECT_ROOT / 'results' / 'stage1c_parameter_codebook_demo_summary.json'
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps(
        {
            **RUN_CONTEXT,
            'artifact_content_hash': semantic_codebook.content_hash(),
            'artifact_path': str(saved_path),
            'target_fields': CONTENT_PARAMETER_NAMES,
            'warning': 'demo_only: ten local images are an engineering smoke corpus, not a research training result',
        },
        indent=2,
        sort_keys=True,
    ),
    encoding='utf-8',
)
print({'artifact': str(saved_path), 'summary': str(summary_path), 'content_hash': semantic_codebook.content_hash()})
"""
        ),
        md(
            """## 6. Interpretation

The saved artifact maps deterministic pooled Swin semantic features to three content-derived teacher targets: relevance, reliability, and task importance. AoI, freshness, urgency, and novelty remain runtime metadata/knowledge calculations. Do not report these ten-image results as model accuracy or generalization evidence."""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_showcase_notebook() -> dict:
    md = lambda source: cell("markdown", source)
    py = lambda source: cell("code", source)
    cells = [
        md(
            """# Stage 1B: Transformer versus semantic-parameter codebook

This notebook sends the four configured test images through a sequential knowledge base, renders the requested image/parameter/header/payload figure, then compares the Transformer and Swin-codebook semantic paths. The existing Stage 1A VQ packet remains separate from the semantic header."""
        ),
        md("## 1. Setup and explicit experiment inputs"),
        py(_base_setup_cell()),
        py(
            """from utils.config_loader import load_yaml_config
from TransformerPipeline.benchmark import artifact_size_bytes, benchmark_callable
from TransformerPipeline.runtime import run_semantic_demo

STAGE1B_CONFIG_PATH = PROJECT_ROOT / 'config' / 'stage1b.yaml'
STAGE1C_CONFIG_PATH = PROJECT_ROOT / 'config' / 'stage1c_parameter_codebook.yaml'
stage1b_config = load_yaml_config(STAGE1B_CONFIG_PATH)
stage1c_config = load_yaml_config(STAGE1C_CONFIG_PATH)
MODEL_DIR = PROJECT_ROOT / 'models' / 'SwinJSCC'
COMMUNICATION_VQ_PATH = PROJECT_ROOT / 'models' / 'Codebooks' / 'swinjscc_tx_k256.joblib'
PARAMETER_CODEBOOK_PATH = PROJECT_ROOT / stage1c_config['parameter_codebook']['artifact_path']
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'stage1b_semantic_showcase'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f'Missing manifest-validated Stage 1A model directory: {MODEL_DIR}')
if not PARAMETER_CODEBOOK_PATH.is_file():
    raise FileNotFoundError(
        f'Missing semantic parameter codebook: {PARAMETER_CODEBOOK_PATH}. Run the local training notebook first.'
    )
VQ_ARGUMENT = COMMUNICATION_VQ_PATH if COMMUNICATION_VQ_PATH.is_file() else None
if VQ_ARGUMENT is None:
    print('Communication VQ artifact is absent: the figure will show VQ payload as unavailable, not fabricate one.')

RUN_CONTEXT.update(
    {
        'stage1b_config': str(STAGE1B_CONFIG_PATH),
        'stage1c_config': str(STAGE1C_CONFIG_PATH),
        'model_dir': str(MODEL_DIR),
        'parameter_codebook': str(PARAMETER_CODEBOOK_PATH),
        'communication_vq': str(VQ_ARGUMENT) if VQ_ARGUMENT else None,
    }
)
print(json.dumps(RUN_CONTEXT, indent=2))
"""
        ),
        md("## 2. Transformer path: sequential knowledge base, semantic sidecar, and showcase figure"),
        py(
            """TRANSFORMER_FIGURE = OUTPUT_DIR / 'transformer_semantic_header_showcase.png'
transformer_result = run_semantic_demo(
    stage1b_config,
    param_extractor='transformer',
    model_dir=MODEL_DIR,
    communication_codebook_path=VQ_ARGUMENT,
    device='auto',
    output_plot=TRANSFORMER_FIGURE,
)
print(json.dumps({key: value for key, value in transformer_result.items() if key != 'parameter_bundles'}, indent=2))
print('Rendered figure:', TRANSFORMER_FIGURE)
"""
        ),
        md("## 3. Semantic parameter-codebook path"),
        py(
            """codebook_result = run_semantic_demo(
    stage1b_config,
    param_extractor='codebook',
    model_dir=MODEL_DIR,
    communication_codebook_path=VQ_ARGUMENT,
    semantic_codebook_path=PARAMETER_CODEBOOK_PATH,
    device='auto',
)
print(json.dumps({key: value for key, value in codebook_result.items() if key != 'parameter_bundles'}, indent=2))
"""
        ),
        md("## 4. Parameter differences and semantic-header overhead"),
        py(
            """import pandas as pd

rows = []
for image_entry, transformer_bundle, codebook_bundle in zip(
    stage1b_config['demo']['images'],
    transformer_result['parameter_bundles'],
    codebook_result['parameter_bundles'],
    strict=True,
):
    for parameter_name, transformer_value in transformer_bundle['values'].items():
        codebook_value = codebook_bundle['values'][parameter_name]
        transformer_score = transformer_value['value']
        codebook_score = codebook_value['value']
        rows.append(
            {
                'image': image_entry['filename'],
                'parameter': parameter_name,
                'transformer_value': transformer_score,
                'codebook_value': codebook_score,
                'absolute_difference': None
                if transformer_score is None or codebook_score is None
                else abs(transformer_score - codebook_score),
                'transformer_available': transformer_value['available'],
                'codebook_available': codebook_value['available'],
            }
        )
comparison = pd.DataFrame(rows)
display(comparison)
comparison.to_csv(OUTPUT_DIR / 'parameter_comparison.csv', index=False)
print(
    {
        'transformer_header_bytes': transformer_result['semantic_header_bytes'],
        'codebook_header_bytes': codebook_result['semantic_header_bytes'],
        'parameter_codebook_bytes': artifact_size_bytes(PARAMETER_CODEBOOK_PATH),
        'communication_vq_bytes': artifact_size_bytes(VQ_ARGUMENT) if VQ_ARGUMENT else None,
    }
)
"""
        ),
        md("## 5. Full-series latency and memory comparison"),
        py(
            """import contextlib
import io


def quiet_run(extractor_name: str):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_semantic_demo(
            stage1b_config,
            param_extractor=extractor_name,
            model_dir=MODEL_DIR,
            communication_codebook_path=VQ_ARGUMENT,
            semantic_codebook_path=PARAMETER_CODEBOOK_PATH if extractor_name == 'codebook' else None,
            device='auto',
        )


# These are environment-specific, end-to-end image-series measurements. They
# include model/artifact loading and must not be presented as radio latency.
benchmark_rows = [
    {'path': 'transformer', **benchmark_callable(lambda: quiet_run('transformer'), warmup_runs=1, measured_runs=3)},
    {'path': 'codebook', **benchmark_callable(lambda: quiet_run('codebook'), warmup_runs=1, measured_runs=3)},
]
benchmark = pd.DataFrame(benchmark_rows)
display(benchmark)
benchmark.to_csv(OUTPUT_DIR / 'latency_memory_comparison.csv', index=False)
"""
        ),
        md(
            """## 6. Validation limits

Inspect the saved PNG at reading size after execution. The codebook is a local ten-image engineering prototype only. Its comparison reports latency, memory, header size, and parameter differences; it does not establish accuracy, generalization, scheduler behavior, or physical-layer header overhead."""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(payload: dict, output: Path) -> None:
    import nbformat

    notebook = nbformat.from_dict(payload)
    for notebook_cell in notebook.cells:
        notebook_cell["id"] = uuid.uuid4().hex[:12]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    training_output = root / "Notebooks" / "Stage1C" / "01_train_parameter_codebook_local.ipynb"
    showcase_output = root / "Notebooks" / "Stage1B" / "02_transformer_vs_codebook_semantic_header.ipynb"
    write_notebook(build_training_notebook(), training_output)
    write_notebook(build_showcase_notebook(), showcase_output)
    print(training_output)
    print(showcase_output)


if __name__ == "__main__":
    main()
