"""Build a portable Kaggle notebook from canonical, hash-recorded source files.

No submission occurs here. Notebook is private and targets the explicitly
configured owner; account verification remains a separate pre-submit gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "Archived/owners_before_simplification_20261027/SemParam/disaster_four_parameters.py", "Scheduler/disaster_rankers.py",
    "Codec/Models/scale_hyperprior_bundle.py", "Scheduler/training/disaster_ranking_study.py",
)
EVIDENCE_SOURCES = SOURCES + (
    "Archived/owners_before_simplification_20261027/SemParam/disaster_evidence_header.py", "Scheduler/training/disaster_trace.py",
)


def build(evidence=False):
    cells = []
    md = lambda text: cells.append(nbformat.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbformat.v4.new_code_cell(text))
    introduction = """# Disaster-only four-parameter latent scheduling study

**Status:** executable research workflow, not a claim of completed training.

Input: native MEDIC images/tables; output: every epoch checkpoint, prediction,
data audit, serialized request, actual digital bundle, raw schedule trace,
comparison table and two figures in `/kaggle/working`.

The frozen CLIP backbone supplies learned UE parameters. The controller compares
a classifier, ordinal regression, a linear pairwise scorer, RankNet and
LambdaMART. Public damage grades validate damage ordering only. They do **not**
establish the correct age/novelty tradeoff. Real contextual training preferences
and the independent reviewer audit remain a separate journal claim gate.

The traffic/UE/capture sequence is explicitly synthetic. Native dataset labels
are never replaced with synthetic emergency labels. Whole digital bundles are
measured using a frozen scale-hyperprior codec; no float tensor size is called
transmitted bits. The replay is a declared communication abstraction, not a
full MAC/PHY or device-latency validation.
"""
    if evidence:
        introduction = introduction.replace("four-parameter", "seven-field evidence")
        introduction += "\nThis additive version retains original P1–P4 and exposes A1–A3. MEDIC native grades remain damage-only supervision; no contextual serve-first labels are inferred. The corrected replay re-times selected jobs and reports realized contention.\n"
    md(introduction)
    bootstrap = """from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, sys, uuid, traceback, shutil, subprocess

RUN_ID = os.environ.get('DISASTER_RUN_ID', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
RUN = Path('/kaggle/working/disaster-ranking') / RUN_ID
RUN.mkdir(parents=True, exist_ok=False)
CONFIG = {'seeds': [11, 23, 37], 'head_epochs': 30, 'ranker_epochs': 30,
          'caps': {'train': 6000, 'validation': 1000, 'test': 1000},
          'workloads': 5, 'max_replay_jobs': 128, 'quality': 3,
          'require_gpu': True, 'allow_official_download': True,
          'dataset_root': os.environ.get('MEDIC_ROOT', ''),
          'clip_model': 'openai/clip-vit-base-patch32',
          'scope': 'native damage ranking; contextual tradeoff not validated'}
(RUN / 'configuration.json').write_text(json.dumps(CONFIG, indent=2))
def persist_status(value):
    (RUN / 'status.json').write_text(json.dumps(value, indent=2), encoding='utf-8')

def persist_failure(stage, exc):
    persist_status({'state': 'failed', 'stage': stage, 'error': str(exc),
                    'dependent_results_valid': False})
    with (RUN / 'execution.log').open('a', encoding='utf-8') as log:
        traceback.print_exc(file=log)
    print('DOWNLOAD ARCHIVE:', shutil.make_archive(str(RUN.parent / RUN.name), 'zip', RUN))

persist_status({'state': 'in_progress', 'stage': 'bootstrap'})
print('ALL ARTIFACTS:', RUN)
"""
    if evidence:
        bootstrap = bootstrap.replace("Path('/kaggle/working/disaster-ranking')", "Path('/kaggle/working/disaster-evidence-seven')")
        bootstrap = bootstrap.replace("native damage ranking; contextual tradeoff not validated",
                                      "seven-field native damage diagnostic; contextual tradeoff not validated")
    code(bootstrap)
    code("""# Keep Kaggle's accelerator-matched torch/torchvision installation.
# CompressAI 1.2.8 requires NumPy <2. Install a matching pandas wheel too;
# downgrading NumPy alone breaks Kaggle's preinstalled pandas binary ABI.
dependencies = ['numpy==1.26.4', 'pandas==2.2.3', 'transformers==4.57.6',
                'huggingface-hub>=0.35,<2', 'scikit-learn>=1.5,<2', 'lightgbm==4.6.0',
                'compressai==1.2.8', 'Pillow>=10,<13', 'matplotlib>=3.8,<4', 'requests']
try:
    persist_status({'state': 'in_progress', 'stage': 'dependencies'})
    with (RUN / 'dependency_install.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *dependencies],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
except Exception as exc:
    persist_failure('dependencies', exc)
    raise
""")
    source_map = {name: (ROOT / name).read_text(encoding="utf-8") for name in (EVIDENCE_SOURCES if evidence else SOURCES)}
    checksums = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in source_map.items()}
    code("# Canonical source snapshot: readable and independently hashable, not a second implementation.\n"
         + "SOURCE_FILES = " + repr(source_map) + "\n"
         + "SOURCE_HASHES = " + repr(checksums) + "\n"
         + """SOURCE = RUN / 'source'
try:
    for name, text in SOURCE_FILES.items():
        if hashlib.sha256(text.encode()).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError('Source hash mismatch: ' + name)
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        for parent in target.parents:
            if parent == SOURCE:
                break
            (parent / '__init__.py').touch(exist_ok=True)
    (RUN / 'source_hashes.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
    sys.path.insert(0, str(SOURCE))
except Exception as exc:
    persist_failure('source_snapshot', exc)
    raise
""")
    md("""## Execution and durable outputs

Mount MEDIC as a Kaggle input dataset and set `MEDIC_ROOT`, or permit the
official archive download. The resumable official archive download verifies
size and ETag continuity and records a local SHA-256; extraction checks gzip
integrity and retains a deterministic image subset and the native tables.
Budget disk for the approximately 11 GB archive plus sampled extraction,
model caches and run artifacts. The dataset/archive stay outside the result
ZIP. Network/disk/runtime availability is checked through actual execution,
not assumed. A local hash records downloaded bytes, not publisher authenticity.

The runner writes `status.json` on failure and packages whatever artifacts
exist. The full notebook console is also captured in `execution.log`. A failed
stage stops dependent claims; no random features or dummy payload sizes replace
missing real inputs. Local smoke tests are separate from these real-data runs.
""")
    execution = """import traceback, shutil
from contextlib import redirect_stdout, redirect_stderr

class Tee:
    def __init__(self, console, log): self.console, self.log = console, log
    def write(self, value):
        self.console.write(value); self.log.write(value); self.log.flush()
    def flush(self): self.console.flush(); self.log.flush()

def enter_stage(value):
    global stage
    stage = value
    persist_status({'state': 'in_progress', 'stage': stage})

enter_stage('preflight')
error = None
with (RUN / 'execution.log').open('a', encoding='utf-8') as log:
  with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
    try:
        import pandas as pd
        import numpy as np
        import torch
        from Scheduler.training.disaster_ranking_study import (
            save_json, provenance, fetch_medic_subset, load_medic, cache_clip_embeddings,
            fit_parameter_heads, extractor_metrics, prepare_ranking_rows,
            train_ranker_family, measure_test_payloads, evaluate_service_replay, render_study_outputs,
        )
        env = provenance(RUN)
        print(env)
        if CONFIG['require_gpu'] and not env['cuda']:
            raise RuntimeError('GPU required: no silent CPU training substitution')
        enter_stage('dataset')
        root = Path(CONFIG['dataset_root']) if CONFIG['dataset_root'] else None
        if root is None:
            matches = sorted(Path('/kaggle/input').rglob('MEDIC_train.tsv'))
            if len(matches) == 1:
                root = matches[0].parent
            elif not matches and CONFIG['allow_official_download']:
                root = fetch_medic_subset(Path('/kaggle/working/medic-source-subset'), sample_modulus=5)
            else:
                raise RuntimeError('Set MEDIC_ROOT to one unambiguous real MEDIC dataset')
        frame = load_medic(root, RUN, caps=CONFIG['caps'])
        print(frame.groupby('role').size())
        enter_stage('frozen_backbone')
        embeddings = cache_clip_embeddings(frame, RUN, model_name=CONFIG['clip_model'])
        del_env = None
        payloads = None
        for seed in CONFIG['seeds']:
            seed_dir = RUN / f'seed_{seed}'
            seed_dir.mkdir()
            enter_stage(f'parameter_training_seed_{seed}')
            importance, class_ids = fit_parameter_heads(embeddings, frame, seed_dir / 'parameters', seed=seed, epochs=CONFIG['head_epochs'])
            extractor_metrics(frame, seed_dir / 'parameters/predictions.npz', seed_dir)
            enter_stage(f'ranking_rows_seed_{seed}')
            rows, features = prepare_ranking_rows(frame, embeddings, importance, class_ids, seed_dir)
            enter_stage(f'ranker_training_seed_{seed}')
            print(train_ranker_family(rows, features, seed_dir / 'rankers', seed=seed, epochs=CONFIG['ranker_epochs']))
            if payloads is None:
                enter_stage('actual_digital_payloads')
                payloads = measure_test_payloads(frame, rows, RUN / 'payloads', quality=CONFIG['quality'])
                payloads.to_csv(RUN / 'payload_measurements.csv', index=False)
            enter_stage(f'service_replay_seed_{seed}')
            evaluate_service_replay(rows, payloads, seed_dir / 'rankers', seed_dir / 'replay', seed=seed,
                                    workloads=CONFIG['workloads'], max_jobs=CONFIG['max_replay_jobs'])
            persist_status({'state': 'in_progress', 'stage': stage, 'completed_seed': seed})
        enter_stage('tables_and_figures')
        print(render_study_outputs(RUN))
        save_json(RUN / 'status.json', {'state': 'completed_bounded_damage_study',
                  'contextual_four_parameter_journal_validation': False,
                  'human_preferences_collected': False, 'novel_algorithm_proven': False})
    except Exception as exc:
        error = exc
        traceback.print_exc()
        persist_status({'state': 'failed', 'stage': stage, 'error': str(exc),
                  'dependent_results_valid': False})
    finally:
        print('OUTPUT DIRECTORY:', RUN)
        archive = shutil.make_archive(str(RUN.parent / RUN.name), 'zip', RUN)
        print('DOWNLOAD ARCHIVE:', archive)
if error is not None:
    raise error
"""
    if evidence:
        execution = execution.replace(
            "rows, features = prepare_ranking_rows(frame, embeddings, importance, class_ids, seed_dir)",
            "rows, features = prepare_ranking_rows(frame, embeddings, importance, class_ids, seed_dir, evidence_prediction_npz=seed_dir / 'parameters/predictions.npz')")
        execution = execution.replace("'completed_bounded_damage_study'", "'completed_seven_field_damage_diagnostic'")
        # Kaggle's live IPython process has already imported NumPy 2.x before
        # pip installs CompressAI's required NumPy 1.x. A fresh subprocess is
        # necessary; re-importing pandas in the same process has an ABI error.
        prefix = ("from pathlib import Path\nimport sys, json, shutil\n"
                  "RUN = Path(" + repr("__RUN_PATH__") + ")\n"
                  "CONFIG = json.loads((RUN / 'configuration.json').read_text())\n"
                  "sys.path.insert(0, str(RUN / 'source'))\n"
                  "def persist_status(value):\n"
                  "    (RUN / 'status.json').write_text(json.dumps(value, indent=2), encoding='utf-8')\n")
        runner = prefix + execution
        code("""# Execute in a fresh interpreter after dependency installation.
runner_source = """ + repr(runner) + """
runner_source = runner_source.replace('__RUN_PATH__', str(RUN))
runner_path = RUN / 'run_study.py'
runner_path.write_text(runner_source, encoding='utf-8')
subprocess.run([sys.executable, '-u', str(runner_path)], check=True)
""")
    else:
        code(execution)
    md("""## Read the saved results

`comparison_table.csv`, `selection.json`, `01_ranking_comparison.png`, and
`02_dynamic_delivery.png` are produced only after the preceding stages succeed.
Epoch checkpoints, native labels, predictions and per-query/per-job traces
remain in the same run directory. The selected method uses validation only.
The test set is reported, not used to nominate a winner retrospectively.

**Not established by this notebook alone:** real emergency importance,
the normative age/novelty tradeoff, global information novelty, real UE latency,
hard wireless deadlines, or algorithmic novelty. These are independent claim gates.
""")
    code("""# Avoid importing pandas/NumPy in the pre-install notebook interpreter.
import csv
from html import escape
from IPython.display import display, HTML, Image
with (RUN / 'comparison_table.csv').open(newline='', encoding='utf-8') as table_file:
    table_rows = list(csv.reader(table_file))
table_html = '<table>' + ''.join(
    '<tr>' + ''.join('<' + tag + '>' + escape(value) + '</' + tag + '>'
                     for value in row) + '</tr>'
    for tag, row in [('th' if index == 0 else 'td', row)
                     for index, row in enumerate(table_rows)]) + '</table>'
display(HTML(table_html))
display(Image(filename=str(RUN / '01_ranking_comparison.png')))
display(Image(filename=str(RUN / '02_dynamic_delivery.png')))
print((RUN / 'selection.json').read_text())
""")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                            "language_info": {"name": "python"}})
    nbformat.validate(notebook)
    destination = ROOT / ("Scheduler/notebooks/04_disaster_evidence_study.ipynb" if evidence
                          else "Scheduler/notebooks/03_disaster_four_parameter_study.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return destination, checksums


def stage(owner, evidence=False):
    import shutil
    if not owner or "/" in owner:
        raise ValueError("Explicit Kaggle owner required")
    source, hashes = build(evidence=evidence)
    folder = ROOT / ("kaggle/disaster-evidence-seven" if evidence else "kaggle/disaster-four-parameter")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, folder / source.name)
    slug = "disaster-seven-field-evidence-study" if evidence else "disaster-four-parameter-study"
    title = "Disaster Seven Field Evidence Study" if evidence else "Disaster Four Parameter Study"
    metadata = {"id": f"{owner}/{slug}", "title": title,
                "code_file": source.name, "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (folder / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", help="Stage private metadata; does not submit or change credentials")
    parser.add_argument("--evidence", action="store_true", help="Build the additive seven-field evidence notebook")
    args = parser.parse_args()
    print(stage(args.owner, evidence=args.evidence) if args.owner else build(evidence=args.evidence)[0])
