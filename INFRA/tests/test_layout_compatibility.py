"""Architecture gates for canonical modules and old import/command surfaces."""
import importlib
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

from INFRA.configuration import ROOT


@pytest.mark.parametrize("old,new", [
    ("INFRA.contracts", "INFRA.scheduling.contracts"),
    ("INFRA.semantic_contracts", "INFRA.scheduling.semantic_contracts"),
    ("INFRA.event_simulation", "INFRA.scheduling.event_simulation"),
    ("INFRA.feature_cache", "INFRA.data.feature_cache"),
    ("INFRA.ladi_manifest", "INFRA.data.ladi_manifest"),
    ("INFRA.latent_contracts", "INFRA.latent.contracts"),
    ("INFRA.latent_simulation", "INFRA.latent.simulation"),
    ("INFRA.latent_evaluation", "INFRA.latent.evaluation"),
    ("INFRA.experiments.static_latent_experiment", "INFRA.latent.experiment"),
    ("Scheduler.policies", "Scheduler.baselines.policies"),
    ("Scheduler.feature_adapter", "Scheduler.semantic.features"),
    ("Scheduler.semantic_policies", "Scheduler.semantic.policies"),
    ("Scheduler.policy_registry", "Scheduler.semantic.registry"),
    ("Scheduler.latent_policies", "Scheduler.latent.policies"),
    ("SemParam.clip_backbone", "SemParam.features.clip"),
    ("SemParam.scheduler_features", "SemParam.features.scheduler"),
    ("SemParam.latent_features", "SemParam.features.latent"),
    ("SemParam.delivered_history", "SemParam.history.delivered"),
    ("SemParam.semantic_header", "SemParam.headers.semantic"),
    ("scripts.build_ladi_review_page", "scripts.data.build_ladi_review_page"),
    ("scripts.freeze_ladi60_prototype", "scripts.data.freeze_ladi60_prototype"),
])
def test_old_modules_are_same_objects_as_canonical_modules(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


@pytest.mark.parametrize("module,name", [
    ("INFRA.contracts", "Job"),
    ("INFRA.latent_contracts", "LatentJob"),
    ("INFRA.semantic_contracts", "FeatureValue"),
])
def test_old_pickle_class_references_still_resolve(module, name):
    reference = f"c{module}\n{name}\n.".encode()
    assert pickle.loads(reference) is getattr(importlib.import_module(module), name)


def test_latent_policy_import_does_not_load_neural_or_codec_packages():
    source = "import sys; import Scheduler.latent.policies; assert not any(name == 'torch' or name.startswith(('torch.', 'transformers', 'EncDec', 'SemParam')) for name in sys.modules)"
    subprocess.run([sys.executable, "-c", source], cwd=ROOT, check=True)


def test_old_script_runs_from_unrelated_directory(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/run_static_latent_scheduler.py"),
               "--batch", str(ROOT / "Scheduler/config/static_latent_demo.json"),
               "--policies", "semantic", "--output", str(tmp_path)]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    assert len(list((tmp_path / "Scheduler").glob("*/semantic.trace.json"))) == 1


def test_cached_static_runtime_needs_no_optional_site_packages(tmp_path):
    # -S removes site-packages so numpy/torch cannot hide a dependency.
    code = ("from INFRA.latent.experiment import compare_batch; "
            f"compare_batch('Scheduler/config/static_latent_demo.json', {str(tmp_path)!r}, verbose=False)")
    subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_lazy_parameter_exports_preserve_public_class_identity():
    from SemParam import SemanticHeaderCodec
    from SemParam.headers.semantic import SemanticHeaderCodec as canonical
    assert SemanticHeaderCodec is canonical
