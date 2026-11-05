"""Apply the reviewed four-owner layout once, preserving compatibility paths.

Run with --plan before --apply. Original bytes and hashes are saved under
results/cleanup. This tool never changes model, data, result or upstream trees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MOVES = {
    "Evaluation/contracts.py": "Evaluation/scheduling/contracts.py",
    "Evaluation/semantic_contracts.py": "Evaluation/scheduling/semantic_contracts.py",
    "Evaluation/scheduling_requests.py": "Evaluation/scheduling/requests.py",
    "Evaluation/semantic_streams.py": "Evaluation/scheduling/streams.py",
    "Evaluation/simulation.py": "Evaluation/scheduling/simulation.py",
    "Evaluation/event_simulation.py": "Evaluation/scheduling/event_simulation.py",
    "Evaluation/semantic_evaluator.py": "Evaluation/evaluation/semantic.py",
    "Evaluation/feature_cache.py": "Evaluation/data/feature_cache.py",
    "Evaluation/ladi_manifest.py": "Evaluation/data/ladi_manifest.py",
    "Evaluation/kaggle_dataset_bundle.py": "Evaluation/data/kaggle_bundle.py",
    "Evaluation/latent_contracts.py": "Evaluation/latent/contracts.py",
    "Evaluation/latent_simulation.py": "Evaluation/latent/simulation.py",
    "Evaluation/latent_evaluation.py": "Evaluation/latent/evaluation.py",
    "Evaluation/experiments/static_latent_experiment.py": "Evaluation/latent/experiment.py",
    "scripts/run_static_latent_scheduler.py": "Evaluation/latent/cli.py",
    "Scheduler/policies.py": "Scheduler/baselines/policies.py",
    "Scheduler/fixtures.py": "Scheduler/baselines/fixtures.py",
    "Scheduler/semantic_policies.py": "Scheduler/semantic/policies.py",
    "Scheduler/feature_adapter.py": "Scheduler/semantic/features.py",
    "Scheduler/policy_registry.py": "Scheduler/semantic/registry.py",
    "Scheduler/latent_policies.py": "Scheduler/latent/policies.py",
    "Archived/owners_before_simplification_20261027/SemParam/clip_backbone.py": "Archived/owners_before_simplification_20261027/SemParam/features/clip.py",
    "Archived/owners_before_simplification_20261027/SemParam/extractors.py": "Archived/owners_before_simplification_20261027/SemParam/features/extractors.py",
    "Archived/owners_before_simplification_20261027/SemParam/observation_builder.py": "Archived/owners_before_simplification_20261027/SemParam/features/observation.py",
    "Archived/owners_before_simplification_20261027/SemParam/scheduler_features.py": "Archived/owners_before_simplification_20261027/SemParam/features/scheduler.py",
    "Archived/owners_before_simplification_20261027/SemParam/ladi_prediction_provider.py": "Archived/owners_before_simplification_20261027/SemParam/features/ladi_prediction.py",
    "Archived/owners_before_simplification_20261027/SemParam/latent_features.py": "Archived/owners_before_simplification_20261027/SemParam/features/latent.py",
    "Archived/owners_before_simplification_20261027/SemParam/delivered_history.py": "Archived/owners_before_simplification_20261027/SemParam/history/delivered.py",
    "Archived/owners_before_simplification_20261027/SemParam/semantic_header.py": "Archived/owners_before_simplification_20261027/SemParam/headers/semantic.py",
    "Archived/owners_before_simplification_20261027/SemParam/packet_builder.py": "Archived/owners_before_simplification_20261027/SemParam/headers/packet.py",
}
SCRIPT_GROUPS = {
    "codecs": ("bootstrap_swinjscc", "download_kodak24", "register_deepjscc",
               "render_kodak_benchmark_table", "run_communication_benchmark",
               "run_deepjscc", "run_ntscc_benchmark", "test_deepjscc_registry"),
    "data": ("build_ladi_review_page", "curate_ladi260", "curate_ladi60",
             "freeze_ladi60_prototype", "package_ladi260_dataset", "package_ladi_review_dataset"),
    "notebooks": ("build_downloaded_run_review", "build_parameter_extraction_demo_notebook",
                  "build_semantic_scheduler_notebooks", "build_stage1a_notebook",
                  "build_stage1b_notebooks", "build_static_latent_notebook"),
    "reports": ("build_scheduler_explanation_doc", "plot_dynamic_priority_matches",
                "plot_stage1a_four_way_comparison", "plot_stage1a_snr_psnr_ssim_cbr",
                "plot_static_dynamic_scheduler_comparison"),
    "kaggle": ("prepare_semantic_kaggle", "prepare_static_latent_kaggle",
               "package_semantic_payload_costs", "package_semantic_train_output"),
    "scheduler": ("evaluate_downloaded_scheduler", "evaluate_dynamic_overload_frozen_dqn",
                  "evaluate_fixed_decision_matches", "replay_static_latent_schedule",
                  "train_class_priority_dqn"),
    "semparam": ("train_parameter_codebook",),
    "maintenance": ("refresh_graph",),
}
MOVES = dict(SOURCE_MOVES)
for group, names in SCRIPT_GROUPS.items():
    for name in names:
        MOVES[f"scripts/{name}.py"] = f"scripts/{group}/{name}.py"


def module(path):
    return path[:-3].replace("/", ".")


def rewrite(text):
    for old, new in sorted(MOVES.items(), key=lambda pair: -len(pair[0])):
        text = re.sub(r"(?<![\w.])" + re.escape(module(old)) + r"(?![\w.])", module(new), text)
    return text


def alias(target, *, command=False):
    prefix = ('"""Compatibility alias; canonical implementation: ' + target + '."""\n')
    if command:
        prefix += "from pathlib import Path\nimport sys\n\nROOT = Path(__file__).resolve().parents[1]\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n"
        return prefix + ("if __name__ == '__main__':\n    from runpy import run_module\n"
            f"    run_module('{target}', run_name='__main__')\nelse:\n"
            "    from importlib import import_module\n    import sys\n"
            f"    sys.modules[__name__] = import_module('{target}')\n")
    return prefix + ("from importlib import import_module\nimport sys\n\n"
                     f"sys.modules[__name__] = import_module('{target}')\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(MOVES, indent=2))
        return
    for old, new in MOVES.items():
        if not (ROOT / old).is_file() or (ROOT / new).exists():
            raise ValueError(f"Migration not applicable: {old} -> {new}")
        if not (ROOT / new).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Destination escaped repository")
    output = ROOT / "results/cleanup" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    originals = {old: (ROOT / old).read_bytes() for old in MOVES}
    manifest = []
    for old, new in MOVES.items():
        saved = output / "originals" / old
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(originals[old])
        manifest.append({"old": old, "canonical": new,
                         "original_sha256": hashlib.sha256(originals[old]).hexdigest()})
    # Write all destinations before replacing any original with an alias.
    for old, new in MOVES.items():
        path = ROOT / new
        path.parent.mkdir(parents=True, exist_ok=True)
        text = rewrite(originals[old].decode("utf-8-sig").replace("\r\n", "\n"))
        if old.startswith("scripts/"):
            text = text.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]")
        compile(text, str(path), "exec")
        path.write_text(text, encoding="utf-8", newline="\n")
        for parent in (path.parent,):
            init = parent / "__init__.py"
            if not init.exists():
                init.write_text('"""' + parent.relative_to(ROOT).as_posix() + ' implementation package."""\n', encoding="utf-8")
    (ROOT / "scripts/__init__.py").write_text('"""Repository utilities grouped by purpose."""\n', encoding="utf-8")
    for old, new in MOVES.items():
        (ROOT / old).write_text(alias(module(new), command=old.startswith("scripts/")), encoding="utf-8")
    # Update clean canonical consumers. Tests and dirty teammate-owned modules
    # intentionally continue to exercise the preserved compatibility imports.
    dirty = set(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines())
    consumers = []
    for owner in ("Evaluation", "Archived.owners_before_simplification_20261027.SemParam", "Scheduler", "Codec"):
        for path in sorted((ROOT / owner).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            protected = {"Evaluation/class_priority_profiles.py", "Scheduler/training/priority_match_objective.py"}
            if relative in MOVES or "tests" in path.parts or "__pycache__" in path.parts or relative in dirty or relative in protected:
                continue
            before = path.read_text(encoding="utf-8-sig")
            after = rewrite(before)
            if before != after:
                saved = output / "originals" / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(path.read_bytes())
                compile(after, str(path), "exec")
                path.write_text(after, encoding="utf-8")
                consumers.append(relative)
    record = {"moves": manifest, "updated_consumers": consumers,
              "preserved_areas": ["models", "data", "references", "external", "Notebooks", "downloaded", "Test", "src", "semcom_benchmark"]}
    (output / "migration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"moved_implementations": len(MOVES), "updated_consumers": len(consumers), "evidence": str(output)}, indent=2))


if __name__ == "__main__":
    main()
