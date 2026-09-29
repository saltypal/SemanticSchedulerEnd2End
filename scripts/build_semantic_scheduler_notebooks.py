"""Generate the two canonical, self-contained semantic scheduler notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Scheduler" / "notebooks"


def md(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


BOOTSTRAP = code(r'''
    from pathlib import Path
    import os, shutil, subprocess, sys

    REPO_URL = "https://github.com/saltypal/SemanticSchedulerEnd2End.git"
    BRANCH = "semantic-aware-ai-scheduler"
    IS_KAGGLE = Path("/kaggle/working").exists()
    IS_COLAB = "COLAB_RELEASE_TAG" in os.environ
    WORKSPACE = Path("/kaggle/working" if IS_KAGGLE else os.environ.get("SCHEDULER_WORKSPACE", Path.cwd())).resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    PROJECT_ROOT = WORKSPACE / "project-src"
    if PROJECT_ROOT.exists():
        if PROJECT_ROOT.resolve().parent != WORKSPACE:
            raise RuntimeError("Refusing to remove a checkout outside the disposable workspace")
        shutil.rmtree(PROJECT_ROOT)
    subprocess.run(["git", "clone", "--single-branch", "--branch", BRANCH, REPO_URL, str(PROJECT_ROOT)], check=True)
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "checkout", "--detach", f"origin/{BRANCH}"], cwd=PROJECT_ROOT, check=True)
    SOURCE_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    sys.path.insert(0, str(PROJECT_ROOT))
    print({"environment": "kaggle" if IS_KAGGLE else "colab" if IS_COLAB else "local",
           "branch": BRANCH, "exact_commit": SOURCE_COMMIT, "checkout": str(PROJECT_ROOT)})
''')

DEPENDENCIES = code(r'''
    import importlib.metadata as package_metadata
    import json, platform
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(PROJECT_ROOT / "environment/requirements-semantic-scheduler-kaggle.txt")], check=True)
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from IPython.display import display
    if os.environ.get("TPU_NAME") or os.environ.get("XRT_TPU_CONFIG"):
        raise RuntimeError("This PyTorch DQN notebook has no TPU/XLA execution path; select CPU or CUDA")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENVIRONMENT = {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": DEVICE, "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "packages": {name: package_metadata.version(name) for name in
                     ("torch", "numpy", "pandas", "scikit-learn", "transformers", "datasets")},
    }
    print(ENVIRONMENT)
''')


def train_notebook():
    cells = [
        md('''
            # Train the semantic whole-latent scheduler

            **Experiment boundary:** 60 distinct LADI v2 images, 36/12/12 group-safe
            split, five simulated UEs, whole SwinJSCC latent jobs. This is the
            stateful dynamic complement to the supplied 12-image static baseline.
            P1/P4 are classifier-derived proxies, P2 is CLIP task relevance, P3 is
            recomputed from successfully delivered history, and P5/P6/P7 use the
            fixed prototype rubric. These annotations are not human reviewed.
            The DQN learns a *next-job decision*, not a one-shot permutation.

            This notebook starts with a disposable clean clone of the latest remote
            `semantic-aware-ai-scheduler` commit. Results go to a separate persistent
            run directory. It requires a complete experiment-ready manifest and
            measured payload when measured mode is selected.
        '''),
        md('## 1. Clean reset and exact source revision'), BOOTSTRAP,
        md('## 2. Dependencies, hardware, and persistent output'), DEPENDENCIES,
        code(r'''
            from datetime import datetime, timezone
            from hashlib import sha256
            from dataclasses import asdict
            import yaml
            from INFRA.ladi_manifest import validate_ladi60_manifest, render_contact_sheet
            from INFRA.kaggle_dataset_bundle import resolve_ladi_manifest
            from INFRA.semantic_config import load_scheduler_config, load_ue_profiles
            from INFRA.experiments.semantic_scheduler_experiment import (
                TASK_ID, prepare_feature_cache, load_catalog, build_suites_from_manifest,
                run_provenance, simulated_fixed_payload_costs,
            )
            from INFRA.semantic_contracts import ResourceBudgetSemantic
            from INFRA.learning.supervised_training import (
                fit_supervised_models, fit_image_priority_classifier)
            from INFRA.learning.dqn_training import DQNConfig, train_dqn_seeds
            from EncDec.integration.scheduler_replay import cache_swinjscc_payloads

            RESULTS_ROOT = Path(os.environ.get("SCHEDULER_RESULTS_DIR",
                "/kaggle/working/outputs" if IS_KAGGLE else WORKSPACE / "semantic-scheduler-results"))
            RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + SOURCE_COMMIT[:8]
            RUN_ROOT = RESULTS_ROOT / RUN_ID
            RUN_ROOT.mkdir(parents=True, exist_ok=False)
            MANIFEST_PATH = Path(os.environ.get("SCHEDULER_MANIFEST",
                "/kaggle/input/semantic-scheduler-ladi60-public/ladi60_semantic_scheduler_v1.csv"))
            PAYLOAD_MODE = os.environ.get("SCHEDULER_PAYLOAD_MODE", "measured")
            if PAYLOAD_MODE not in ("measured", "simulated_fixed"):
                raise ValueError("SCHEDULER_PAYLOAD_MODE must be measured or simulated_fixed")
            MODEL_DIR = Path(os.environ.get("SCHEDULER_SWIN_MODEL_DIR", PROJECT_ROOT / "models/SwinJSCC"))
            UPSTREAM_ROOT = PROJECT_ROOT / "external/SwinJSCC"
            SCHEDULER_SETTINGS = load_scheduler_config(PROJECT_ROOT / "INFRA/config/semantic_scheduler_demo.yaml")
            UE_PROFILES = load_ue_profiles(PROJECT_ROOT / "INFRA/config/ue_profiles.yaml")
            DQN_SETTINGS = yaml.safe_load((PROJECT_ROOT / "Scheduler/config/semantic_dqn.yaml").read_text(encoding="utf-8"))
            if DQN_SETTINGS["episodes"] != SCHEDULER_SETTINGS.train_episodes_per_seed:
                raise ValueError("DQN and stream episode counts disagree")
            (RUN_ROOT / "environment.json").write_text(json.dumps(ENVIRONMENT, indent=2), encoding="utf-8")
            print({"run_id": RUN_ID, "output": str(RUN_ROOT), "payload_mode": PAYLOAD_MODE})
        '''),
        md('''
            ## 3. Experiment-ready prototype dataset gate

            Attach the private prototype dataset and set `SCHEDULER_MANIFEST` if
            needed. It contains 60 unique images, six selection strata, 30
            CLIP-proposed evidence pairs, and a 36/12/12 group-safe split.
            P7=1 and the evidence pairs are prototype assumptions, not reviewed truth.
        '''),
        code(r'''
            MANIFEST_PATH = resolve_ladi_manifest(MANIFEST_PATH, RUN_ROOT / "input_bundle")
            if not MANIFEST_PATH.is_file():
                raise FileNotFoundError(
                    f"Prototype manifest missing: {MANIFEST_PATH}. Attach the private LADI dataset "+
                    "or set SCHEDULER_MANIFEST before running this notebook.")
            reviewed_rows = validate_ladi60_manifest(MANIFEST_PATH, require_reviewed=True)
            frame = pd.DataFrame(reviewed_rows)
            display(frame.groupby(["split", "selection_stratum"]).size().unstack(fill_value=0))
            sheet = render_contact_sheet(MANIFEST_PATH, RUN_ROOT / "ladi60_contact_sheet.png")
            from PIL import Image
            display(Image.open(sheet))
            print("All 60 hashes and fixed prototype annotations validated; no human-review claim is made.")
        '''),
        md('## 4. Extract CLIP features and cached image-classifier predictions once'),
        code(r'''
            cache = prepare_feature_cache(MANIFEST_PATH, RUN_ROOT / "cache",
                include_ladi_predictions=True, device=DEVICE)
            feature_rows = [json.loads(line) for line in cache["features"].read_text(encoding="utf-8").splitlines()]
            extraction = pd.DataFrame({"image_id": row["image_id"],
                                       "latency_ms": row["extraction_latency_ms"]} for row in feature_rows)
            display(extraction.describe(include="all"))
            print("P1/P4 come from the cached image classifier; embeddings remain internal and refresh P3 at every decision.")
        '''),
        md('''
            ## 5. Cache payload cost once

            Measured mode runs the existing checkpoint and records complex channel
            uses separately from latent storage bytes. The checkpoint demands its
            exported Python/PyTorch runtime; an incompatibility stops this cell.
            `simulated_fixed` is only for an explicitly labelled infrastructure smoke run.
        '''),
        code(r'''
            if PAYLOAD_MODE == "simulated_fixed":
                PAYLOAD_COSTS = simulated_fixed_payload_costs(
                    MANIFEST_PATH, RUN_ROOT / "payload" / "payload_costs.csv")
            else:
                supplied_costs = os.environ.get("SCHEDULER_PAYLOAD_COSTS_CSV")
                if not supplied_costs:
                    attached_costs = MANIFEST_PATH.parent / "payload_costs.csv"
                    if not attached_costs.is_file():
                        attached_costs = Path("/kaggle/input/semantic-scheduler-measured-costs/payload_costs.csv")
                    supplied_costs = str(attached_costs) if attached_costs.is_file() else None
                if supplied_costs:
                    PAYLOAD_COSTS = Path(supplied_costs)
                    if not PAYLOAD_COSTS.is_file():
                        raise FileNotFoundError(PAYLOAD_COSTS)
                else:
                    PAYLOAD_COSTS = cache_swinjscc_payloads(
                        reviewed_rows, RUN_ROOT / "payload", model_dir=MODEL_DIR,
                        upstream_root=UPSTREAM_ROOT, device=DEVICE)
            cost_frame = pd.read_csv(PAYLOAD_COSTS)
            if cost_frame.resource_unit.nunique() != 1:
                raise ValueError("A run cannot mix resource units")
            if PAYLOAD_MODE == "measured" and cost_frame.resource_unit.iloc[0] != "complex_channel_uses":
                raise ValueError("Measured mode requires checkpoint-derived channel uses")
            display(cost_frame.head())
        '''),
        md('## 6. Freeze streams, UE profiles, deadlines, and transport tapes'),
        code(r'''
            catalogue = load_catalog(MANIFEST_PATH, cache["features"], PAYLOAD_COSTS)
            suites = build_suites_from_manifest(catalogue, MANIFEST_PATH,
                                                 config=SCHEDULER_SETTINGS,
                                                 ue_profiles=UE_PROFILES)
            unit = catalogue[0].candidate.resource_unit
            service_rate = max(item.candidate.resource_units for item in catalogue) / 10.0
            budget = ResourceBudgetSemantic(service_rate, resource_unit=unit)
            print({"train_episodes_per_seed": SCHEDULER_SETTINGS.train_episodes_per_seed,
                   "validation_jobs_per_episode": len(suites["validation"][0].jobs),
                   "validation_episodes": len(suites["validation"]),
                   "jobs_per_train_episode": len(suites["train_by_seed"][42][0].jobs),
                   "service_rate_units_per_ms": service_rate, "resource_unit": unit})
            episode_file = RUN_ROOT / "train_episodes.jsonl"
            with episode_file.open("w", encoding="utf-8") as handle:
                for seed, episodes in suites["train_by_seed"].items():
                    for episode in episodes:
                        compact = {"episode_id": episode.episode_id, "seed": episode.seed,
                                   "jobs": [{"job_id": job.candidate.job_id,
                                             "image_id": job.candidate.image_id,
                                             "ue_id": job.candidate.ue_id,
                                             "arrival_ms": job.candidate.arrival_ms,
                                             "deadline_ms": job.candidate.deadline_ms,
                                             "transport_success": job.transport_success}
                                            for job in episode.jobs]}
                        handle.write(json.dumps(compact) + "\n")
        '''),
        md('## 7. Fit image-only and feature-based models on 36 training images'),
        code(r'''
            split_by_image = {row["image_id"]: row["split"] for row in reviewed_rows}
            train_jobs = [job for job in catalogue if split_by_image[job.candidate.image_id] == "train"]
            resolved_config = {
                "scheduler": asdict(SCHEDULER_SETTINGS), "ue_profiles": [asdict(p) for p in UE_PROFILES],
                "dqn": DQN_SETTINGS, "payload_mode": PAYLOAD_MODE,
                "service_rate_units_per_ms": service_rate, "resource_unit": unit,
                "seeds": [42, 43, 44],
            }
            (RUN_ROOT / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8")
            measured_checkpoint_sha = (str(cost_frame.checkpoint_sha256.iloc[0])
                if PAYLOAD_MODE == "measured" else None)
            provenance = run_provenance(PROJECT_ROOT, MANIFEST_PATH,
                config=resolved_config, checkpoint_sha256=measured_checkpoint_sha)
            if provenance["git_commit"] != SOURCE_COMMIT or provenance["git_dirty"]:
                raise RuntimeError("Training checkout changed after the clean Git sync")
            models_dir = RUN_ROOT / "models"
            supervised = fit_supervised_models(train_jobs, models_dir, provenance=provenance)
            image_classifier = fit_image_priority_classifier(
                train_jobs, catalogue, models_dir, provenance=provenance)
            print({"feature_models": {name: str(path) for name, path in supervised.items()},
                   "image_classifier": {name: str(path) for name, path in image_classifier.items()}})
        '''),
        md('''
            ## 8. Train three small standard DQNs (two-hour training cap)

            Each action selects one eligible whole latent. Replay uses a target
            network, masked Bellman targets, Huber loss and 559→64→32→25 layers.
            Validation uses fixed streams and history-aware semantic-utility AUC.
            Each seed has a 40-minute wall-clock cap. All three seeds together
            therefore have a two-hour training cap; extraction and evaluation
            run outside that training budget.
        '''),
        code(r'''
            dqn_config = DQNConfig(**DQN_SETTINGS)
            dqn_manifests = train_dqn_seeds(
                suites["train_by_seed"], suites["validation"], budget, models_dir,
                config=dqn_config, provenance=provenance, device=DEVICE,
                enable_delivered_history=True)
            training_rows = []
            for seed in (42, 43, 44):
                path = models_dir / f"dqn_seed{seed}_history.json"
                training_rows.extend(json.loads(path.read_text(encoding="utf-8")))
            training_frame = pd.DataFrame(training_rows)
            training_frame.to_csv(RUN_ROOT / "training_metrics.csv", index=False)
            display(training_frame.groupby("seed").tail(1))
        '''),
        md('## 9. Diagnostics and durable artifact manifest'),
        code(r'''
            plots = RUN_ROOT / "plots"
            plots.mkdir(exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for seed, group in training_frame.groupby("seed"):
                axes[0].plot(group.episode, group.reward, label=str(seed), alpha=.8)
                axes[1].plot(group.episode, group.loss, label=str(seed), alpha=.8)
                observed = group.dropna(subset=["validation_semantic_utility_auc"])
                axes[2].plot(observed.episode, observed.validation_semantic_utility_auc, marker="o", label=str(seed))
            for axis, title in zip(axes, ("Episode reward", "Huber loss", "Validation semantic-utility AUC")):
                axis.set_title(title); axis.set_xlabel("Episode"); axis.grid(alpha=.2)
            axes[0].legend(title="Seed")
            fig.tight_layout(); fig.savefig(plots / "training_diagnostics.png", dpi=170)
            display(fig); plt.close(fig)
            action_counts = np.sum([json.loads((models_dir / f"dqn_seed{seed}_actions.json")
                                    .read_text(encoding="utf-8"))["action_index_counts"]
                                    for seed in (42, 43, 44)], axis=0)
            fig, axis = plt.subplots(figsize=(10, 3))
            axis.bar(np.arange(25), action_counts, color="#13678A")
            axis.set(xlabel="Eligible candidate index at decision", ylabel="Selections",
                     title="DQN action-index distribution across three seeds")
            fig.tight_layout(); fig.savefig(plots / "action_distribution.png", dpi=170)
            display(fig); plt.close(fig)

            import shutil
            shutil.copy2(MANIFEST_PATH, RUN_ROOT / "dataset_manifest.csv")
            shutil.copy2(PAYLOAD_COSTS, RUN_ROOT / "payload_costs.csv")
            shutil.copy2(cache["features"], RUN_ROOT / "feature_cache.jsonl")
            shutil.copy2(cache["predictions"], RUN_ROOT / "ladi_predictions.json")
            def file_hash(path):
                return sha256(Path(path).read_bytes()).hexdigest()
            required = ["resolved_config.yaml", "dataset_manifest.csv", "payload_costs.csv", "feature_cache.jsonl",
                        "ladi_predictions.json", "models/ordinal_classifier.npz",
                        "models/pairwise_ranker.npz", "models/clip_priority_classifier.npz",
                        "models/clip_priority_predictions.json", "models/clip_priority_manifest.json"] + [f"models/dqn_seed{seed}_best.npz" for seed in (42, 43, 44)]
            artifact_hashes = {name: file_hash(RUN_ROOT / name) for name in required}
            run_manifest = {
                **provenance, "run_id": RUN_ID, "status": "trained_on_prototype_fixed_manifest",
                "payload_mode": PAYLOAD_MODE, "resource_unit": unit,
                "service_rate_units_per_ms": service_rate,
                "artifacts_sha256": artifact_hashes,
                "dqn_best_validation": {str(seed): data["best_validation_semantic_utility_auc"]
                                         for seed, data in dqn_manifests.items()},
            }
            (RUN_ROOT / "run_manifest.json").write_text(
                json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print("Persistent run:", RUN_ROOT)
            print("Kaggle CLI: kaggle kernels output <username>/semantic-scheduler-train -p downloaded/train-run")
        '''),
    ]
    # Reuse the complete evaluation workflow immediately after training. The
    # separate evaluation notebook remains useful for later, exact-commit reruns.
    cells.extend([
        md('''
            ## 10. Evaluate every policy in this training run

            The remaining cells run the same comparison and faculty walkthrough
            as notebook 2 against the artifacts just produced. They save results
            under this run directory; no second Kaggle job is needed.
        '''),
        code(r'''
            from dataclasses import asdict, replace
            from INFRA.experiments.semantic_scheduler_experiment import (
                load_catalog, build_suites_from_manifest, evaluate_policy_suite)
            from INFRA.semantic_streams import (
                with_task_coupled_deadlines, with_transport_probability)
            from Scheduler.semantic_policies import NumpyDQNPolicy
            from INFRA.event_simulation import SemanticSchedulingEnvironment
            from SemParam.semantic_header import SemanticSchedulingHeaderV2Codec

            TRAIN_ROOT = RUN_ROOT
            EVAL_ID = RUN_ID + "-evaluation"
            EVAL_ROOT = RUN_ROOT / "evaluation"
            EVAL_ROOT.mkdir(parents=True, exist_ok=True)
            training = run_manifest
            print("Evaluation in the same run:", EVAL_ROOT)
        '''),
    ])
    evaluation_cells = evaluation_notebook().cells
    evaluation_start = next(index for index, cell in enumerate(evaluation_cells)
                            if cell.cell_type == "markdown" and
                            cell.source.startswith("## 3. Rebuild fixed streams"))
    cells.extend(evaluation_cells[evaluation_start:])
    return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})


def evaluation_notebook():
    cells = [
        md('''
            # Compare every whole-latent scheduling policy

            This notebook is independent of in-memory training cells. It clones
            the **latest** remote branch, verifies the training artifact's exact
            Git commit, manifest hash, schema and file hashes, then evaluates
            policies on identical immutable dynamic streams. The 60-job faculty
            scenario includes all curated images and a visible decision.

            Results are simulator results unless the final checkpoint replay
            succeeds. The header is a logical 24-byte control sidecar, separate
            from SwinJSCC complex channel uses.
        '''),
        md('## 1. Clean reset and exact source revision'), BOOTSTRAP,
        md('## 2. Dependencies, hardware, and persistent output'), DEPENDENCIES,
        code(r'''
            from datetime import datetime, timezone
            from hashlib import sha256
            from dataclasses import asdict, replace
            import csv
            import yaml
            from INFRA.ladi_manifest import validate_ladi60_manifest, render_contact_sheet
            from INFRA.kaggle_dataset_bundle import resolve_ladi_manifest
            from INFRA.semantic_config import load_scheduler_config, load_ue_profiles
            from INFRA.experiments.semantic_scheduler_experiment import (
                load_catalog, build_suites_from_manifest, evaluate_policy_suite, git_provenance)
            from INFRA.semantic_contracts import ResourceBudgetSemantic
            from INFRA.semantic_streams import (
                with_task_coupled_deadlines, with_transport_probability)
            from Scheduler.policy_registry import create_semantic_policy
            from Scheduler.semantic_policies import NumpyDQNPolicy
            from INFRA.event_simulation import SemanticSchedulingEnvironment, run_episode
            from SemParam.semantic_header import SemanticSchedulingHeaderV2, SemanticSchedulingHeaderV2Codec

            RESULTS_ROOT = Path(os.environ.get("SCHEDULER_RESULTS_DIR",
                "/kaggle/working/outputs" if IS_KAGGLE else WORKSPACE / "semantic-scheduler-results"))
            EVAL_ID = datetime.now(timezone.utc).strftime("evaluation-%Y%m%dT%H%M%SZ") + "-" + SOURCE_COMMIT[:8]
            EVAL_ROOT = RESULTS_ROOT / EVAL_ID
            EVAL_ROOT.mkdir(parents=True, exist_ok=False)
            TRAIN_ROOT = Path(os.environ.get("SCHEDULER_TRAIN_RUN_DIR",
                "/kaggle/input/semantic-scheduler-train-output"))
            MANIFEST_PATH = Path(os.environ.get("SCHEDULER_MANIFEST",
                "/kaggle/input/semantic-scheduler-ladi60-public/ladi60_semantic_scheduler_v1.csv"))
            MANIFEST_PATH = resolve_ladi_manifest(MANIFEST_PATH, EVAL_ROOT / "input_bundle")
            run_manifest_path = TRAIN_ROOT / "run_manifest.json"
            if not run_manifest_path.is_file() or not MANIFEST_PATH.is_file():
                raise FileNotFoundError("Attach the private training-output and prototype LADI datasets; set SCHEDULER_TRAIN_RUN_DIR and SCHEDULER_MANIFEST")
            training = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if training["git_commit"] != SOURCE_COMMIT:
                raise RuntimeError("Latest remote commit differs from trained model. Rerun notebook 1 on this commit.")
            current_source = git_provenance(PROJECT_ROOT)
            if (current_source["git_dirty"] or
                    current_source["source_digest_sha256"] != training["source_digest_sha256"]):
                raise RuntimeError("Evaluation source digest differs from the trained run")
            if training["manifest_sha256"] != sha256(MANIFEST_PATH.read_bytes()).hexdigest():
                raise RuntimeError("Prototype manifest differs from the training dataset")
            if (training["feature_schema_version"] != 2
                    or training["header_schema_id"] != 0x0201
                    or training["header_version"] != 2):
                raise RuntimeError("Training artifact schema is incompatible")
            for name, expected in training["artifacts_sha256"].items():
                path = TRAIN_ROOT / name
                if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Training artifact missing or changed: {name}")
            resolved_config = yaml.safe_load((TRAIN_ROOT / "resolved_config.yaml").read_text(encoding="utf-8"))
            expected_config_hash = sha256(json.dumps(resolved_config, sort_keys=True).encode()).hexdigest()
            if expected_config_hash != training["config_sha256"]:
                raise RuntimeError("Resolved training config changed")
            reviewed_rows = validate_ladi60_manifest(MANIFEST_PATH, require_reviewed=True)
            print({"evaluation_id": EVAL_ID, "exact_commit": SOURCE_COMMIT,
                   "training_run": training["run_id"], "payload_mode": training["payload_mode"]})
        '''),
        md('## 3. Rebuild fixed streams and load every policy from one registry'),
        code(r'''
            catalogue = load_catalog(MANIFEST_PATH, TRAIN_ROOT / "feature_cache.jsonl",
                                     TRAIN_ROOT / "payload_costs.csv")
            scheduler_settings = load_scheduler_config(PROJECT_ROOT / "INFRA/config/semantic_scheduler_demo.yaml")
            ue_profiles = load_ue_profiles(PROJECT_ROOT / "INFRA/config/ue_profiles.yaml")
            if resolved_config["scheduler"] != asdict(scheduler_settings):
                raise RuntimeError("Latest branch scheduler config differs from the training run")
            if resolved_config["ue_profiles"] != [asdict(p) for p in ue_profiles]:
                raise RuntimeError("Latest branch UE profiles differ from the training run")
            suites = build_suites_from_manifest(catalogue, MANIFEST_PATH,
                config=scheduler_settings, ue_profiles=ue_profiles)
            budget = ResourceBudgetSemantic(training["service_rate_units_per_ms"],
                                            resource_unit=training["resource_unit"])
            priority_by_image = {job.candidate.image_id: job.base_priority for job in catalogue}
            predictions = json.loads((TRAIN_ROOT / "ladi_predictions.json").read_text(encoding="utf-8"))
            image_priority_predictions = json.loads(
                (TRAIN_ROOT / "models/clip_priority_predictions.json").read_text(encoding="utf-8"))
            def model_path(filename):
                nested = TRAIN_ROOT / "models" / filename
                return nested if nested.is_file() else TRAIN_ROOT / filename
            artifacts = {
                "feature_ordinal_classifier": model_path("ordinal_classifier.npz"),
                "pairwise_linear_ranker": model_path("pairwise_ranker.npz"),
                "dqn": model_path(f"dqn_seed{max(training['dqn_best_validation'], key=training['dqn_best_validation'].get)}_best.npz"),
            }
            policy_names = (
                "fifo", "ue_round_robin", "fixed_priority", "edf", "priority_edf",
                "sjf", "aging", "semantic_greedy", "semantic_weighted",
                "semantic_lexicographic", "ladi_classifier_priority",
                "clip_image_priority", "clip_image_priority_edf",
                "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn",
                "privileged_marginal_utility")
            allowed_inputs = pd.DataFrame([
                {"policy": name, "observes_privileged_rubric_priority": name in
                 ("fixed_priority", "priority_edf", "privileged_marginal_utility"),
                 "observes_evaluator_evidence_group": name == "privileged_marginal_utility",
                 "observes_image_labels": False,
                 "uses_image_only_prediction": name in
                 ("ladi_classifier_priority", "clip_image_priority", "clip_image_priority_edf"),
                 "observes_semantic_features": name in
                 ("semantic_greedy", "semantic_weighted", "semantic_lexicographic",
                  "feature_ordinal_classifier", "pairwise_linear_ranker", "dqn")}
                for name in policy_names])
            display(allowed_inputs)
            print("The privileged one-step utility policy is a reference, not an upper bound on the optimal schedule.")
        '''),
        md('## 3a. Held-out image classifier quality'),
        code(r'''
            from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
            from Scheduler.semantic_policies import (
                ClipImagePriorityPolicy, LADIClassifierPriorityPolicy,
                OrdinalClassifierPolicy)

            split_by_image = {row["image_id"]: row["split"] for row in reviewed_rows}
            ladi_priority_model = LADIClassifierPriorityPolicy(predictions)
            clip_priority_model = ClipImagePriorityPolicy(image_priority_predictions)
            feature_priority_model = OrdinalClassifierPolicy(
                artifacts["feature_ordinal_classifier"])
            classifier_rows = []
            for item in catalogue:
                image_id = item.candidate.image_id
                if split_by_image[image_id] not in {"validation", "test"}:
                    continue
                classifier_rows.extend((
                    {"split": split_by_image[image_id], "model": model,
                     "image_id": image_id, "true_priority": item.base_priority,
                     "predicted_priority": predicted}
                    for model, predicted in (
                        ("LADI_image_classifier", ladi_priority_model.predict_priority(image_id)),
                        ("CLIP_image_priority_classifier",
                         clip_priority_model.predict_priority(image_id)),
                        ("feature_ordinal_oracle_assisted",
                         feature_priority_model.predict_priority(item.candidate)),
                    )
                ))
            classifier_predictions = pd.DataFrame(classifier_rows)
            classifier_predictions.to_csv(EVAL_ROOT / "priority_classifier_predictions.csv", index=False)
            classification_scores = []
            for (split, model), group in classifier_predictions.groupby(["split", "model"]):
                classification_scores.append({
                    "split": split, "model": model, "n_images": len(group),
                    "macro_f1": f1_score(group.true_priority, group.predicted_priority,
                                         labels=[0, 1, 2, 3], average="macro", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(
                        group.true_priority, group.predicted_priority),
                })
            classifier_metrics = pd.DataFrame(classification_scores)
            classifier_metrics.to_csv(EVAL_ROOT / "priority_classifier_metrics.csv", index=False)
            display(classifier_metrics)
            image_test = classifier_predictions[
                (classifier_predictions.split == "test") &
                (classifier_predictions.model == "CLIP_image_priority_classifier")]
            matrix = confusion_matrix(image_test.true_priority,
                                      image_test.predicted_priority,
                                      labels=[0, 1, 2, 3])
            fig, axis = plt.subplots(figsize=(5, 4))
            axis.imshow(matrix, cmap="Blues")
            for row in range(4):
                for col in range(4):
                    axis.text(col, row, int(matrix[row, col]), ha="center", va="center")
            axis.set(xticks=range(4), yticks=range(4), xlabel="Predicted priority",
                     ylabel="Rubric priority", title="CLIP image-only classifier: held-out 12 images")
            fig.tight_layout()
            fig.savefig(EVAL_ROOT / "priority_classifier_confusion.png", dpi=170)
            display(fig); plt.close(fig)
            print("The feature ordinal model uses rubric-provided P5/P6. LADI and CLIP image classifiers use image-derived predictions.")
        '''),
        md('''
            ## 4. Fair test tracks

            Each policy receives the same episode and pre-generated transport
            result for every job. Image-disjoint test streams and closed-catalogue
            sequence streams are reported separately. The latter does not test
            unseen-image generalization.
        '''),
        code(r'''
            all_metrics = []
            for track, episodes in (("image_disjoint_distinct", suites["image_disjoint_distinct"]),
                                    ("closed_catalogue", suites["closed_catalogue"]),
                                    ("image_disjoint_repeated_25", suites["image_disjoint"])):
                _, rows = evaluate_policy_suite(episodes, budget,
                    priority_by_image=priority_by_image, predictions_by_image=predictions,
                    image_priority_predictions_by_image=image_priority_predictions,
                    model_artifacts=artifacts, policy_names=policy_names)
                all_metrics.extend({"track": track, "transport_p": 1.0, **row} for row in rows)
            robustness = tuple(with_transport_probability(episode, .9)
                               for episode in suites["image_disjoint_distinct"])
            _, robust_rows = evaluate_policy_suite(robustness, budget,
                priority_by_image=priority_by_image, predictions_by_image=predictions,
                image_priority_predictions_by_image=image_priority_predictions,
                model_artifacts=artifacts, policy_names=policy_names)
            all_metrics.extend({"track": "image_disjoint_distinct_robustness", "transport_p": .9, **row}
                               for row in robust_rows)
            metrics = pd.DataFrame(all_metrics)
            # The primary metric is history-aware semantic utility AUC. It rewards
            # useful evidence delivered early and changes when successful delivery
            # updates P3. Priority-delivery AUC remains a secondary rubric metric.
            reference = (metrics[metrics.policy == "privileged_marginal_utility"]
                        [["track", "episode_id", "normalized_utility_auc"]]
                        .rename(columns={"normalized_utility_auc": "reference_auc"}))
            metrics = metrics.merge(reference, on=["track", "episode_id"], how="left",
                                    validate="many_to_one")
            metrics["relative_to_privileged_auc"] = np.where(
                metrics.reference_auc > 0,
                metrics.normalized_utility_auc / metrics.reference_auc,
                np.nan)
            metrics.to_csv(EVAL_ROOT / "all_policy_metrics.csv", index=False)
            display(metrics.groupby(["track", "policy"])[["normalized_utility_auc",
                "priority_delivery_auc",
                "relative_to_privileged_auc",
                "critical_on_time_rate", "deadline_miss_rate", "weighted_priority_inversion",
                "jain_fairness"]].mean().round(3))
        '''),
        md('''
            ## 5. Uncertainty, cost sensitivity, and input-masking sensitivity

            The P1/P2/P3/P4 row is the automatic semantic-feature comparator.
            Masking inputs to classifier, ranker, or DQN models trained with all
            five features is an out-of-distribution stress test. It is **not**
            evidence for how separately retrained automatic-only models perform.
        '''),
        code(r'''
            rng = np.random.default_rng(2026)
            def bootstrap_interval(values, repetitions=1000):
                clean = np.asarray(pd.Series(values).dropna(), dtype=float)
                if len(clean) == 0:
                    return (np.nan, np.nan, np.nan)
                means = np.mean(rng.choice(clean, size=(repetitions, len(clean)), replace=True), axis=1)
                return (float(clean.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)))
            ci_rows = []
            for (track, policy), group in metrics.groupby(["track", "policy"]):
                center, lower, upper = bootstrap_interval(group.normalized_utility_auc)
                ci_rows.append({"track": track, "policy": policy, "mean_auc": center,
                                "std_auc": float(group.normalized_utility_auc.std()),
                                "ci95_low": lower, "ci95_high": upper})
            ci = pd.DataFrame(ci_rows)
            ci.to_csv(EVAL_ROOT / "priority_delivery_bootstrap_ci.csv", index=False)
            display(ci[ci.track == "image_disjoint_distinct"].sort_values("mean_auc", ascending=False))
            paired = metrics[["track", "episode_id", "policy", "normalized_utility_auc"]].copy()
            for baseline in ("fifo", "edf"):
                baseline_rows = (paired[paired.policy == baseline]
                                 [["track", "episode_id", "normalized_utility_auc"]]
                                 .rename(columns={"normalized_utility_auc": f"{baseline}_auc"}))
                paired = paired.merge(baseline_rows, on=["track", "episode_id"],
                                      validate="many_to_one")
                paired[f"gain_vs_{baseline}"] = paired.normalized_utility_auc - paired[f"{baseline}_auc"]
            paired.to_csv(EVAL_ROOT / "dynamic_paired_priority_gains.csv", index=False)
            display(paired[paired.track == "image_disjoint_distinct"]
                    .groupby("policy")[["gain_vs_fifo", "gain_vs_edf"]].mean().round(3))

            from INFRA.semantic_contracts import FEATURE_NAMES
            ablations = {
                "full_oracle_assisted": FEATURE_NAMES,
                "automatic_p1_p2_p3_p4": ("scene_importance", "context_relevance",
                                            "delivered_history_novelty",
                                            "interpretation_confidence"),
                "without_knowledge_base_p3": tuple(name for name in FEATURE_NAMES
                                                     if name != "delivered_history_novelty"),
                "network_only": (),
            }
            ablation_rows = []
            for label, fields in ablations.items():
                _, rows = evaluate_policy_suite(suites["image_disjoint_distinct"][:30], budget,
                    priority_by_image=priority_by_image, predictions_by_image=predictions,
                    image_priority_predictions_by_image=image_priority_predictions,
                    model_artifacts=artifacts,
                    policy_names=("semantic_greedy", "feature_ordinal_classifier",
                                  "pairwise_linear_ranker", "dqn"),
                    visible_features=tuple(fields))
                ablation_rows.extend({"ablation": label, **row} for row in rows)
            ablation = pd.DataFrame(ablation_rows)
            ablation.to_csv(EVAL_ROOT / "feature_ablation_metrics.csv", index=False)
            display(ablation.groupby(["ablation", "policy"]).normalized_utility_auc.mean().unstack().round(3))
            print("Input masking leaves reward/deadlines fixed. Learned-policy rows are OOD stress tests, not retrained ablations.")
            sensitivity_rows = []
            for coefficient in (0.0, 0.1, 0.2):
                _, rows = evaluate_policy_suite(suites["image_disjoint_distinct"][:30], budget,
                    priority_by_image=priority_by_image, predictions_by_image=predictions,
                    image_priority_predictions_by_image=image_priority_predictions,
                    model_artifacts=artifacts,
                    policy_names=("fifo", "semantic_greedy", "dqn"),
                    cost_coefficient=coefficient)
                sensitivity_rows.extend({"cost_coefficient": coefficient, **row} for row in rows)
            sensitivity = pd.DataFrame(sensitivity_rows)
            sensitivity.to_csv(EVAL_ROOT / "fixed_policy_cost_sensitivity.csv", index=False)
            display(sensitivity.groupby(["cost_coefficient", "policy"])
                    [["episode_reward", "priority_delivery_auc"]].mean().round(3))
            print("This is fixed-policy reward sensitivity. Only the 0.1 coefficient trained the DQN; retrain to test policy adaptation.")

            coupled_episodes = tuple(with_task_coupled_deadlines(episode)
                                     for episode in suites["image_disjoint_distinct"][:30])
            _, coupled_rows = evaluate_policy_suite(coupled_episodes, budget,
                priority_by_image=priority_by_image, predictions_by_image=predictions,
                image_priority_predictions_by_image=image_priority_predictions,
                model_artifacts=artifacts,
                policy_names=("fifo", "edf", "clip_image_priority_edf",
                              "semantic_greedy", "dqn", "fixed_priority"))
            deadline_sensitivity = pd.DataFrame(coupled_rows)
            deadline_sensitivity.to_csv(EVAL_ROOT / "task_coupled_deadline_sensitivity.csv",
                                        index=False)
            display(deadline_sensitivity.groupby("policy")
                    [["priority_delivery_auc", "critical_on_time_rate"]].mean().round(3))
            print("This sensitivity uses the same images and arrivals but makes packet deadlines a proxy for semantic urgency. The main experiment keeps them independent.")
        '''),
        md('## 5b. Experiment A: semantic ordering when every image is ready'),
        code(r'''
            from INFRA.semantic_evaluator import static_ranking_metrics
            rank_rows = []
            for track, episodes in (("static_image_disjoint_12", suites["static_image_disjoint"]),
                                    ("static_closed_catalogue_25", suites["static_closed_catalogue"])):
                for episode in episodes:
                    static_results, _ = evaluate_policy_suite((episode,), budget,
                        priority_by_image=priority_by_image,
                        predictions_by_image=predictions,
                        image_priority_predictions_by_image=image_priority_predictions,
                        model_artifacts=artifacts,
                        policy_names=policy_names, retain_results=True,
                        enable_delivered_history=False)
                    rank_rows.extend({"track": track, "episode_id": episode.episode_id,
                                      "policy": result["policy"],
                                      **static_ranking_metrics(result)}
                                     for result in static_results)
            rank_frame = pd.DataFrame(rank_rows)
            rank_frame.to_csv(EVAL_ROOT / "static_ranking_metrics.csv", index=False)
            display(rank_frame.groupby(["track", "policy"])[
                ["ndcg_at_5", "critical_recall_at_5", "priority_pair_agreement"]]
                .mean().round(3))
            reference = (rank_frame[rank_frame.policy == "fifo"]
                         [["track", "episode_id", "ndcg_at_5"]]
                         .rename(columns={"ndcg_at_5": "fifo_ndcg_at_5"}))
            paired_rank = rank_frame.merge(reference, on=["track", "episode_id"],
                                           validate="many_to_one")
            paired_rank["ndcg_at_5_gain_vs_fifo"] = (
                paired_rank.ndcg_at_5 - paired_rank.fifo_ndcg_at_5)
            paired_rank.to_csv(EVAL_ROOT / "static_paired_rank_gains.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for axis, track in zip(axes, ("static_image_disjoint_12",
                                          "static_closed_catalogue_25")):
                rows = paired_rank[paired_rank.track == track]
                means = rows.groupby("policy").ndcg_at_5_gain_vs_fifo.mean()
                means.sort_values().plot.barh(ax=axis, color="#13678A")
                axis.set(title=track, xlabel="Paired nDCG@5 gain versus FIFO")
                axis.axvline(0, color="black", linewidth=1)
                axis.grid(axis="x", alpha=.2)
            fig.tight_layout()
            fig.savefig(EVAL_ROOT / "static_semantic_rank_gains.png", dpi=170)
            display(fig); plt.close(fig)
            print("All static jobs arrive at 0 ms; same-priority ties are excluded from pair agreement.")
        '''),
        md('''
            ## 5c. Evidence ladder: recognize importance, rank images, deliver early

            These are three different questions. The image-only classifier is
            evaluated on held-out images. Static nDCG@5 tests whether a policy
            orders simultaneously ready images by the fixed task rubric.
            Dynamic priority-delivery AUC tests whether priority survives UE
            arrivals and deadlines. FIFO and EDF are the predeclared network
            comparators. Rubric-assisted and privileged rows are labeled.
        '''),
        code(r'''
            focused_policies = (
                "fifo", "edf", "ladi_classifier_priority",
                "clip_image_priority", "clip_image_priority_edf",
                "semantic_greedy", "feature_ordinal_classifier",
                "pairwise_linear_ranker", "dqn", "fixed_priority",
            )
            classifier_focus = classifier_metrics[
                (classifier_metrics.split == "test") &
                (classifier_metrics.model.isin(("LADI_image_classifier",
                    "CLIP_image_priority_classifier", "feature_ordinal_oracle_assisted")))]
            static_focus = (rank_frame[rank_frame.track == "static_image_disjoint_12"]
                            .groupby("policy").ndcg_at_5.mean())
            dynamic_focus = (metrics[metrics.track == "image_disjoint_distinct"]
                             .groupby("policy").priority_delivery_auc.mean())
            ladder = pd.DataFrame({
                "policy": focused_policies,
                "static_ndcg_at_5": [static_focus.get(name, np.nan) for name in focused_policies],
                "dynamic_priority_delivery_auc": [dynamic_focus.get(name, np.nan)
                                                  for name in focused_policies],
                "input_type": ["network", "network", "image only", "image only",
                               "image plus deadline", "rubric assisted", "rubric assisted",
                               "rubric assisted", "rubric assisted", "privileged rubric"],
            })
            ladder.to_csv(EVAL_ROOT / "evidence_ladder.csv", index=False)
            display(classifier_focus[["model", "n_images", "macro_f1", "balanced_accuracy"]])
            display(ladder)
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
            palette = {"network": "#777777", "image only": "#13678A",
                       "image plus deadline": "#11A09A", "rubric assisted": "#D98432",
                       "privileged rubric": "#8040A0"}
            colors = [palette[kind] for kind in ladder.input_type]
            axes[0].barh(ladder.policy, ladder.static_ndcg_at_5, color=colors)
            axes[0].set(xlabel="nDCG@5 on 12 held-out images", xlim=(0, 1),
                        title="A. All images ready together")
            axes[1].barh(ladder.policy, ladder.dynamic_priority_delivery_auc,
                         color=colors)
            axes[1].set(xlabel="Priority-delivery AUC on unseen images", xlim=(0, 1),
                        title="B. Five UE queues and deadlines")
            for axis in axes:
                axis.invert_yaxis()
                axis.grid(axis="x", alpha=.2)
            fig.tight_layout()
            fig.savefig(EVAL_ROOT / "semantic_evidence_ladder.png", dpi=180)
            display(fig); plt.close(fig)
            print("Bars are measured simulator outcomes on the fixed prototype rubric; compare input types before drawing claims.")
        '''),
        md('## 6. Plots for the faculty comparison'),
        code(r'''
            plot_dir = EVAL_ROOT / "plots"; plot_dir.mkdir(exist_ok=True)
            subset = metrics[metrics.track == "image_disjoint_distinct"]
            means = subset.groupby("policy").mean(numeric_only=True)
            order = means.normalized_utility_auc.sort_values(ascending=False).index
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            means.loc[order, "normalized_utility_auc"].plot.bar(ax=axes[0,0], color="#13678A")
            axes[0,0].set_title("History-aware semantic utility over time (AUC)")
            means.loc[order, "critical_on_time_rate"].plot.bar(ax=axes[0,1], color="#11A09A")
            axes[0,1].set_title("Critical/emergency delivered on time")
            means.loc[order, "weighted_priority_inversion"].plot.bar(ax=axes[1,0], color="#D98432")
            axes[1,0].set_title("Eligible priority inversion severity")
            axes[1,1].scatter(means.mean_policy_inference_ms, means.normalized_utility_auc)
            for name in means.index:
                axes[1,1].annotate(name, (means.loc[name,"mean_policy_inference_ms"],
                                           means.loc[name,"normalized_utility_auc"]), fontsize=8)
            axes[1,1].set(xlabel="Policy inference (ms)", ylabel="Semantic utility AUC",
                           title="Inference cost versus priority delivery")
            for axis in axes.flat: axis.grid(alpha=.15)
            fig.tight_layout(); fig.savefig(plot_dir / "policy_comparison.png", dpi=170)
            display(fig); plt.close(fig)
        '''),
        md('''
            ## 7. Faculty scenario: 60 images, five UEs, one visible decision

            The table exposes the complete queue snapshot at a decision point:
            seven values, availability masks, payload units, deadlines, score and
            selected action. The backward-compatible 24-byte V2 header carries
            P2/P3/P5/P6/P7; P1/P4 remain in the logical scheduling envelope.
        '''),
        code(r'''
            demo = suites["faculty_demo"][0]
            _, demo_metrics = evaluate_policy_suite((demo,), budget,
                priority_by_image=priority_by_image, predictions_by_image=predictions,
                image_priority_predictions_by_image=image_priority_predictions,
                model_artifacts=artifacts, policy_names=policy_names)
            demo_results, _ = evaluate_policy_suite((demo,), budget,
                priority_by_image=priority_by_image, predictions_by_image=predictions,
                image_priority_predictions_by_image=image_priority_predictions,
                model_artifacts=artifacts, policy_names=policy_names, retain_results=True)
            display(pd.DataFrame(demo_metrics).sort_values("normalized_utility_auc", ascending=False))
            image_paths = {row["image_id"]: row["image_path"] for row in reviewed_rows}
            from PIL import Image
            fig, axes = plt.subplots(6, 10, figsize=(22, 14))
            for axis, job in zip(axes.flat, demo.jobs):
                with Image.open(image_paths[job.candidate.image_id]) as opened:
                    axis.imshow(opened.convert("RGB"))
                axis.set_title(f"{job.candidate.ue_id} | {job.candidate.arrival_ms:.0f} ms", fontsize=8)
                axis.axis("off")
            fig.suptitle("All 60 distinct LADI images assigned to five simulated UEs")
            fig.tight_layout(); fig.savefig(plot_dir / "faculty_60_inputs.png", dpi=130)
            display(fig); plt.close(fig)
            chosen_demo = next(result for result in demo_results if result["policy"] == "dqn")
            first = next(row for row in chosen_demo["decision_trace"] if len(row["candidate_snapshot"]) > 1)
            candidates = pd.DataFrame(first["candidate_snapshot"])
            feature_names = ("P1 scene importance", "P2 relevance", "P3 novelty",
                             "P4 interpretation confidence", "P5 impact",
                             "P6 urgency", "P7 sufficiency")
            for index, name in enumerate(feature_names):
                candidates[name] = candidates.features.map(lambda values: values[index])
                candidates[name + " available"] = candidates.availability.map(lambda mask: mask[index])
            display(candidates.drop(columns=["features", "raw_features", "availability"]))
            print("DQN selected:", first["job_id"], "at", first["time_ms"], "ms; reward:", first["reward"])
            wire = bytes.fromhex(first["header_hex"])
            decoded_header = SemanticSchedulingHeaderV2Codec().decode(wire)
            assert len(wire) == first["header_size_bytes"] == 24
            print("Logical control header:", len(wire), "bytes /", len(wire)*8,
                  "bits; hex:", wire.hex())
            print("Decoded decision-time fields:", decoded_header)

            dqn = NumpyDQNPolicy(artifacts["dqn"])
            demo_environment = SemanticSchedulingEnvironment(
                demo, budget, enable_delivered_history=True)
            for recorded_decision in chosen_demo["decision_trace"]:
                snapshot = demo_environment.snapshot()
                if recorded_decision is first:
                    break
                demo_environment.step(recorded_decision["action_index"])
            assert snapshot.now_ms == first["time_ms"]
            q_values, valid_mask = dqn.q_values(snapshot, budget)
            assert int(np.argmax(np.where(valid_mask, q_values, -np.inf))) == first["action_index"]
            display(pd.DataFrame({"job_id": [job.job_id for job in snapshot.jobs],
                "eligible": valid_mask[:len(snapshot.jobs)],
                "Q_value": q_values[:len(snapshot.jobs)]}).sort_values("Q_value", ascending=False))
        '''),
        md('## 8. Delivery timeline and selected real checkpoint replay'),
        code(r'''
            demo_trace = []
            for result in demo_results:
                for row in result["outcomes"]:
                    demo_trace.append({"policy": result["policy"], **row})
            timeline = pd.DataFrame(demo_trace)
            timeline.to_csv(EVAL_ROOT / "faculty_timeline.csv", index=False)
            display(timeline[timeline.policy == "dqn"].sort_values("started_ms")[
                ["image_id", "ue_id", "arrival_ms", "started_ms", "completed_ms",
                 "status", "utility_delta"]])
            novelty_rows = []
            for decision in chosen_demo["decision_trace"]:
                for candidate in decision["candidate_snapshot"]:
                    novelty_rows.append({"time_ms": decision["time_ms"],
                        "history_version": decision["history_version"],
                        "image_id": candidate["image_id"],
                        "P3_novelty": candidate["raw_features"][2],
                        "P3_available": candidate["availability"][2]})
            novelty_frame = pd.DataFrame(novelty_rows)
            display(novelty_frame[novelty_frame.P3_available]
                    .sort_values(["history_version", "P3_novelty"])
                    .head(20))
            print("The table demonstrates that P3 is recomputed from receiver knowledge after successful deliveries.")
            fig, axis = plt.subplots(figsize=(11,5))
            for result in demo_results:
                times, values = zip(*result["utility_curve"])
                axis.step(times, values, where="post", label=result["policy"], alpha=.8)
            axis.set(xlabel="Simulated time (ms)", ylabel="Accumulated semantic utility",
                     title="Identical faculty stream, different scheduling decisions")
            axis.grid(alpha=.2); axis.legend(ncol=3, fontsize=8)
            fig.tight_layout(); fig.savefig(plot_dir / "faculty_utility_timeline.png", dpi=170)
            display(fig); plt.close(fig)

            from EncDec.integration.scheduler_replay import replay_selected_jobs
            replay_status = {"status": "unavailable", "reason": "payload mode is simulated"}
            if training["payload_mode"] == "measured":
                model_dir = Path(os.environ.get("SCHEDULER_SWIN_MODEL_DIR", PROJECT_ROOT / "models/SwinJSCC"))
                selected_ids = [row["image_id"] for row in chosen_demo["outcomes"]
                                if row["status"] in ("completed_on_time", "completed_late")][:3]
                try:
                    attached_cache = TRAIN_ROOT / "payload" / "payload_costs.csv"
                    if not attached_cache.is_file():
                        from EncDec.integration.scheduler_replay import cache_swinjscc_payloads
                        selected_rows = [row for row in reviewed_rows if row["image_id"] in selected_ids]
                        attached_cache = cache_swinjscc_payloads(selected_rows,
                            EVAL_ROOT / "selected_payload_cache", model_dir=model_dir,
                            upstream_root=PROJECT_ROOT / "external/SwinJSCC", device=DEVICE)
                    replay_path = replay_selected_jobs(selected_ids, image_paths,
                        attached_cache, EVAL_ROOT / "replay",
                        model_dir=model_dir, upstream_root=PROJECT_ROOT / "external/SwinJSCC",
                        device=DEVICE)
                    replay_status = {"status": "measured_checkpoint_inference", "csv": str(replay_path)}
                    display(pd.read_csv(replay_path))
                except (FileNotFoundError, RuntimeError, ValueError) as error:
                    replay_status = {"status": "unavailable", "reason": str(error)}
            (EVAL_ROOT / "replay_status.json").write_text(json.dumps(replay_status, indent=2), encoding="utf-8")
            print("Codec replay:", replay_status)
        '''),
        md('## 9. Claim ledger and output persistence'),
        code(r'''
            claims = pd.DataFrame([
                {"category":"Implemented", "component":"P1-P7 dynamic features, successful-delivery knowledge base, policies, DQN, metrics"},
                {"category":"Simulated", "component":"UE arrivals, queues, deadlines, service duration, transport outcome"},
                {"category":"Measured when available", "component":"SwinJSCC checkpoint latency, channel uses, selected PSNR/SSIM"},
                {"category":"Proposed", "component":"Real 5G MAC/PHY control channel, RB allocation, distributed KB sync"},
            ])
            display(claims)
            claims.to_csv(EVAL_ROOT / "claim_ledger.csv", index=False)
            summary = {"evaluation_id": EVAL_ID, "exact_commit": SOURCE_COMMIT,
                       "training_run_id": training["run_id"], "payload_mode": training["payload_mode"],
                       "image_disjoint_streams": len(suites["image_disjoint"]),
                       "closed_catalogue_streams": len(suites["closed_catalogue"]),
                       "replay": replay_status}
            (EVAL_ROOT / "evaluation_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("Persistent evaluation:", EVAL_ROOT)
            print("Kaggle CLI: kaggle kernels output <username>/semantic-scheduler-evaluate -p downloaded/evaluation-run")
        '''),
    ]
    dynamic_only = []
    skip_code_after_heading = False
    for cell in cells:
        if (cell.cell_type == "markdown" and
                (cell.source.startswith("## 5b. Experiment A")
                 or cell.source.startswith("## 5c. Evidence ladder"))):
            skip_code_after_heading = True
            continue
        if skip_code_after_heading and cell.cell_type == "code":
            skip_code_after_heading = False
            continue
        dynamic_only.append(cell)
    return nbformat.v4.new_notebook(cells=dynamic_only, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, notebook in (
        ("01_train_semantic_scheduler.ipynb", train_notebook()),
        ("02_evaluate_all_policies.ipynb", evaluation_notebook()),
    ):
        nbformat.validate(notebook)
        path = OUT / filename
        nbformat.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
