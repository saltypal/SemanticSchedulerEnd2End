# Team consolidation inventory (commit `1e44baa`)

This document records the teammate's layout at the consolidation commit. The
current modular architecture and commands are in `PROJECT_MAP.md` and `README.md`.
The research benchmark implementation now lives in `EncDec/Research/`;
`semcom_benchmark/` preserves the original notebook imports. The notebook list
below is historical research material, while the numbered notebooks directly
inside each module are the active modular guides.

At the consolidation commit, the team checkout was `image_semcom/SemanticSchedulerEnd2End_team`; `fc24320` was its base commit. This repository can be checked out at a different path.

## Notebooks

- `Notebooks/Kodak24_All_Model_Benchmark.ipynb`: current Kodak-24 all-model evaluation.
- `Notebooks/DeepJSCC_NTSCC_Final_Repaired.ipynb`: latest executed DeepJSCC/NTSCC research benchmark.
- `Notebooks/Stage1A/01_stage1a_complete_kaggle.ipynb`: active Stage 1A training workflow.
- `Notebooks/Stage1B/02_transformer_vs_codebook_semantic_header.ipynb`: Stage 1B showcase.
- `Notebooks/Stage1C/01_train_parameter_codebook_local.ipynb`: Stage 1C demonstration.

Other notebooks already tracked in the repository are retained as historical work. The list above identifies the current entry points.

## Models and checkpoints

- `models/DeepJSCC/cifar10/`: one `epoch_999.pkl` per distinct training configuration, with matching YAML in `models/DeepJSCC/configs/`.
- `models/NTSCC/calibration/`: the rate-conditioned calibration cache and provenance.
- `models/ADJSCC/`: the available ADJSCC checkpoint, which still needs a compatible architecture before evaluation.
- `models/SwinJSCC/`, `models/SemanticCodebooks/`, and `models/Codebooks/`: existing team model outputs and placeholders.

The two validated production `.pth` files remain at their original paths: `../Deep-JSCC-PyTorch/out/imagenet_10_0.33_200.00_32_19.pth` and `../NTSCC_JSAC22/checkpoints/ntscc_hyperprior_quality_1_psnr.pth`. The repository contains no `.pth` files. Their hashes are recorded in the model README files. `models/MANIFEST.sha256` covers the model files held inside this repository.

The `.pt` and `.pkl` files are stored through Git LFS; teammates need Git LFS to hydrate them after cloning. The checkpoint-compatible research source trees remain under `references/`. Canonical benchmark code is under `EncDec/Research/`, with `semcom_benchmark/` as a compatibility package.

## Code and results

The former stage-based implementation remains reachable through `src/` compatibility imports. Root `config.yaml` and the EncDec/SemParam/Scheduler/INFRA modules own current execution. `data/Kodak24/` is the benchmark dataset; `results/` holds generated outputs.
