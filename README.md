# SemanticSchedulerEnd2End

This repository is the single source of truth for the Stage 1A semantic-communication experiment. The only active Stage 1A notebook is [01_stage1a_complete_kaggle.ipynb](notebooks/Stage1A/01_stage1a_complete_kaggle.ipynb). It clones this repository at a pinned revision and invokes the repository code; the notebook does not duplicate a second implementation.

## Stage 1A boundary

`INFRA` owns interfaces, typed artifacts, registries, orchestration and runtime state only. Encoders, codebooks, channels, classical baselines and evaluation live in their independent top-level packages. `TransformerPipeline/TransformerParam` and `TransformerPipeline/KnowledgeBases` are siblings: current-input evidence and historical/context evidence remain separate.

Operational Stage 1A components are SwinJSCC, AWGN/Rayleigh channel primitives, K=256 MiniBatchKMeans VQ, JPEG plus optional Sionna 5G-LDPC/QPSK, and image/compression/communication/report evaluation. Semantic extraction, knowledge stores, semantic/RL schedulers, NS-3, Sionna system-level and O-RAN adapters fail explicitly. They do not return fake output.

## Reproducible local checks

Use a dedicated Python 3.12 environment. The workstation's Python 3.14 runtime is not a supported model runtime.

```powershell
cd D:\Bunker\BaseCamp\WirelessCommunication\SemanticComms\SemanticSchedulerEnd2End
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r environment\requirements-local-inference.lock
pip install -e .[metrics,test]
python scripts\bootstrap_swinjscc.py
python -m pytest Test\tests
python E2EMain.py --config config\stage1a.yaml --action validate
```

`bootstrap_swinjscc.py` checks out exactly `a6d0e6da53548976acbe9317839a077ef31f190f` and applies a narrow, version-checked correction: complex real/imaginary halves are paired inside each image rather than across a flattened minibatch. This is required for valid multi-sample `DataParallel` shards.

## Artifact policy

`models/` contains only trained runtime assets and is not source code. A completed Stage 1A run exports explicit tensor-state artifacts:

```text
models/SwinJSCC/stage1_swinjscc_full.pt
models/SwinJSCC/stage1_swinjscc_encoder.pt
models/SwinJSCC/stage1_swinjscc_decoder.pt
models/SwinJSCC/stage1_manifest.json
```

The manifest checks the architecture, source commit and patched-source digest, AWGN/MSE lock, SNR/rate grid, Python/PyTorch major-minor runtime, data provenance, and hashes of all three tensor-state files before inference. `results/` holds run outputs only.

## Research interpretation

This is a corrected modern PyTorch port, **not** a claim of paper-exact replication. Native SwinJSCC is measured in complex channel uses/CBR. Float latent storage is a diagnostic only and is never presented as over-air bits. The codebook packet includes fixed-width indices, a full per-image rate-mask cost, metadata and CRC. JPEG frames record LDPC/QPSK channel uses and CRC failure independently from image quality.

## Kaggle

Set `PROJECT_GIT_URL` in the active notebook after this local Git repository is pushed to a remote, retain its `PROJECT_REF`, and enable Kaggle Internet for the initial clone. The notebook verifies Kaggle's CUDA-enabled PyTorch environment; it never replaces the supplied torch wheel. It hard-fails if DIV2K/Kodak paths are absent and never trains on synthetic fallback images.
