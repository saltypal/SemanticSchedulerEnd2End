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
pip install -e .[benchmark,metrics,test]
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

## Shared Kodak-24 benchmark

`Notebooks/Kodak24_All_Model_Benchmark.ipynb` is the model-agnostic local
evaluation entry point. It evaluates all 24 native-resolution images, records
encoder/decoder time, memory and model complexity, PSNR/SSIM/LPIPS, exact source
payload compression, and complex-channel-use CBR where the method defines it.
JPEG quality points and lossless RAW RGB24 are built in. See
`docs/KODAK24_BENCHMARK.md` for the metric and team CSV contract.

To reproduce the repaired DeepJSCC/NTSCC notebook methods on Windows, point
`SEMCOM_EXPERIMENT_ROOT` at the local experiment directory that contains its
`.venv`, `semcom_benchmark`, official source trees, checkpoints, and calibration
cache. The universal notebook then adds native DeepJSCC, native NTSCC,
DeepJSCC K=256 codebook, and rate-conditioned NTSCC PQ4 K=256 automatically.
The NTSCC codebook excludes every Kodak calibration record for an unbiased
24-image test. Its width-2 fallback codebook is derived only from active values
in those same CIFAR-10 training latents. From this repository, the reproducible
local command is:

```powershell
.\scripts\run_repaired_kodak24_notebook.ps1
```

The runner uses one deterministic channel seed per image, one timed pass, all
24 images, LPIPS, and atomic CSV append. Override `-ExperimentRoot`,
`-Contributor`, or `-RunId` when needed.
