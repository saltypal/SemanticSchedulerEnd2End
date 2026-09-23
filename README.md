# SemanticSchedulerEnd2End

This repository is the team workspace for Stage 1A communication, Stage 1B semantic headers, and the Stage 1C parameter-codebook demonstration. Stage 1A remains isolated in [01_stage1a_complete_kaggle.ipynb](Notebooks/Stage1A/01_stage1a_complete_kaggle.ipynb); Stage 1B/1C notebooks import repository modules rather than duplicating their implementation. See [ORGANIZATION.md](ORGANIZATION.md) for the current file layout.

## Stage 1A boundary

`INFRA` owns interfaces, typed artifacts, registries, orchestration and runtime state only. Encoders, codebooks, channels, classical baselines and evaluation live in their independent top-level packages. `TransformerPipeline/TransformerParam` and `TransformerPipeline/KnowledgeBases` are siblings: current-input evidence and historical/context evidence remain separate.

Operational Stage 1A components are SwinJSCC, AWGN/Rayleigh channel primitives, K=256 MiniBatchKMeans VQ, JPEG plus optional Sionna 5G-LDPC/QPSK, and image/compression/communication/report evaluation. Stage 1B adds a CLIP-backed image semantic extractor, a local NetworkX knowledge-base baseline, and a compact logical semantic-header sidecar. Semantic/RL scheduling, NS-3, Sionna system-level, O-RAN generation, Redis, vector-store, global-memory, and richer-KG adapters remain explicit unavailable placeholders.

## Reproducible local checks

Use a dedicated Python 3.12 environment. The workstation's Python 3.14 runtime is not a supported model runtime.

```powershell
cd path\to\image_semcom\SemanticSchedulerEnd2End_team
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r environment\requirements-local-inference.lock
pip install -e .[benchmark,metrics,test,stage1b,notebook]
python scripts\bootstrap_swinjscc.py
python -m pytest Test\tests
python E2EMain.py --config config\stage1a.yaml --action validate
python E2EMain.py --config config\stage1b.yaml --action semantic-validate
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

To reproduce the repaired DeepJSCC/NTSCC notebook methods on Windows, run from
this repository, with `SEMCOM_EXPERIMENT_ROOT` pointing here. The two validated
`.pth` checkpoints stay in the sibling `Deep-JSCC-PyTorch` and `NTSCC_JSAC22`
folders. The universal notebook then adds native DeepJSCC, native NTSCC,
DeepJSCC K=256 codebook, and rate-conditioned NTSCC PQ4 K=256 automatically.
The NTSCC codebook excludes every Kodak calibration record for an unbiased
24-image test. Its width-2 fallback codebook is derived only from active values
in those same CIFAR-10 training latents. The repaired one-image notebook is
`Notebooks/DeepJSCC_NTSCC_Final_Repaired.ipynb`. From this repository, the reproducible
local command is:

```powershell
.\scripts\run_repaired_kodak24_notebook.ps1
```

The runner uses one deterministic channel seed per image, one timed pass, all
24 images, LPIPS, and atomic CSV append. Override `-ExperimentRoot`,
`-Contributor`, or `-RunId` when needed.
## Stage 1B semantic-header prototype

Stage 1B uses exactly seven configurable fields: `AoI_seconds`, `freshness`, `novelty`, `relevance`, `reliability`, `urgency`, and `task_importance`. Every field is a `ParameterValue` with value, raw evidence, availability, method, source, version, producer level, and provenance. Missing evidence stays unavailable; it is never replaced with a random or zero score.

The Transformer path runs CLIP once per image, caches task/concept text embeddings, queries the knowledge base before recording the image, and emits a `SemanticParameterBundle`. `reliability` is specifically a visual semantic-confidence proxy, not an independently verified source-trust score. `urgency` is derived only from a deadline/context and is unavailable when no deadline is supplied.

The compact semantic header is a versioned TLV sidecar paired with, but not injected into, Stage 1A's native latent or communication VQ packet. It must therefore be measured separately from existing Stage 1A VQ bit accounting.

To run the fixed demonstration sequence in `Test/images`, first supply a manifest-validated Stage 1A model directory and an optional pre-trained Stage 1A communication-VQ artifact:

```powershell
python E2EMain.py --config config\stage1b.yaml --action semantic-run --param-extractor transformer --model-dir models\SwinJSCC --communication-codebook models\Codebooks\swinjscc_tx_k256.joblib --output-plot results\stage1b_transformer_showcase.png
```

The CLI prints `SEMANTIC HEADER`, then `NATIVE LATENT PAYLOAD`, then `COMMUNICATION VQ PAYLOAD` for each image. Omit `--communication-codebook` only when the existing Stage 1A VQ artifact is unavailable; the result explicitly labels that payload as unavailable.

## Stage 1C demonstration codebook

`CodebookParam` is a semantic-parameter codebook and is completely separate from `EncDecPipeline/Codebooks`. Offline training pairs a pre-modulation SwinJSCC representation with CLIP teacher targets. At runtime it consumes only pooled Swin semantic features plus metadata/knowledge-base evidence; it never invokes CLIP.

The local codebook training corpus must contain at least ten distinct images outside `Test/images`. The training notebook hard-fails on SHA-256 overlap and saves an artifact marked `demo_only=true`; it is not generalization evidence.

```powershell
python scripts\build_stage1b_notebooks.py
jupyter lab Notebooks\Stage1C\01_train_parameter_codebook_local.ipynb
jupyter lab Notebooks\Stage1B\02_transformer_vs_codebook_semantic_header.ipynb
```

After the demo artifact exists, select the same downstream bundle API through:

```powershell
python E2EMain.py --config config\stage1b.yaml --action semantic-run --param-extractor codebook --model-dir models\SwinJSCC --semantic-codebook models\SemanticCodebooks\emergency_building_fire_demo.npz
```

The codebook rejects a missing or mismatched task-profile artifact rather than silently invoking CLIP or fabricating values. No scheduler behavior is implemented in this stage.
