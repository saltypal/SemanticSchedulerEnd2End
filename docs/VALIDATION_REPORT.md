# Full rebuild: implementation and validation report

Repository: `D:\Bunker\BaseCamp\WirelessCommunication\SemanticComms\SemanticSchedulerEnd2End`
Base Git commit: `fc243207bbc08934628eccb31e2c6168b878522c`
This report records the original modular rebuild before the later team-main merge.
For the current integration, see `docs/MAIN_MERGE_1E44BAA.md` and the post-merge
checks below. Git is the source of truth for the current branch and remote state.

## Delivered

| Requirement | Implementation / acceptance |
|---|---|
| Protect existing work | Source ZIP, binary patches, original status and asset hash inventory saved before migration. **80 pre-existing data/model/result files match their original SHA256 hashes.** User's untracked graph script is unchanged. |
| Module ownership | 150 reusable source files migrated to canonical EncDec, SemParam, Scheduler and INFRA packages. `src/` retains compatibility wrappers; stage notebooks preserved as historical copies. |
| EncDec | Shared BenchmarkCodec contract, capability catalog, configured Swin/DeepJSCC paths, retained NTSCC wrappers, process workers, RAW/JPEG benchmarks, per-image failure records, CSV v1 schema and append dedup preserved. |
| SemParam | Image-only CLIP runtime independent of codec checkpoints; injected FeatureProvider for semantic codebooks; evidence availability/provenance retained; settings and training owned by SemParam. |
| Scheduler | FIFO, round robin, fixed class priority, class priority + EDF within class. Stable typed IDs, immutable snapshots, explicit budget units, deterministic tie breaking. |
| INFRA | Config precedence/validation/path resolution, UE job lifecycle, arrivals/expiry/clock, resource limits, transport selection, run artifacts, subprocess workers and legacy combined showcase. |
| Training / notebooks | Eight active repository-backed guides with clean Git sync option, CPU/CUDA detection, explicit TPU rejection and results outside disposable checkout. Reusable training modules, checkpoint persistence, provenance and CPU loop test. |
| Evaluation / graphs | Failure quality stays unavailable; latent bytes, payload bits and complex channel uses remain distinct. Timed channel/decoder stages for EncDecCodec; matched-run plotting; sample counts and SD labels; lossless infinity shown explicitly. Representative PNGs visually inspected. |
| Collaboration | AGENTS.md, PROJECT_MAP.md, module READMEs, migration maps and run guide. Both sides of previous README conflict retained in historical guide. Graphify refreshed: 1,048 nodes, 1,524 edges. |

## Validation evidence

- Baseline before migration: **24 passed, 3 skipped**.
- Final model-enabled suite: **58 passed, 1 skipped**. The sole skip needs two CUDA GPUs. Existing upstream deprecation warnings remain.
- Final plotting-only check after visual correction: **3 passed**.
- Eight active notebooks passed nbformat structural validation and local execution of every Python cell. Training switches stayed disabled. This validates setup, not a full training run or hosted Kaggle/Colab session.
- Editable installation succeeded in both local environments; uv.lock refreshed. Canonical modules compile; old Stage 1A/1B validation commands run.
- Worker protocol exercised through an actual subprocess using a clean, pinned test-factory repository.
- JPEG Q20: 24 images, zero failures, mean PSNR **29.5108 dB**, SSIM **0.8320**. This is codec reconstruction without digital PHY.
- RAW: 24 images, zero failures; lossless PSNR is positive infinity, SSIM 1. No invented finite quality value.
- Swin: real saved checkpoint, Kodak `kodim01`, AWGN 10 dB, rate 96, CPU smoke passed. This is one image, **not** a complete Swin Kodak evaluation. EncDec outcomes separately recorded encoding, channel, decoding and latent memory.
- CLIP: four real demo images processed without Swin loading. Saved semantic codebook and communication K256 codebook also passed real-asset smoke checks. Semantic codebook remains demo-only.

Original assets and prior failed validation attempts were preserved. Missing SSIM dependency and the initial torch-version mismatch were resolved using the declared metric package and an isolated torch 2.10 environment. Checkpoint/source guards were retained.

## Assigned IDs and synthetic baseline evaluation

The following labels are explicit fixtures, not AIDER labels or classifier predictions. Smaller priority class means higher priority. Budget: 100 complex channel uses/slot, one job/slot, 10 ms slots, 200 ms horizon. These resource demands are synthetic values, not measured image payloads.

| Job | Image | UE | Class | Arrival (ms) | Deadline (ms) | Demand (complex uses) |
|---|---|---|---:|---:|---:|---:|
| job-0001 | synthetic-image-001 | ue-001 | 2 | 0 | 100 | 60 |
| job-0002 | synthetic-image-002 | ue-002 | 0 | 0 | 30 | 60 |
| job-0003 | synthetic-image-003 | ue-001 | 0 | 0 | 20 | 40 |
| job-0004 | synthetic-image-004 | ue-003 | 1 | 10 | 60 | 60 |
| job-0005 | synthetic-image-005 | ue-002 | 0 | 20 | 40 | 40 |
| job-0006 | synthetic-image-006 | ue-003 | 2 | 30 | 120 | 100 |

| Policy | Selected job order | Completed | Deadline met | Mean wait of completed jobs (ms) |
|---|---|---:|---:|---:|
| fifo | job-0001, job-0002, job-0004, job-0005, job-0006 | 5/6 | 83.3% | 8.0 |
| round_robin | job-0001, job-0002, job-0004, job-0005, job-0006 | 5/6 | 83.3% | 8.0 |
| fixed_priority | job-0002, job-0003, job-0005, job-0004, job-0001, job-0006 | 6/6 | 100.0% | 15.0 |
| priority_edf | job-0003, job-0002, job-0005, job-0004, job-0001, job-0006 | 6/6 | 100.0% | 15.0 |

FIFO and RR expire job-0003 on this fixture. Priority baselines deliver all six but increase mean waiting time among completed jobs. Different completed subsets affect the mean; these six jobs establish behavior, not general policy superiority. EDF sorts deadlines **inside** each class. Full rows are in `scheduler-decisions.csv`.

## Reproduce

From the repository root, with its virtual environment active:

```powershell
python -m pip install -e .[encdec,semparam,scheduler,metrics,benchmark,test,notebook]
python E2EMain.py --module INFRA --action demo
python E2EMain.py --module Scheduler --action evaluate --implementation all
python E2EMain.py --module EncDec --action benchmark --implementation jpeg
python E2EMain.py --module SemParam --action extract
python -m pytest -q
```

Existing Swin artifact tests require the compatible environment:

```powershell
$env:STAGE1_MODEL_DIR = "models/SwinJSCC"
.\.venv-models\Scripts\python.exe -m pytest -q
.\.venv-models\Scripts\python.exe E2EMain.py --module EncDec --action smoke --implementation swinjscc
```

See `docs/REBUILD_GUIDE.md` for CPU-wheel setup, profiles, workers, training and old entry points. Root `config.yaml` selects paths/actions/implementations; SemParam now owns `config/extraction.yaml` and `config/codebook_training.yaml`. Image-codebook training is also available through `--module SemParam --action train --implementation semantic_codebook`.

Graph refresh: `python scripts/refresh_graph.py --semantic docs/architecture_semantics.json` using the environment with Graphify installed. Graph content digest: `eaa7f3476ecc65f9080d3211f28cf5f13df4ee5b7bfb2c23a8ca069783198138`. The digest and per-file hashes describe the working tree independently of Git HEAD.

## Availability and remaining limits

- **Validated locally:** RAW, JPEG, Swin saved-model smoke, CLIP image extraction, saved communication/semantic codebooks, scheduler fixtures and worker protocol.
- **Unvalidated external assets:** DeepJSCC and NTSCC remain implemented adapters/wrappers requiring their configured sources/checkpoints/environments. No replacement checkpoints were fabricated. Optional Sionna runtime was not executed.
- **Training:** portable CPU/CUDA path implemented; a test-only tiny model verified a real CPU optimization/checkpoint cycle. Full model training, dual GPU execution, T4/P100/A100 performance and hosted clean-sync execution are not validated. TPU/XLA is unsupported.
- **Research deferred:** AIDER labeling/classifier training, a finalized faculty urgency rubric, RL scheduling and full multi-UE wireless system simulation. Current simulation is nonpreemptive slotted service; it does not infer PHY delivery success or reconstructed quality.
- **Metrics:** LPIPS disabled in this run. Missing timings/metrics are null. No claim of model accuracy, research novelty or scheduler superiority follows from the rebuild.
- **Repository state at this snapshot:** the rebuild was local when these
  original measurements were made. The later team-main merge has separate
  validation below.

## Post-merge validation (teammate `main` at `1e44baa`)

- Compatible model environment: **62 passed, 1 skipped**. The skip requires
  two CUDA GPUs. The three warnings are existing dependency/runtime warnings.
- New team compatibility imports resolve to `EncDec/Research/`; the four-method
  evaluator factory imports successfully. Focused worker/factory tests pass.
- Both incoming notebooks pass `nbformat` structural validation. Their full
  execution was not attempted because the production model checkpoints are absent.
- The 32 entries in `models/MANIFEST.sha256` initially arrived as Git LFS
  pointers, each with an object ID matching the manifest. Local `main`
  fast-forward hydrated all 32; **actual asset bytes** now match their
  recorded SHA-256 hashes. This validates storage integrity, not inference.
- The two production `.pth` checkpoints for the repaired DeepJSCC/NTSCC
  comparison are not present in this repository. No post-merge real inference
  result is claimed for those methods.

## Artifact index

- `PROJECT_MAP.md`, `docs/MIGRATION_MAP.md`, `docs/NOTEBOOK_MIGRATION.md`, `docs/REBUILD_GUIDE.md` in the repository.
- `model-availability.json`, `preservation-audit.json`, `notebook-validation.json`, `codebook-smoke.json`, `final-tests.log` beside this report.
- `scheduler-comparison.png`, `jpeg-kodak24.png`, `raw-reconstruction.png`, `scheduler-decisions.csv` beside this report.
- `architecture-graph.html`, `architecture-graph.json`, `architecture-source_revision.json`, `architecture-GRAPH_REPORT.md`, `architecture-AUDIT.md` beside this report.
- Full source backup: `C:\Users\satya\Documents\Codex\2026-09-22\ill-keep-this-prompt-here-ok\work\rebuild-backup\before-rebuild.zip`.
