# Rebuild run guide

Run commands from the repository root. Python 3.12 is supported. `config.yaml` paths are relative to its own directory, independent of the shell's current directory. `--output` is an explicit path relative to the calling shell.

## Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[encdec,semparam,scheduler,metrics,benchmark,test,notebook]
```

Scheduler validate needs only the base project (PyYAML); graphs require the scheduler extra. Neural modules additionally need a suitable torch/torchvision installation. For the existing Swin artifact, Python 3.12 and torch 2.10 are enforced by its manifest. CPU installation example in a separate environment:

```powershell
py -3.12 -m venv .venv-models
.\.venv-models\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-models\Scripts\python.exe -m pip install -e .[encdec,semparam,scheduler,metrics,benchmark,test,notebook]
```

Use the appropriate PyTorch CUDA wheel for training hardware; notebook setup never silently replaces a hosted runtime's GPU wheel. Local CLIP extraction defaults to cached models; configure `semantic.local_files_only: false` explicitly to permit first-time download.

## Reproduce

```powershell
python -m pytest -q
python E2EMain.py --module EncDec --action availability
python E2EMain.py --module EncDec --action benchmark --implementation jpeg
python E2EMain.py --module EncDec --action benchmark --implementation raw
python E2EMain.py --module EncDec --action smoke --implementation swinjscc
python E2EMain.py --module SemParam --action validate
python E2EMain.py --module SemParam --action extract
python E2EMain.py --module Scheduler --action evaluate --implementation all
python E2EMain.py --module INFRA --action demo
```

Enable existing exported-model regression tests with `$env:STAGE1_MODEL_DIR = "models/SwinJSCC"` in the compatible environment. A dual-CUDA test skips on CPU. If local sandbox TEMP ACLs interfere, supply pytest `--basetemp=<new writable directory>`; do not interpret a permission error as a model failure.

Every CLI run writes resolved config, Git revision, dirty state, source digest, package versions, status and result JSON to `results/<module>/<run-id>` or configured destination. Scheduler traces include IDs and per-slot policy runtime. Kodak CSVs retain v1 columns and duplicate-key append behavior; historical shared CSVs are never automatically overwritten. To append deliberately use `append_rows_csv(destination, rows)`.

## Configuration

Precedence: module defaults → selected root `profiles` entry → root settings → explicitly supplied CLI. Root values intentionally override profiles. Use `--module`, `--action`, `--implementation`, `--profile`, `--device`, `--seed`, `--output`. Advanced settings use YAML; unsupported values fail explicitly.

`paths` selects Kodak, checkpoints, source and image roots. `benchmark` selects rate/SNR/JPEG quality/repetitions/LPIPS. `scheduler` selects slot/horizon/resource budget/max jobs. `transport.resource_unit` determines the synthetic resource accounting unit. `dependencies` configures external workers; there is no automatic download of teammate assets.

## External model workers

Configure `dependencies.external_worker` (or `dependencies.ntscc`) with absolute/resolvable `python`, `source_root`, exact clean Git `revision`, and `factory: package.module:function`. The function receives resolved config and returns a sequence of BenchmarkCodec instances. Add checkpoint paths as dependency fields. `EncDec.worker` runs the same CSV benchmark in the configured interpreter and writes a worker log, outcomes and environment. Failed model imports/checkpoints remain failures.

Existing `EncDec.Evaluation.repaired_notebook_models.build_repaired_notebook_codecs` preserves teammate DeepJSCC/NTSCC/native/codebook methods and their calibration exclusion logic. Run it in the owning environment; see historical team notes. The worker factory is the adapter seam, not a fabricated replacement checkpoint. Validate that external source is clean and pinned before comparing its results.

## Notebooks and training

Active notebooks are the numbered files directly inside each module's `notebooks/`. Their first cell offers clean Git sync into a new disposable checkout, a specified revision, and persistent output outside checkout. Local mode validates current work; remote mode requires the rebuild commit to be pushed first. Kaggle output must be saved with a notebook version; Colab mounts Drive.

Training is deliberately disabled by default in guides. Configure real train/validation directories and enable `RUN_TRAINING`. Source APIs live in `EncDec.training` and `SemParam.training`. The CPU/CUDA training API supports a single GPU or all visible GPUs; TPU/XLA is explicitly unsupported. The historical strict two-T4 helper remains available for exact old experiments. For CLI Swin training set `paths.train`, `paths.validation`, then run `python E2EMain.py --module EncDec --action train --implementation swinjscc`.

## Reading results

Synthetic priority labels are fixtures, not image-classifier predictions. Deadline performance is conditional on the supplied arrivals/budgets. Timing is measured on the local hardware, not a radio-latency guarantee. CSV failures have no PSNR/SSIM; null values mean unavailable. LPIPS is optional and disabled by default. Plots group compatible runs and identify sample counts; error bars are across-image sample SD, not confidence intervals from repeated experiments.
