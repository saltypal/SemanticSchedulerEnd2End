# Consolidation into SemanticSchedulerEnd2End

## Purpose and destination

`SemanticSchedulerEnd2End` is the single working checkout. It contains the
existing four-component project, the current `BetterScheduler` parameter package,
and preserved research records. No source checkout was reset or deleted.

| Location | Responsibility |
|---|---|
| `BetterScheduler/` | Current parameter extractor, SPI/CLIP/YOLO research, tests, reports and run map |
| `INFRA/` | Existing lifecycle and decoder infrastructure |
| `Scheduler/` | Existing policies and decoder policy implementations |
| `SemParam/` | Existing semantic feature/knowledge/header implementations |
| `EncDec/` | Existing codec and latent processing implementations |
| `research_archive/consolidation_20261027/` | Original checkout source snapshots, conflict ledger and historical Build artifacts |
| `results/` and `BetterScheduler/results/` | Existing and recovered experiment outputs |

Start with the root README and `BetterScheduler/README.md`. The target branch is
`semantic-aware-ai-scheduler`, based at `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`.
The source snapshots were copied from `SemanticSchedulerParameterAudit` on
`new-scheduler` and `SemanticSchedulerBuild`, whose worktree shared the parameter
checkout's Git database. They contained uncommitted work. Source originals were
preserved, not moved or deleted.

## Preservation and conflict policy

`research_archive/consolidation_20261027/manifest.json` records the copy source,
destination, SHA-256, size and conflict path for every copied file. The completed
operation preserved6,835 files:2,803 in the parameter package,626 source
snapshots,130 non-conflicting project files and3,276 historical artifacts.
It recorded169 conflicting source paths; the pre-existing destination files were
not replaced. Incoming file paths absent from the destination were copied first
from ParameterAudit, then from Build only when still absent. This selection is a
preservation rule, not a claim that one implementation is scientifically best.

The separate root-level ParameterAudit `results/` tree was also copied into
the destination's `results/`:173 files,36,323,614 bytes, no path conflicts.
Its individual source/destination hashes are in
`research_archive/consolidation_20261027/parameter_audit_results_manifest.json`.
These historical results are not new benchmark runs or proof of current accuracy.

Twenty-one historical artifacts >=10MiB use hardlinks to their original files to
avoid duplicating several gigabytes on the same filesystem. Treat those artifacts
as immutable: an in-place modification through either pathname changes the same
file. For a new run, write a new output rather than modifying an old one. The
complete list and hashes are in the manifest. Smaller artifacts are copies. The
archive is Git-ignored to avoid accidentally committing full historical logs or
large binaries; the record and source provenance remain readable locally.

After that preservation pass, the copied BetterScheduler README was simplified.
Its previous exact contents were saved separately as
`source_snapshots/parameter_audit/BetterScheduler/README.md`, SHA-256
`822b1eef56fdf95ac3a33298e68a3632f055093a433a633b6414595c6f925c49`.

## Integration contract

- `BetterScheduler` is discoverable as an installable package. The existing
  root module entry points and old public APIs remain intact. The optional
  `parameters` dependency group describes the image-parameter runtime; installing
  it can download large packages, so it was not installed automatically.
- `BetterScheduler/run_parameters.ps1` uses this checkout's `.venv`, CLIP model,
  selected SPI artifact and `BetterScheduler/results/user_runs`. It reports an
  unavailable detector if local YOLO weights are not provided.
- The current parameter-only header is35 bytes; the older core remains28 bytes.
  The existing decoder queue has not been adapted to the new format. Do not
  describe the repository as an end-to-end validated 35-byte scheduler.
- The corrected `20261027finalc` VLM text controls returned9/12, Medium0/3;
  no four-class image-label training occurred. The existing three-class head is
  a weak development fallback. Its original validation Critical recall was2/10.
- The independent YOLO worker returned300 scene records,202 with no retained
  detections. Those are model outputs, not evidence of true object absence or
  an independently measured detector accuracy.

## Validation and remaining work

Before integration edits, Luna High verified representative imports,50 parameter/
extraction tests and16 decoder/contextual tests in the consolidated checkout;
see `results/consolidation_baseline_20261027/`. A static-latent CLI smoke was
subsequently run successfully with an explicit unique output. After integration,
66 combined parameter/decoder tests,8 existing latent tests, representative
imports, package metadata and a real-image launcher check passed; see
`results/consolidation_final_20261027/`. These checks are engineering checks
against a mutable dirty worktree, not evidence of ranking quality.

The default full-suite attempt did not reach test execution: collection of
`INFRA/tests/test_disaster_four_parameter_study.py` failed because
`EncDec.Models.scale_hyperprior_bundle` is absent from all three preserved
checkouts and their inspected history. The file and tests remain visible, not
silently skipped or replaced with a fabricated codec. A separately logged
remaining-suite run excludes this one module and must be read with that limit.

The first remaining-suite run exposed a version mismatch:33 scene tests expected
the queue/context injection supported by the Build version of `DecoderService`,
while the newly copied Audit version lacked those optional arguments. The
backward-compatible optional queue/context hook was merged into the newly copied
service, without replacing the original service snapshots. Luna rechecks both
legacy and scene paths after that fix. A separate terminal replay also refers to
an absent historical `20261023_stageB_v1/full/study` selection artifact; that
artifact was not found in the preserved checkouts and has not been fabricated.

Further work should consume one stable parameter contract, establish how the
35-byte record reaches the receiver queue, and validate image/mission labels and
YOLO measurements independently before choosing or training a ranking policy.
Do not delete archived versions, retrain on evaluation labels or infer that
multiple snapshots are interchangeable merely because imports pass. The user is
reviewing parameter and policy design separately.
