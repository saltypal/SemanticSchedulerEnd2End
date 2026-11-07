> Current MEDIC study and the new image-provider/bundle interface are documented
> in [MEDIC_PARAMETER_HANDOFF.md](docs/MEDIC_PARAMETER_HANDOFF.md). The commands
> below retain the earlier vector-classifier compatibility path. A model being
> trained or selected is not automatically approved for runtime extraction.
>
> Four-class SPI execution is active. The launcher requires Low/Medium/High/Critical
> by default and will refuse the historical selected three-class artifact until
> a four-class release passes acceptance. `-AllowHistoricalThreeClass` is an
> explicit historical comparison option, not the final SPI result. Direct CLI
> consumers can use `--require-four-classes` for the same early guard.

# Parameters

This is the relocated BetterScheduler package, kept intact for the upcoming SPI
parameter redesign. No extraction, test or training run was performed during the
move. Runtime imports use `parameters`; previous experimental results describe
the earlier source layout. See the root PROJECT_MAP.md for ownership.

# parameters: image parameters

This directory contains the current parameter-extraction research for the
semantic scheduler. The single project root is `SemanticSchedulerEnd2End`.
See the root [README](../README.md) and [project map](../PROJECT_MAP.md) for the
other components.

## Run on an image

In PowerShell from the project root:

```powershell
.\parameters\run_parameters.ps1 -Images 'C:\path\image.png' -Mission 'Inspect visible road damage.'
```

The launcher uses this project's `.venv`, the existing CLIP checkpoint under
`models/priority_classifiers/clip`, and the selected SPI model. It saves the
parameter record and UE knowledge base under `parameters/results/user_runs/`.

Pass `-YoloCheckpoint 'C:\path\yolov8s-worldv2.pt'` only when local detector
weights and optional dependencies are available. Otherwise the scene fields are
explicitly unavailable. The script prints which SPI classes the model supports.
Use a different `-UeId` for an unrelated sender or mission; each UE/mission has
its own persisted knowledge base.

For direct programmatic use, `parameters` exports `FinalParameterExtractor`,
`ExtendedParameterHeader`, `SceneMeasurements`, `MissionSpec`, and
`SenderMemory`. Their implementations are in `minimal_parameters.py`,
`scene_measurements.py`, `contracts.py`, and `extractor.py`.

## What the parameters mean

- **SPI:** visible-response-attention estimate. The available fallback supports
  Low, High, Critical; Medium is unsupported. It is not validated urgency.
- **Relevance:** CLIP cosine between image and operator mission paragraph.
- **Novelty:** appearance difference from the same UE's causal history. A
  cold-start flag distinguishes the first image from a measured difference.
- **Capture:** trusted timestamp and operator-specified allowed delay.
- **Scene:** detection count, distinct detected-category count, largest box
  fraction and rectangular union coverage, from one configured YOLO-World model.

The version2 parameter header is35 bytes; version1 remains28 bytes. This is a
parameter serialization format; the existing receiver queue has not yet been
adapted to it. Model/profile identities require an agreed registry.

## Read the evidence

- [Parameter report](docs/PARAMETER_FINAL_REPORT.md): methods, definitions,
  experiments, byte layout, limitations and reproduction.
- [Teammate handoff](PARAMETER_HANDOFF.md): fields and information boundary.
- [Canonical plan](PLAN.md): current decision and earlier plans.
- [Kaggle runs](KAGGLE_RUNS.md) and [journal](JOURNAL.md): exact run identities,
  saved outputs, tests and failures.
- [Project consolidation](../docs/CONSOLIDATION_20261027.md): archived source
  versions, conflicts and baseline validation.

The corrected four-class VLM text controls returned9/12, with Medium0/3; no
four-class image labels or model were produced. YOLO processed300 cached images,
with202 yielding no detections. These are model outputs and diagnostics, not
independently validated urgency or detector-accuracy results. The older README is
preserved in `research_archive/consolidation_20261027/source_snapshots/parameter_audit/parameters/README.md`.
