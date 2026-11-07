# SPI zero-Medium diagnosis and three-class training decision

27 October 2026. User authorized fixing/training and explicitly allowed removing
one class. No original label, rubric, source image or previous result is edited.

## What is established

The recovered candidate-c archive contains180 available labels:82Low,0Medium,
73High,25Critical. All180 raw JSON response levels agree with the saved labels.
The parser accepts Medium and the dedicated prompt includes all four classes.
`review_structured` uses the supplied SPI system prompt rather than injecting the
parameter-review system prompt. Vision pixels and thinking-off receipts are in
saved execution records. Thus Medium was not removed by a parser, label mapping,
training split or serialization bug: the teacher never emitted it.

Both sources lack Medium: LADI58Low/35High/7Critical; MEDIC24Low/38High/18Critical.
The support failure occurs before optimization; it cannot be fixed by training
longer or by class weighting. There are no Medium examples to weight.

## What likely contributed, and what remains unknown

The rubric defines Medium as useful situational evidence without prompt-response
need. Low includes benign informational content, while High includes visible
hazards/disruption. Without a particular mission, usefulness does not give a
sharp visual boundary from Low. Saved Low explanations repeatedly say no visible
hazard; High explanations repeatedly cite damage/flooding. This is consistent
with teacher boundary collapse and the benign-versus-hazard composition of the
small development cohort. It is NOT a controlled demonstration of either cause.
A single Low output-format example is another possible prompt influence, not
an established cause. Independent image labels or a controlled boundary study
would be needed to distinguish dataset support from teacher behaviour.

Critical also needs caution:13/25 evidence strings triggered a lexical triage
flag because damage wording did not explicitly mention ongoing harm/rescue.
That heuristic neither adjudicates the image nor proves the label wrong. It
reveals a risk that the teacher confuses severe aftermath with imminent harm.
Labels are preserved; this model is not independently validated action urgency.

## Fix selected now

Train an explicitly declared three-output classifier: Low, High, Critical. Use
the original180 pseudo-labels unchanged. Medium is UNSUPPORTED, not assigned
zero probability as an evidence-backed assessment, automatically merged, or
fabricated. This resolves the unsupported output-space assumption; it does not
repair all potential teacher-label errors or prove a universal three-level scale.

`spi.py` keeps default four-class training and v1 artifact loading for historical
callers. New explicit `--levels Low High Critical` creates a v2 artifact recording
its supported/unsupported classes and original rubric hash. It rejects any
available train/validation target outside that declared output space; no row
is silently dropped or remapped. Every configured class still needs real train
support. Unknown inputs have no validated OOD/abstention threshold yet.

Output indices0/1/2 map through canonical names to existing header classes
Low3/High1/Critical0, leaving the old Medium code unused. Metrics identify the
three-class order; compressed-order RPS/MAE are not numerically equivalent to
four-class ordinal metrics.

## Reproduce

From SemanticSchedulerParameterAudit using the existing project Python:

```powershell
& '..\SemanticSchedulerEnd2End\.venv\Scripts\python.exe' -m BetterScheduler.spi train --features BetterScheduler/results/spi_inputs_20261026a/features.npz --labels BetterScheduler/results/spi_three_class_20261027_inputs/spi_labels.json --levels Low High Critical --output BetterScheduler/results/spi_three_class_20261027 --epochs 50
```

The output path must be fresh. Source labels and diagnosis receipt:
`results/spi_three_class_20261027_inputs/`. Actual trained status, validation
results and model live under the output above; read those artifacts rather than
assuming successful training from this document. Validation is70 previously
exposed images grouped away from110 training images, not a fresh final test or
human-labelled urgency benchmark. No new dataset/model download or GPU run.


## Executed result

CPU fitting completed50epochs on110 train images. The exported512->128->3 head
is `results/spi_three_class_20261027/models/spi.json`; all13 focused engineering
tests passed. NumPy/torch prediction maximum difference1.62e-7. Actual cached
CLIP + image + head + minimal extractor CLI inference succeeded (High, finite
relevance,28-byte header). No inference download or new Kaggle run was required.

| Development metric | Student | Training-prior comparator |
|---|---:|---:|
| Teacher-label accuracy,70 images | 62.86% (44/70) | 47.14% (33/70) |
| RPS, compressed three-level order | 0.136516 | 0.186175 |
| Low recall | 26/33 | 33/33 |
| High recall | 16/27 | 0/27 |
| Critical recall | 2/10 | 0/10 |

Confusion rows true Low/High/Critical, columns predicted in the same order:
`[[26,7,0],[7,16,4],[3,5,2]]`. Training accuracy99/110=90%, including11/15
Critical, versus2/10Critical on validation. Thus training now works but the
urgency-generalization problem remains. Small support, teacher label errors and
representation limits are possible causes, not independently established causes.
Do not call the head reliable, universally applicable, human-calibrated or final
emergency deployment evidence. Merely tuning on these70 images cannot establish
those claims. No extra grid or thresholds were selected to hide the misses.

Full logs/provenance: `results/validation/spi_three_class_20261027/ops_summary.json`.
Validation/data/input hashes are saved there; original teacher labels remain
unchanged. A static assessment-only Medium review found no blocking issue in the
explicit-class code or wire mapping. Legacy v1 compatibility was inspected, not
claimed as a separately executed legacy-artifact regression. The initial cwd
and shell argument invocation mistakes failed before computation and were
preserved alongside successful corrected receipts.
