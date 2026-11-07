# SPI classifier repair: results and decision

## Delivered
Measured grouped training-size curves for the existing frozen-CLIP head, internal MLP epoch curves, original 96-row confusion matrix, and a completed EfficientNet-B1 transfer experiment with epoch loss/accuracy/F1/recall curves, checkpoint and per-image predictions. No VLM labeling repeated. All 20 v15 recovery files checksum verified.

## Same96-image development comparison
| Metric | Frozen CLIP + balanced linear | EfficientNet-B1 transfer |
|---|---:|---:|
| Macro-F1 | .4637 | .4462 |
| Accuracy | .5104 | .5417 |
| Balanced accuracy | .5738 | .4852 |
| Low recall |23/43=.5349|24/43=.5581|
| Medium recall |5/8=.6250|3/8=.3750|
| High recall |16/38=.4211|22/38=.5789|
| Critical recall |5/7=.7143|3/7=.4286|

These are descriptive teacher-agreement results on exposed development data, not human urgency accuracy or final-test/SOTA results. Do not infer statistical superiority from these small counts. Architectures have distinct preprocessing/fitting procedures, so this is a bounded candidate comparison, not proof of a single architectural mechanism.

## Diagnosis
The local learning curve rises from CV macroF1 .420 with about36 fitting images to .518 with136; training F1 remains higher. This supports testing additional diverse training groups. Local sklearn1.9.1 differs from cloud1.7.2; absoluteCV differences are not claimed as head improvements. Three fold/subset-seed repetitions are descriptive and not independent datasets.
EfficientNet inner stopping selected epoch 6; selection continued to 12 before patience ended. Fitting macroF1 approaches .79 while inner macroF1 remains around .46-.50. This is consistent with limited generalization; it does not uniquely identify label noise, sample scarcity or representation failure.

## Decision applied
Do NOT replace the baseline or promote a cache: EfficientNet reduces macroF1 and minority-class recall. Both models fail the unchanged .70-per-class requirement. Runtime/header architecture remains unchanged. Missing diagnostic logging and plotting ARE fixed. A stronger visual architecture was implemented/tested and rejected on this cohort.

## Next experimental change justified by evidence
Expand independent training groups with targeted Medium/Critical support and varied incident/viewpoint/source content; preserve an untouched grouped evaluation cohort BEFORE further tuning. Audit label-boundary consistency separately. More examples of the same event are not independent diversity, and more normal images alone worsen class imbalance. No forced labels, severity-to-urgency mapping, or automatic full-dataset labeling follows from this experiment.

## Artifacts
- runs/20261028_123917/: local learning curves/raw rows/epoch history/environment.
- runs/notebook9_20261028_v15/monitor_v15/recovery_resumable/extracted/efficientnet_b1/: verified CNN curves, raw histories, validation, predictions/checkpoint/config/backbone identity.
- results/recall_comparison.png and comparison.json: measured comparison.
- RESEARCH_AND_DECISION.md: primary references and configuration; REVIEW.md and RESULTS_REVIEW.md: independent checks.
- parameters/KAGGLE_RUNS.md: versions14/15, recovery and resource records. Version14 packaging failure fixed using isolated archive import testing.
