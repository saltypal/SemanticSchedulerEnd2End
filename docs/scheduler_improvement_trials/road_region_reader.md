# Road-region reader development gate

The previous native run decoded all 24 messages at 464 image payload bytes each. Its reader found zero of the 12 visible road-damage positives after reconstruction. This follow-up tests whether a receiver can improve recognition by looking at road-heavy parts of the image before changing the codec rate or scheduler.

## Frozen comparison

The candidate uses the existing SQ-GAN bundle's road segmenter and the frozen LADI-v2 large reference classifier. The segmenter sees an aspect-preserving fit in a 512 by 256 canvas. Its road predictions rank nine overlapping half-size tiles. The reader scores up to three road-heavy tiles at source resolution, reports their maximum road-damage score and falls back to the full frame if no road is detected. The full-frame comparator uses the same damage classifier. The crop rule uses pixels alone. It never receives evaluator labels, a sender mask or a human-supplied road location.

Calibrate each scoring path on the previous 24-image development batch. For each path, place its threshold immediately above the third-highest score among calibration negatives. This caps calibration false positives at two. Test the frozen rules on fresh publisher validation originals. The source gate requires at least six correct positive calls out of nine and at most two false reports among 12 negatives. Only a passing source receipt unlocks the paired SQ-GAN run. A reconstruction-only failure triggers a rate comparison against an equal-byte control.

The acquisition excluded 69 previously used validation identities across four manifests and screened raw hashes and RGB thumbnail differences against their local files. Several source folders occur in both calibration and evaluation. This is a fresh-image development screen with potential event correlation. LADI labels visible road damage at image level and does not independently label Critical urgency. [Publisher dataset and label scope](https://huggingface.co/datasets/MITLL/LADI-v2-dataset)

## Measured decision

| Fresh source path | Frozen threshold | True positives | False reports | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Road-region reader | 0.01001665 | 9/9 | 3/12 | 0.925 |
| Full-frame comparator | 0.00191013 | 8/9 | 1/12 | 0.977 |

The road-region candidate failed the false-report gate. It recovered one extra positive and added two false reports relative to the comparator. Both added false reports came from selected crops. Three images used the full-frame fallback, including the false report shared by both readers. The whole-frame comparator passed on this batch but failed on the previous separate fresh batch. This batch alone does not establish that the comparator is qualified across source groups.

The paired codec run stayed closed because the road-region reader failed on originals. The codec rate curve is conditional on a failure after reconstruction from a source-qualified reader. No rate was changed and no scheduler was retrained. The protected multi-event holdout stayed closed.

## Reproduction

Use the same pinned bundle and reader snapshot as the [large reference trial](strong_reader_trial.md). The four prior manifests referenced by the acquisition command must have their source images available. The frozen reader evaluation saves per-image scores, crop counts and source hashes.

```powershell
python -m Evaluation.scheduler_fresh_reader_cohort $validationCsv $priorManifest $fresh --exclude-images-from $calibrationManifest --exclude-images-from $previousRetryManifest --exclude-images-from $variantManifest --positive-count 9
python -m Evaluation.scheduler_road_region_reader $calibrationManifest $reader $bundle "$fresh/calibration.json"
python -m Evaluation.scheduler_road_region_reader "$fresh/manifest.json" $reader $bundle "$fresh/source_screen.json" --calibration "$fresh/calibration.json"
```

The fresh evaluation reproduced every score and decision exactly on a second run. Runtime measurements varied. The paired CLI rejected the failed source receipt, and 21 focused scheduler and codec tests passed. [Receipt hashes and verification](road_region_receipts/run_manifest.json)
