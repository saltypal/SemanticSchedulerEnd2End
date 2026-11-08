# Event-separated visible-obstruction reader trial

## Data decision

The LADI validation audit excluded row IDs used in the prior balanced, variant, large-reader, calibration retry, and road-region trials. Five road-present positives remain. All five have numeric source prefixes with unresolved event identity. They cannot support the proposed event-separated check.

This trial uses an explicit new endpoint, visible physical obstruction in a centered road region. The [CRASAR road-condition benchmark](https://arxiv.org/abs/2512.12128) distinguishes obstruction from flooding, particulate, road-surface damage and uncertainty. All source files come from the publisher's training partition. The protected scheduler holdout and the publisher test partitions remain closed.

## Source screen protocol

Use Hurricane Ian for training, Champlain Towers Collapse for threshold calibration, and Hurricane Harvey and Hurricane Ida for the fresh source screen. Train one logistic head on frozen CLIP features of the central 256-by-256 road region of each 512-by-512 crop. Train the same head on the whole crop as a comparator. Each variant uses the same training images and no scheduler features. Freeze each threshold immediately above its third-highest negative calibration score.

The centered candidate must recover at least two-thirds of positives and report at most two false positives separately in calibration and each screen event. The screen is a development gate with small per-event denominators. Native codec work is conditional on a passing source reader. Critical urgency requires separate reference labels.

Positive crops use road-line intersections with publisher Partial or Total Obstruction polygons. Negative candidates use the publisher Road Line geometry with no adverse or unknown annotation intersecting the target square. This follows the publisher's distinction between its base clear-road class and adverse overlays, and remains a crop-level label derived from spatial annotations. The rule excludes windows with incomplete alpha, crops shifted by raster boundaries, overlapping same-source windows and similar RGB thumbnails. The reader assumes a centered road request. It does not establish automatic road localization from arbitrary camera views.

## Acquisition correction

The first download built the 32 Ian training crops and stopped at calibration because the Champlain raster is stripped RGBA. The existing range reader expected tiled TIFFs. The revised reader handles both layouts. A local synthetic tiled and stripped pair reproduced the expected RGB pixels and alpha coverage. A real 512-by-512 Champlain window also decoded at full alpha coverage. The retry kept the same event roles, labels, quotas and source decision thresholds. The first attempt had no reader scores.

The retry stalled during Harvey range requests. It resumed from 56 hash-verified crops using a journal, a request timeout and grouped strip reads. The final manifest contains 86 unique images. A separate pass verified all crop hashes, dimensions and pairwise thumbnail differences.

## Measured source results

Each nontraining event contains six obstruction positives and twelve clear-road controls. A pass requires at least four detections and at most two false reports in each event.

| Event | Role | Centered reader TP / FP | Whole-crop comparator TP / FP |
| --- | --- | ---: | ---: |
| Hurricane Ian | Training | 16/16 and 3/16 | 16/16 and 6/16 |
| Champlain Towers Collapse | Calibration | 3/6 and 2/12 | 3/6 and 2/12 |
| Hurricane Harvey | Source screen | 2/6 and 2/12 | 3/6 and 5/12 |
| Hurricane Ida | Source screen | 5/6 and 0/12 | 6/6 and 4/12 |

The centered model passed on Ida. Both models failed the overall source gate. A retrospective threshold diagnostic found that either model could recover at most two Harvey positives while allowing at most two false reports. It did not change the frozen operating point. The controls are annotation-derived clear-road regions. They are not a separately adjudicated set of confusable shadows, wet pavement and roadside debris.

## Decision versus the original scheduler

- The no-COCO header improvement remains verified. In the matched archived replay, the complete trial application header and CRC averaged 110.45 B before removal and 84.16 B after removal. SPI-first made identical decisions across 15 episodes. [Matched original-format comparison](no_coco_full_header.md)
- The new centered reader improves the false-report count over the whole-crop comparator on Harvey and Ida, but it fails the per-event recall gate on Harvey and calibration. This experiment measures obstruction rather than the original scheduler's proxy-urgency waiting penalty.
- The 464 B paired trial, conditional rate curve, and scheduler retraining stayed closed after source qualification failed. There is no verified gain in timely correct evidence or independent Critical service. The protected scheduler holdout stayed closed.
- The next reader should train on more than one event and use separately reviewed hard negatives and fresh event-separated evaluation. The Harvey result supports improving discrimination through data and model changes before choosing another threshold.

## Reproduction

```powershell
python -m Evaluation.scheduler_obstruction_cohort docs/scheduler_improvement_trials/crasar_training_condition_candidates.json $cohortDir
python -m Evaluation.scheduler_obstruction_reader "$cohortDir/cohort.json" $clipSnapshot "$cohortDir/source_screen.json"
```

The acquisition requires Shapely, fsspec HTTP support, tifffile and imagecodecs. The crop journal can resume an interrupted download with `--resume`. The pinned CLIP snapshot and its weights hash are recorded with the fitted heads, scores and metrics. The 20-minute checkpoint fired at 10:23 IST while acquisition was still running and exited after one check.

The second reader run reproduced the fitted heads, thresholds, per-crop scores and decisions exactly. Timing was excluded from equality. Thirteen focused scheduler and codec tests passed. [Cohort and result hashes](event_obstruction_receipts/run_manifest.json)
