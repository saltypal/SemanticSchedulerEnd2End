# Fixed-deadline replay and road-corridor reader ablation

## Replay changes and verification

The fixed-deadline mode uses seeds 0 through 9 for matched workload permutations, regular and bursty captures, and nominal loads of 0.5, 0.9 and 1.2. Capture cadence uses total measured receiver service divided by the capture window. A separate six-image SQ-GAN profile freezes the experimental deadline at 83.816 s from capture. The profile shares no source or payload hashes with the 24-image replay. The deadline is an engineering test setting.

The receipt distinguishes offered work and work admitted within the capture window, backlog at the end of that window, and the later observation horizon. Arrivals are synthetic and delivery is ideal. Reports keep late decisions, failed decodes, unfinished service and unstarted work separate. Critical completion and correct task decisions are separate counters. They remain unavailable on the real pilot because it has no independent Critical references. Regression tests exercise the distinction with synthetic references.

All 60 scenarios preserve arrivals, payloads and deadlines across FIFO, SPI-first and SPI-first plus the cue. The archived replay and the new fixed-deadline rerun reproduced byte-for-byte. At 1.2 bursty load, admitted receiver service was 1.2 times the capture window, and the median queued count at window end was four for each policy. Median p95 observed waits across ten seeds were 247.4 s, 521.8 s and 522.1 s respectively. The cue changed dispatch order in 34 of 60 scenarios. Each policy delivered zero timely correct positives on the measured reader outcomes.

A perfect-reader sensitivity check replaces observed decisions with evaluation truth without showing it to the dispatch policies. At bursty load 1.2, the median timely positives out of 12 were 1 for FIFO, 2.5 for SPI-first and 3 for SPI-first plus the cue. This estimates room for improvement under the test deadline with the existing policies and times. It is a counterfactual, with no measured gain for the cue.

## Reader diagnosis

Visually audited the five Mayfield false reports and all five missed obstructions from the prior fresh screen using the centered image and the masked version side by side. Some misses show apparent roadside or boundary objects, making the published spatial label difficult to verify from the centered view alone. Labels remain as published. Six same-raster obstruction and clear-road training pair candidates were reviewed. Two were retained as relatively clear review examples. This small visual review is not independently adjudicated and does not supply a fresh test cohort.

The ablation uses the nearest annotated road line and local pixel scale to keep a 7.2 m corridor, following the width used by the [CRASAR benchmark](https://arxiv.org/abs/2512.12128). It masks outside that corridor and fits the same centered reader with the original training and calibration partitions. The geometry is an annotation-derived corridor rather than a verified travel-lane boundary. Median center-crop retention was 90.4%. The scored Mayfield and Michael images were previously exposed, so this is a development comparison.

| Event | Centered reader TP / FP | Corridor-masked reader TP / FP |
| --- | ---: | ---: |
| Champlain, calibration | 3/6 and 2/12 | 4/6 and 2/12 |
| Mayfield, development | 4/6 and 5/12 | 4/6 and 4/12 |
| Michael, development | 3/6 and 0/12 | 3/6 and 0/12 |

The corridor removed two Mayfield false reports and introduced one new false report on a clear-road crop containing a vehicle. It recovered one previously missed positive and lost one previously correct detection. The false-report gate still fails in Mayfield and recall still fails in Michael.

## Decision

The native codec pilot remains verified and the simulation upgrade is complete for controlled replay. The road-corridor reader failed development qualification. The new paired 464 B codec run, conditional equal-byte rate curve and protected holdout remain gated. The encoder cue still has insufficient positive post-decode outcomes for predictive qualification beyond SPI. Keep no-COCO SPI-first as the verified baseline. The next data requirement is a set of independently reviewed travel-lane boundaries and matched on-lane and roadside examples, with fresh events available for qualification.

## Reproduction

```powershell
python -m Evaluation.scheduler_codec_study docs/scheduler_improvement_trials/strong_reader_receipts/paired_spi.json $replay --deadline-profile docs/scheduler_improvement_trials/native_sqgan_six_pilot.json
python -m Evaluation.scheduler_lane_ablation $reviewedCohort $laneOutput
python -m Evaluation.scheduler_obstruction_reader "$laneOutput/cohort.json" $clipSnapshot "$laneOutput/source_screen.json"
```

The lane reader rerun reproduced all scores and decisions exactly after excluding timing. Sixteen focused tests passed. The [receipt index](fixed_replay_lane_receipts/run_manifest.json) links the per-condition metrics, crop audit, review decisions and full replay digest.
