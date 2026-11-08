# Multi-event obstruction reader retry

## Frozen development protocol

The previous reader trained on Ian alone and failed the source gate on Harvey and Champlain. This retry trains the same centered and whole-crop logistic heads on frozen CLIP features from Ian, Harvey and Ida. The 68 training crops were already used in the previous development round. Champlain supplies the same 18 calibration crops. Its results are development evidence.

Mayfield Tornado and Hurricane Idalia were initially selected from the publisher test partition for this one-shot reader qualification. Idalia had only three geometrically eligible positives, below the fixed quota of six. Before any reader inference, Hurricane Michael replaced Idalia based on annotation availability. Mayfield and Michael are used for reader development. Idalia is now acquisition-exposed and unscored. The project's protected scheduler holdout remains closed. Each scored event has a target of six obstruction positives and twelve clear-road controls. Use the previous crop geometry and prioritize clear target regions with adverse annotations nearby, outside the centered target. These controls are a defined context challenge and require visual audit before any claim about specific confounders.

The centered reader is the candidate. Freeze the threshold immediately above the third-highest negative calibration score. Keep the previous gate of at least two-thirds recall and at most two false reports separately in calibration and each new event. A passing source gate unlocks paired SQ-GAN at 464 B. The conditional byte-rate control and matched queue comparison follow only if their input gates pass.

## Blind crop review

The first Mayfield contact sheet revealed ambiguous controls, including a target centered on disturbed ground with no clearly visible road. Automatic scoring was stopped before it opened any evaluation outcomes. Acquire six additional negative candidates per event and extend by four if six are insufficient. In acquisition order, select the first twelve clear-road controls that visibly contain a usable road segment in the target. Keep ordinary traffic, shadow and nearby debris where the centered road is visibly clear. Exclude a target if it misses the road, its condition is obscured, possible debris obstructs the centered travel surface, or it visibly repeats a selected view. Record the exclusions and preserve the original annotations. Keep the original event quotas and false-report cap. The review is a visual quality check, with no independent second adjudicator. The six positives per event retain the publisher-derived spatial labels.

The reviewed cohort contains 122 unique crops. The 24 selected evaluation negatives include ten nearby-adverse context controls in Mayfield and eleven in Michael. The other three are visually clear road controls. The review pool and the retained crop hashes were locked before scoring. The hard-negative review applies to the new evaluation cohort. Training uses the existing annotation-derived controls across the three training events.

## Results and ordered decision

Each event below has six positives and twelve negatives. The gate requires at least four detections and at most two false reports in every row.

| Event | Ian-only centered TP / FP | Multi-event centered TP / FP | Multi-event whole crop TP / FP |
| --- | ---: | ---: | ---: |
| Champlain, calibration | 3 / 2 | 3 / 2 | 3 / 2 |
| Mayfield, fresh screen | 3 / 3 | 4 / 5 | 2 / 3 |
| Michael, fresh screen | 3 / 0 | 3 / 0 | 4 / 2 |

Relative to the frozen Ian-only head on the same fresh images, centered average precision improved from 0.634 to 0.757 on Mayfield and from 0.743 to 0.822 on Michael. At the calibrated operating points, the centered model gained one detection and added two false reports across the fresh screen. A retrospective diagnostic found no common threshold that passed all three events. Even an event-specific threshold on Champlain could recover at most three positives within the two-false-report cap. This diagnostic did not select a new threshold.

1. **Source reader.** Three-event training and the reviewed fresh-source comparison are complete. Both new heads failed the source gate. The next reader needs better spatial discrimination and independently reviewed training negatives near obstruction boundaries. The new events must stay out of any later fresh evaluation.
2. **Paired SQ-GAN.** The new 464 B run and conditional equal-byte rate curve stayed closed. No codec setting was selected.
3. **Cue and queue.** The matched policy comparison was not rerun with this reader. The earlier controlled replay remains the evidence for the cue, and no new scheduling benefit is established.
4. **Holdout.** No-go. Keep the no-COCO SPI-first baseline. The verified original-format comparison still saves 23.8% mean trial header overhead while preserving SPI-first decisions. [Header comparison](no_coco_full_header.md)

## Reproduction

The [reviewed cohort and receipt index](multievent_obstruction_receipts/run_manifest.json) preserve the accepted crop hashes, exclusions, thresholds, heads and scores. Use the same pinned CLIP snapshot and crop provenance as the earlier obstruction trial.

```powershell
python -m Evaluation.scheduler_obstruction_reader $reviewedCohort $clipSnapshot $newReceipt --reference-receipt docs/scheduler_improvement_trials/event_obstruction_receipts/source_screen.json
```

Two reruns reproduced every score, fitted head, threshold and decision exactly after excluding timing. The CLI rejects an unreviewed multi-event pool before model inference. Thirteen focused scheduler and codec tests passed.
