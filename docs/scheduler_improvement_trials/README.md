# Scheduler improvement trials

## What we are testing

The receiver has one decoder. When several images are ready, the next image should ideally tell the receiver something it still needs to know. A second picture of a road the receiver already understands might be less useful than a picture of a different road. That depends on whether the scheduler can recognize the repeat before decoding and whether the reconstruction actually supports a correct decision.

The main comparison changes one thing: whether the scheduling score uses completed receiver observations. Both policies must receive the same header, service estimate, duplicate gate and age guard. The unit of useful evidence is a verified road-segment condition within a defined time window. A true condition earns credit once and only after a correct reader output within its time budget.

## Day 1 — Development qualification, 7 November 2026

The branch starts from scheduler commit `43aa658b7e91f5ab574f3aba920454add918c085`. The original teammate checkout contained local changes, so the trials use a separate checkout. The protected scheduler test is closed. The 30-minute checkpoint is 15:00 IST.

### Reader and header-only ceiling

The first probe uses the existing 12-image class-stratified LADI sample. The sample was already used for development. It has native image-level `roads_damage` labels, which describe visible road damage. They do not provide a road-segment identifier, obstruction adjudication or an arrival trace. The frozen MIT Lincoln Laboratory reference reader uses its published road-damage logit. Its model card identifies the training split as the training source.

The probe compares three paths under the same 0.5 decision threshold:

1. Source pixels → frozen reader.
2. Source pixels → frozen reader → one-byte quantized score → receiver decision.
3. Source pixels → repository JPEG Q55 → actual bytes → reconstructed pixels → frozen reader.

The prototype packs the quantized score and a 64-bit difference hash into nine bytes. Those bytes exclude message identity and transport overhead. The hash describes coarse appearance. It is computed before dispatch and has no access to reference labels. Its availability alone does not validate repeat linkage. Header-only scoring pays for the full sender inference used to produce its score. JPEG is a byte-codec control while the pinned SQ01 bundle is being located. JPEG measurements cannot substitute for SQ-GAN measurements.

The first attempt stopped because the installed Pillow did not expose the method used to read the downsampled pixels. The corrected probe uses the grayscale buffer directly. The run receipt stores source, model, and manifest hashes and separate timings for sender scoring and header production, encode, decode, and receiver inference.

### Result from the initial 12 images

All three paths called all 12 images negative. Each matched 11 native negatives and missed the only native positive. The positive's source score was 0.00213 and JPEG-reconstruction score was 0.00250. Both are far below the fixed 0.5 threshold. The header rounded the source score to one quantization step. The result shows that the initial reader setting failed its only positive example. Identical decisions here do not establish that a header-only system solves the task.

The median local CPU time was 1,623.7 ms for source image preparation, scoring and header creation, 47.8 ms for JPEG encoding, 61.9 ms for JPEG decoding and 1,505.9 ms for the receiver reader. The median payload was 236,698 bytes. All 12 local images lacked capture timestamps and GPS in the inspected EXIF fields. The first batch therefore cannot qualify chronology or road linkage.

The follow-up uses a separate, label-balanced public LADI validation batch and keeps the test split closed. Sampling is conditional on a visible road and is for development diagnostics. No population precision or field prevalence will be inferred from it.
### Balanced follow-up

The first request for a 100-row dataset page returned HTTP 502 after some images had downloaded. The retry used a new output directory, 25-row pages and bounded transient-error retries. It obtained 24 development images from the official LADI validation split: 12 native road-damage positives and 12 negatives, all with a native road-present label. The class balance is by design.

| Path | True positives | False negatives | False positives | True negatives |
| --- | ---: | ---: | ---: | ---: |
| Source reader | 3 | 9 | 0 | 12 |
| Header-only score | 3 | 9 | 0 | 12 |
| JPEG reconstruction reader | 4 | 8 | 0 | 12 |

The reconstruction path recovered one more positive in aggregate at the fixed threshold. The result is too small and too development-specific to establish an improvement. Median local CPU measurements were 1,026.6 ms for sender preparation and header scoring, 31.5 ms for JPEG encoding, 41.6 ms for JPEG decoding and 1,147.6 ms for receiver reading. Median JPEG payload was 312,803 bytes. The reader is the dominant receiver cost in this setup. None of the 24 assets supplied capture time or GPS in the inspected EXIF fields.

All 24 downloaded images have distinct raw hashes, and none exactly matches the initial 12. No event or near-duplicate isolation is asserted. A leave-one-image-out development diagnostic placed each path's threshold just above its highest remaining negative score. Source and JPEG reconstruction each recovered 8 positives, while the 8-bit header recovered 7. Each path produced 1 false positive. This exploratory threshold check used the same small development set and is not a held-out estimate. It suggests the default threshold explains more of the measured loss than JPEG on this batch, and that score quantization can remove separation at low values.

The team confirmed there is no receiver-arrival log or independently reviewed same-road linkage available internally. The CRASAR training-only audit found 18,590 road-line IDs in 36 annotation files and zero IDs reused across files. Rounding the two geographic endpoints to five decimals produced 548 candidate endpoint pairs appearing in multiple files. These are leads for a geographically matched cohort. They do not establish stable road identity, condition agreement, temporal availability or dispatch-time linkability. The source GeoTIFFs are large. A 64 KiB byte-range probe on one training raster succeeded. The smallest model configuration inspected in this pass still lists a roughly 977 MB checkpoint. A full model bundle has not been selected or qualified.


### Input qualification

- LADI supplies native image-level road-damage labels and can support a bounded reader study. The local sample cannot establish same-road identity.
- RescueNet supplies blocked and clear road labels from Hurricane Michael. One disaster cannot support a multi-event claim by itself.
- The CRASAR-U-DROIDs road benchmark supplies aligned road annotations over ten disasters. Its orthomosaics and aligned labels are a candidate source for matched road views. Orthomosaic order is not a receiver arrival trace. Repeated views need independent linkage and must remain separate from runtime observations. Access to file listings establishes availability of the listing only. It does not qualify the dataset or any model for this trial.

Sources: [LADI dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset), [frozen reference reader](https://huggingface.co/MITLL/LADI-v2-classifier-small-reference), [RescueNet](https://doi.org/10.1038/s41597-023-02799-4), [CRASAR road benchmark](https://ojs.aaai.org/index.php/AAAI/article/view/41248), [CRASAR dataset](https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/tree/4c2732a7ab9658517ca8caa69bfee9e1f87ca532).

## Day 2 — Protocol and holdout gate

The controlled development replay is frozen in the saved command and run receipts. It uses a per-source-image fact proxy and exact injected repeats. It cannot answer the natural road-segment question. The production-style holdout remains unqualified.

Before opening a task holdout, require reviewed same-segment linkage across distinct views, road-condition labels and event roles with source and near-duplicate checks. The scheduling feature must be available from the message bytes at admission. Keep evaluation road IDs, condition labels and future service outcomes out of the policy. Freeze the actual source profile, reader, codec, deadline definition, effect threshold, false-report limit, tail-wait limit and analysis code. A physical arrival trace and a chronological capture replay must remain separate scopes. Without a field arrival log, report controlled replay with explicit synthetic transport and queue assumptions.

## Day 3 — Matched development replay and correctness checks

The scheduler prototype receives a token, arrival, completion deadline, task score, appearance signature and expected service duration. Its static and conditional variants both discount exact signatures for completed positive reports. The conditional variant also reduces priority for near signatures. Both share the age guard, deadline check and service estimate. The guard prioritizes an old ready job once the wait threshold is reached. It does not guarantee a maximum wait under overload or nonpreemptive service.

The replay uses recorded sender and JPEG encoding costs before admission and recorded decode and reader times during receiver service. It assumes independent preparation for each offer and no measured network delay or shared sender contention. It uses common offered jobs, three nominal load levels, and a capture-relative deadline of six median service times. The same image list appears twice to inject exact repeats. Failed and unfinished jobs cannot credit evidence. Receiver state uses completed positive reports, including false positives. In this development variant, late successful reports may inform subsequent scheduling but earn zero timely evidence.

Seven focused tests pass. They cover empty queues, exact-repeat equivalence, an actionable near-repeat fixture, age and deadline behavior, failed and unfinished jobs, false state and unique credit. The fixture confirms the mechanism can change a decision when its conditions hold. Its labels and results are constructed. The historical SPI-first and marginal urgency baselines need an SPI input aligned to the new cohort and were not run in this road-damage replay.

| Nominal load | FIFO timely / offered | Static duplicate-aware | Receiver-conditioned | Max started wait, FIFO / static / conditioned |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 4/12 | 4/12 | 4/12 | 10 / 10 / 10 ms |
| 0.9 | 4/12 | 4/12 | 4/12 | 2,157 / 3,025 / 3,025 ms |
| 1.2 | 2/12 | 3/12 | 3/12 | 10,710 / 10,710 / 10,710 ms |

All cells had zero observed false reports on this small selected sample. At load 1.2, all policies left five jobs unstarted. There were 21 late completions under FIFO and 27 under each score-based policy. The two score-based policies had identical dispatch orders on these traces. Their extra timely positive at load 1.2 comes with more late completions. Maximum started wait excludes unstarted jobs and must be read with the latter count.

[Development figure](dev_replay_coverage.png) · [Detailed CSV](dev_replay_table.csv) · [Reproduction hashes](development_manifest.json)

## Day 4 — Holdout remains closed

The qualification gates have not been met for a multi-event road-segment experiment. The 24-image sample has no independently verified cross-view road linkage, and the public training annotation audit yielded candidate links only. No holdout result, event-level confidence interval or leave-one-event-out statistic is reported.

## Day 5 — Current decision

**No-go for a journal claim from this run.** The controlled pilot provides a working and testable comparison. Its receiver-conditioned policy shows no measured incremental gain over the matched duplicate-aware control. The follow-up dataset work should qualify physically distinct views of the same independently identified road segment and demonstrate reliable completed-reader decisions before spending effort on another learned scheduler.

## Reproduction commands

Run from the trial checkout. Use new paths for raw probe and replay receipts.

```powershell
python -m Evaluation.scheduler_trial_dev_samples <new-development-image-dir>
python -m Evaluation.scheduler_trial_probe --sample-manifest <manifest> --model-dir <local-reference-model> --output <probe.json>
python -m Evaluation.scheduler_trial_crasar_audit <linkage-audit.json>
python -m Evaluation.scheduler_trial_replay <probe.json> <replay.json>
python -m Evaluation.scheduler_trial_report docs/scheduler_improvement_trials/dev_balanced_replay.json docs/scheduler_improvement_trials
python -m pytest Scheduler/tests/test_trial_evidence_policy.py Scheduler/tests/test_scheduler_trial_replay.py -q
```
