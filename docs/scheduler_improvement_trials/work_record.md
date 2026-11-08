# Scheduler improvement trials — consolidated work record

This is the index for the published trial sequence through the multi-event obstruction reader retry. Each stage links to its detailed method, results and available receipts. The trials branched from original scheduler commit `43aa658`.

## Why 76.3% became 23.8%

Both experiments saved the same **26.288 bytes per message on average**. They divided that saving by different totals.

| What was counted | Original | No COCO | Saving | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Scene block only, average | 34.451 B | 8.163 B | 26.288 B | 76.3% |
| Fixed application overhead and CRC | 76 B | 76 B | 0 B | 0% |
| Complete trial application header and CRC, average | 110.451 B | 84.163 B | 26.288 B | 23.8% |

The same 6,000 cached scene records were used. The old scene block is already sparse, so many unused COCO entries occupied no per-entry space before removal. The fixed overhead is a 72-byte application prefix and a four-byte CRC trailer. The prefix carries the version, job/source/session identity, sequence, arrival/capture/deadline data, SPI scores, relevance, novelty, availability and lengths. The CRC protects the header and payload against accidental corruption. Payload bytes and link-layer framing are outside both measurements.

At p95 the scene block fell from 45 to 8 B and the complete application overhead fell from 121 to 84 B. The maximum possible dense scene block fell from 307 to 30 B. With this envelope, those dense limits correspond to 383 and 106 B. They are format limits. The observed scene maxima in the cache were 73 and 16 B. The paper's proposed 64-byte design is a separate layout.

## What was done, in order

1. **Created an isolated branch.** Started `scheduler_improvement_trials` from active scheduler commit `43aa658` in a separate checkout. Published the experiment code and receipts to that branch. The 30-minute checkpoint verified the first trial publication against GitHub. Historical models, default packet readers, and earlier result folders were preserved.
2. **Tested the reader and the value of decoding.** The original 12-image development probe had one native road-damage positive. All paths missed it. A balanced 24-image LADI validation batch had 12 positives and 12 negatives. At a fixed 0.5 threshold, the source and header-only paths detected 3/12 positives. The JPEG reconstruction reader detected 4/12. All had zero observed false positives on that selected batch. Median CPU times were 1,026.6 ms for source preparation and header scoring, 31.5 ms for JPEG encode, 41.6 ms for JPEG decode, and 1,147.6 ms for the receiver reader. The native LADI label describes visible road damage at image level. It does not establish obstruction or same-road identity.
3. **Compared scheduling policies on controlled replays.** Used the same synthetic arrivals, measured JPEG and reader durations, and fixed scoring rules across FIFO, static duplicate-aware, and receiver-conditioned dispatch. At nominal load 1.2, FIFO delivered 2/12 positives on time and both duplicate-aware policies delivered 3/12. In the exact-repeat replay, both score-based policies also produced six more late completions than FIFO. Mild brightness and blur variants did not give the conditional policy a gain over the duplicate controls. The independent header-only control delivered 3/12 without receiver decoding in the same zero-transport timing model.
4. **Checked cross-view repeat cues.** Audited 36 public CRASAR training annotation files. Rounded road endpoints yielded 548 cross-file geographic candidates, of which 12 shared a known condition label and three shared an obstruction label at the midpoint. The three came from Hurricane Ian. Retrieved six small windows using HTTP byte ranges and inspected them. The four-bit difference-hash gate matched 23/24 artificial pairs and none of the three selected geographic pairs. A 128-bit CLIP sketch still overlapped distinct-location pairs and took about 128 ms per crop on the local CPU. I separately checked the existing local footprint fixtures. They show that an overlap rule can help a true repeat and also suppress a nearby road or a rapidly changed condition. Those footprint changes were already in the checkout and were excluded from the COCO commits.
5. **Implemented opt-in COCO removal.** Added a v4 scene transport alongside v3. It removes all 80 COCO slots from the transmitted block and keeps five project objects and eight regions. The receiver retains the existing 387-column layout for old models by expanding omitted slots as zero values with unavailable masks. The codec checks vocabulary identity, format, masks, sizes and malformed sparse entries. The 6,000-record audit reproduced the old compact-size evidence and verified the retained fields after decode for every record. The sender detector still runs the original prompts. The measured reduction is in transmission bytes.
6. **Serialized complete trial packets.** Added the same application envelope to both arms. Measured actual emitted bytes for all 6,000 records with representative routing and timing values. The receiver reconstructed the scheduler input from the wire, checked framing and CRC, and verified job identity, scalars, retained fields and payload bytes. All 12 reused JPEG control payloads decoded to identical pixels in both arms. These are codec controls rather than paired reconstructions of every archived job.
7. **Reran the 15 saved development episodes.** Joined the saved arrivals, cached features and archived silver reference tiers by image identity and hash. FIFO reproduced the archived dispatch order and waiting penalty on every trace. SPI-first produced identical orders and metrics with both scene formats. Frozen LambdaMART changed order in all 15 traces. The replay used synthetic service and cached scene/SPI inputs. Dynamic relevance and novelty were unavailable in both arms. A packet regression separately checked preservation of nonzero scalar values.
8. **Refit one reduced-feature scheduler.** Trained a fixed histogram gradient boosting model using 67 retained input columns plus current wait. Used 3,170 training images after excluding development groups and exact hashes. The target was the increase in reference waiting cost for one more unit of delay. Used 40 iterations, at most 15 leaves, and seed 1729. No SPI classifier, detector, or codec training was performed for this change.

## No-COCO controlled comparison

| Policy | Mean waiting penalty, lower is better | Critical jobs within target |
| --- | ---: | ---: |
| FIFO | 5.822 | 13/75 |
| SPI-first, either scene format | 3.374 | 67/75 |
| Frozen LambdaMART, original scene | 3.609 | 37/75 |
| Frozen LambdaMART, no COCO | 3.416 | 38/75 |
| Reduced-feature refit, no COCO | 3.367 | 55/75 |

The refit's small aggregate improvement cost 12 Critical targets relative to SPI-first. It failed the guardrail, so SPI-first is the policy supported for this trial. Two complete runs agreed on shared packet and policy aggregates. The test runs passed 19 focused tests, six historical transport tests and 48 scheduler numerical tests. The numerical suite emitted one LightGBM feature-name warning.

The 75 proxy-Critical offers are pooled across the 15 original validation episodes. This replay evaluates waiting against archived urgency tiers. The 24-image native codec batches below evaluate visible road damage after real decoding and contain no independently labeled Critical cases. The original leaderboard reported LambdaMART at 3.2183, while a separate saved run reported 3.7036. Neither value replaces the 3.609 measured for the frozen model in the matched header replay above. [Original leaderboard](https://github.com/saltypal/SemanticSchedulerEnd2End/blob/43aa658b7e91f5ab574f3aba920454add918c085/experiments/results_analysis/runs/analysis-20261104/ANALYSIS.md) · [Separate saved run](https://github.com/saltypal/SemanticSchedulerEnd2End/blob/43aa658b7e91f5ab574f3aba920454add918c085/experiments/master_study/runs/repair-20261102/offline_report/REPORT.md)

## Codec and reader trials after the no-COCO replay

| Stage | Measured outcome | Decision and evidence |
| --- | --- | --- |
| 9. JPEG preflight and staged simulator | The pinned SQ-GAN bundle was initially unavailable. On 24 letterboxed JPEG controls, the fixed reader detected 3/12 positives on originals, 2/12 after resize and 0/12 after JPEG. All three paths had zero false reports. | Built a replay with separate capture, sender, arrival, queue, decode and decision times. Verified matched workloads and FIFO fallback when SPI is missing. [Preflight, simulator and receipts](sqgan_staged_replay.md) |
| 10. Pinned native SQ-GAN pilot | The supplied ZIP passed bundle and checkpoint checks. Six real 464 B SQ01 jobs decoded successfully. The reader found 1/3 positives on originals and 0/3 on reconstructions at 0.5. The cue was defined for five images and cost one byte and a median 1.05 ms to compute. | Native codec and three policy paths ran. No timely correct positive was delivered. [Six-image pilot and replay](native_sqgan_pilot.md) |
| 11. Native extension and rate control | All 18 additional 464 B jobs decoded. The reader found 2/9 positives on originals, 1/9 after resizing and 0/9 after SQ-GAN. A locked lower threshold still found 0/9. On the first six images, raising the image-latent fraction from 25% to 50% increased payload from 464 to 624 B and still yielded 0/3 post-decode positives. | The combined 24-image replay and original six scenarios reproduced. The cue and policy did not qualify. The 624 B control did not justify changing the default rate. [Extension, qualification and rate control](native_sqgan_extension.md) |
| 12. Larger frozen reader and matched replay | Fresh originals gave 4/12 true positives and 1/12 false reports. The same 24 messages decoded at 464 B and gave 0/12 true positives and 1/12 false reports. A separate source-folder-disjoint retry with a locked threshold yielded 3/9 true positives and 3/12 false reports. | All three policies delivered zero timely correct positives across six scenarios. Bursty overload p95 observed waits were 238 s for FIFO, 364 s for SPI-first and 570 s with the cue. A four-cycle stress run exercised unfinished work with 24 unique images and 96 offers. [Reader, figure and run manifest](strong_reader_trial.md) |
| 13. Frozen road-region reader | On 21 fresh originals, the crop reader found 9/9 positives and made 3/12 false reports. The full-frame comparator found 8/9 with 1/12 false reports. The frozen gate required 6/9 and at most 2/12 false reports. All scores and decisions reproduced. | The crop reader failed the source gate, and the paired CLI rejected that receipt. The full-frame comparator passed on this batch after failing on the earlier fresh batch. The paired run and conditional rate curve remain unopened for this reader. [Road-region protocol, results and hashes](road_region_reader.md) |
| 14. Event-separated obstruction reader | Acquired 86 verified CRASAR training-partition crops across four events. The centered reader passed Ida at 5/6 true positives and 0/12 false reports. It failed Harvey at 2/6 and 2/12 and calibration at 3/6 and 2/12. | Both variants failed qualification. The event-separated endpoint is visible obstruction. Paired codec and scheduling gates remain closed. [Event-separated methods and receipts](event_obstruction_reader.md) |
| 15. Multi-event reader and reviewed controls | Trained on Ian, Harvey and Ida. On fresh Mayfield and Michael crops, centered average precision rose from 0.634 to 0.757 and from 0.743 to 0.822 versus the frozen Ian-only reader. At the calibrated threshold, the centered reader found 4/6 and 3/6 positives and made 5/12 and 0/12 false reports, respectively. | The source gate failed. The paired 464 B run, cue replay and holdout remain closed. [Blind review, results and receipts](multievent_obstruction_reader.md) |

## Original five-step plan and team handoff

Four of the five planned development stages ran and have reproducible results. That is **80% by stage execution**, with each stage weighted equally. The fifth stage has a recorded no-go decision. Its qualified holdout evaluation is pending. The execution count includes experiments that failed their success criteria.

| Original step | Verified status | Result |
| --- | --- | --- |
| 1. Run a real SQ-GAN pilot | Complete | Measured per-image sender, payload, decoder and reader outcomes. The native decoder worked, while the road-damage reader failed at the tested rate. |
| 2. Improve the simulator | Engineering complete | Replaced fixed service with measured per-image service in matched controlled replays. Verified bursts, overload, backlog and failure handling. Field timing and Critical misses need suitable data. |
| 3. Test the encoder-derived cue | Development test complete. Cue unqualified. | Measured one header byte and median 3.01 ms of extra sender work in the larger-reader pilot. The predictive comparison could not qualify because it had no successful positive outcomes. |
| 4. Compare FIFO, SPI-first and SPI-first plus cue | Development replay complete. Benefit unverified. | All three returned zero timely correct positives. FIFO had the shortest p95 observed wait in scenarios with queueing. Critical preservation remains unmeasured. |
| 5. Make the holdout decision | No-go recorded. Holdout run pending. | A passing reader and an independently labeled multi-event cohort are needed before freezing and opening the holdout. |

The verified improvement in this branch is the **23.8% mean reduction in complete trial application header overhead**, with SPI-first decisions preserved on the archived replay. The measured-service simulator is an engineering improvement. A gain in timely correct evidence from the new cue or policy has yet to be demonstrated.

## Current decision and open work

- **Header.** The no-COCO application packet is verified with SPI-first and JPEG controls under the archived replay. Learned-policy decisions are sensitive to COCO removal. Sender savings from reducing detector prompts and link-layer timing remain unmeasured.
- **Codec and reader.** Native SQ-GAN inference and controlled replay are verified. Successful decoding at 464 B did not yield reliable road-damage decisions. The one-byte cue was measured, but it did not demonstrate predictive benefit beyond SPI. No post-pilot scheduler refit was performed.
- **Research gates.** The protected holdout remains closed. The team has no field receiver-arrival log or independently reviewed same-road linkage. The selected geographic development candidates came from one event. Independent Critical labels and a qualified multi-event cohort are still missing.
- **Next experiment.** Qualify a reader on independently labeled originals from separate events. If it passes, test paired reconstructions. A reconstruction-only failure triggers a byte-rate curve with an equal-byte control. Revisit scheduling efficacy only once the receiver reliably produces correct reports.
- **Local footprint work.** Footprint fixtures and related worktree changes are separate from the published trial commits. The cross-view section above records the limited local finding.

## Code and evidence

- [Living research log and initial development runs](README.md)
- [Scene-only engineering notes](no_coco.md) and [6,000-record byte audit](no_coco_wire_audit.json)
- [Complete application-header protocol, results and rerun command](no_coco_full_header.md)
- [Complete replay, class outcomes and refit receipt](no_coco_full_header_receipt.json)
- [Wire adapter](../../Evaluation/no_coco_scene_transport.py), [application packet adapter](../../Evaluation/trial_scheduler_packet.py) and [replay entry point](../../Evaluation/scheduler_trial_no_coco_replay.py)
