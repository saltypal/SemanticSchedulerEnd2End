# Scheduler improvement trials — consolidated work record

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

## Final controlled comparison

| Policy | Mean waiting penalty, lower is better | Critical jobs within target |
| --- | ---: | ---: |
| FIFO | 5.822 | 13/75 |
| SPI-first, either scene format | 3.374 | 67/75 |
| Frozen LambdaMART, original scene | 3.609 | 37/75 |
| Frozen LambdaMART, no COCO | 3.416 | 38/75 |
| Reduced-feature refit, no COCO | 3.367 | 55/75 |

The refit's small aggregate improvement cost 12 Critical targets relative to SPI-first. It failed the guardrail, so SPI-first is the policy supported for this trial. Two complete runs agreed on shared packet and policy aggregates. The test runs passed 19 focused tests, six historical transport tests and 48 scheduler numerical tests. The numerical suite emitted one LightGBM feature-name warning.

## Current boundary

The no-COCO application packet is verified with SPI-first and JPEG controls under the tested replay. Learned-policy decisions are sensitive to COCO removal. The protected scheduler holdout stayed closed. SQ-GAN inference, live radio/link-layer traffic, sender savings from reduced detector prompts, and a multi-event natural-repeat claim remain unverified. The team had no receiver-arrival log or independently reviewed same-road linkage. The geographic development candidates came from one event, so they did not qualify a holdout.

## Code and evidence

- [Living research log and initial development runs](README.md)
- [Scene-only engineering notes](no_coco.md) and [6,000-record byte audit](no_coco_wire_audit.json)
- [Complete application-header protocol, results and rerun command](no_coco_full_header.md)
- [Complete replay, class outcomes and refit receipt](no_coco_full_header_receipt.json)
- [Wire adapter](../../Evaluation/no_coco_scene_transport.py), [application packet adapter](../../Evaluation/trial_scheduler_packet.py) and [replay entry point](../../Evaluation/scheduler_trial_no_coco_replay.py)
