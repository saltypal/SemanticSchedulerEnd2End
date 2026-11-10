# Complete application-header verification

Both arms use the same opt-in trial envelope and the same existing received-feature scheduler. The sender serializes job/source/session identity, sequence, arrival and capture times, allowed delay, SPI scores, relevance and novelty, availability, and scene and payload lengths. A CRC covers the envelope and payload. The receiver validates framing and reconstructs the scheduler input from bytes. The only format difference is the scene block. This envelope is a trial implementation. The archived scheduler did not serialize a combined application header for these replays.

| Overhead across 6,000 cached records | Original scene | No COCO |
| --- | ---: | ---: |
| Complete application header and CRC, mean | 110.45 B | 84.16 B |
| Complete application header and CRC, p95 | 121 B | 84 B |
| Scene block only, mean | 34.45 B | 8.16 B |

The total falls by 26.29 bytes per record, or 23.8% on average. JPEG payload bytes and lower-layer framing are outside the table. Metadata for the 6,000-record size sweep is representative. The separate 15-episode dispatch replay uses archived development arrivals and cached feature rows. Each packet's identity, scalars, retained fields and payload matched the intended values at ingress. The old and new formats produced identical decoded pixels for all 12 JPEG payloads. Those payloads are codec controls reused across jobs. The experiment does not measure SQ-GAN inference or physical radio performance.

All 15 original FIFO orders and waiting penalties reproduced. SPI-first had identical orders and metrics across the two formats. Frozen LambdaMART changed order in all 15 episodes. The experiment uses model-generated silver tiers and synthetic service. Historical dynamic relevance and novelty values are unavailable in these traces. Nonzero scalar preservation is covered separately by the packet regression test. The protected holdout was not evaluated.

## Matched scheduler results

| Policy | Mean waiting penalty | Critical jobs within target |
| --- | ---: | ---: |
| FIFO | 5.822 | 13/75 |
| SPI-first, either header | 3.374 | 67/75 |
| Frozen LambdaMART, original scene | 3.609 | 37/75 |
| Frozen LambdaMART, no COCO | 3.416 | 38/75 |
| One reduced-feature refit, no COCO | 3.367 | 55/75 |

The refit used a fixed histogram gradient boosting model with 40 iterations, at most 15 leaves, seed 1729, 67 retained header features and current wait. It fit the one-unit reference postponement cost at ten fixed wait values per image. Training used 3,170 cached training images after excluding exact image hashes and groups present in the development traces. The refit narrowly improved the penalty versus SPI-first and missed its Critical-class guardrail by 12 jobs. Keep SPI-first for the trial. The SPI classifier and JPEG codec required no retraining. Detector prompts still use the existing producer contract.

Two complete runs agreed exactly on the shared header and policy aggregates. The focused packet and trial suite passed 19 tests, the historical compact-transport suite passed six, and the scheduler numerical suite passed 48 with one LightGBM feature-name warning.

```powershell
python -m Evaluation.scheduler_trial_no_coco_replay 7a1e4b164360007642615b0705fd391b137b8da7 <new-receipt.json> --images <development-jpeg-directory>
```
