# No-COCO scene header trial

See [complete application-header verification](no_coco_full_header.md) for the 15-trace packet and refit results. The nine-trace table below is the earlier scene-only replay.

The old scene block included slots for 80 standard COCO classes. The new opt-in block transmits the five project objects (building, bridge, utility pole, traffic sign and animal) and eight regions (road, water, debris, smoke, flame, snow, ground and grass). It also transmits the detector status and availability information. The receiver expands omitted slots as unavailable, which preserves the historical feature layout for a controlled inference comparison.

## Measured size

| Scene block | Minimum | Median | Mean | p95 | Observed maximum | Dense maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing compact v3 | 29 B | 33 B | 34.45 B | 45 B | 73 B | 307 B |
| No-COCO trial v4 | 8 B | 8 B | 8.16 B | 8 B | 16 B | 30 B |

The mean is 76.3% smaller, saving 26.29 bytes per cached scene block. This measures the actual encoders on 6,000 quantized records reconstructed from the tracked feature cache. Round-trip and retained-field checks passed for every record. The legacy byte counts also agree with the historical cache receipt. The audit binds the exact scene registry recorded by the archived development model. Both endpoints must configure the same producer and ordered vocabulary profile for actual use.

The scene block includes its own four-byte version, flags and profile prefix. Routing, SPI, relevance, novelty, timestamps, optional camera footprint, transport framing and codec payload are outside these numbers. The proposed 64-byte report layout describes a separate design. The no-COCO scene format can be tried independently without changing historical header readers, model weights or published runs. The original detector still produces the 93-slot record in this transport-only trial.

## What to rerun

1. **Transport and receiver checks.** Enable v4 at both endpoints of an isolated replay. Compare against v3 on the same arrived jobs. Existing v3 clients reject v4 explicitly. The v4 decoder expands into a 93-slot record with zero values and unavailable masks for COCO slots.
2. **Matched development ablation.** On the same trace identities and labels, compare the original model on the original header, the same model on v4 expanded headers, and the unchanged SPI-first control on both. Compare urgency penalty and class outcomes as well as tail waits and unfinished jobs. A model that accepts 387 inputs can still change its decisions when 320 COCO-related values and masks disappear.
3. **Train selectively if needed.** FIFO and SPI-first do not need retraining for scene-field removal. The SPI classifier and image codec have separate inputs and do not need retraining for this wire change. Scene-dependent learned rankers need a development replay before reuse. If their masked-input performance regresses, or if the model contract itself is reduced from 387 to 67 features, refit the affected scheduler ranker and its normalization on the reduced training contract. Start with the strongest prior learned comparator and the simple priority rule. Reuse cached retained scene measurements for this ablation.
4. **Measure detector compute as a separate change.** If the goal also includes fewer detector prompts at the sender, run the reduced detector on the same development images. Recompute the features and compare retained detections, since prompt and post-processing changes can affect results. No detector-time savings are inferred from this transport result.

The byte audit does not estimate scheduling quality. It does not read urgency labels or open the protected scheduler holdout.

``## Development replay result

The archived LambdaMART export was evaluated on both the original scene values and the values decoded from v4. The replay recovered nine complete saved development traces by joining archived FIFO arrivals, cached image features and the saved silver reference tiers. The original FIFO dispatch order and waiting penalty reproduced on every included trace. Six other traces lacked at least one saved reference in the FIFO events. Relevance and novelty were unavailable in both arms, so this does not reproduce the complete historical LambdaMART feature stream.

| Policy | Mean waiting penalty | Mean p95 capped wait | Critical within target |
| --- | ---: | ---: | ---: |
| FIFO | 2.915 | 13.08 | See per-episode receipt |
| SPI-first, either scene format | 1.592 | 14.84 | 33/36 |
| Frozen LambdaMART, original scene | 1.760 | 14.85 | 25/36 |
| Frozen LambdaMART, no COCO | 1.598 | 14.84 | 26/36 |

The masked model changed dispatch order in all nine episodes and lowered its mean waiting penalty by 9.2%. SPI-first remained identical across headers and had the lowest penalty and best Critical target attainment among these arms. The mean unfinished count was 1.78 in each arm. Time is in synthetic replay units. This result supports keeping SPI-first as the immediate comparator and testing the full reduced-scene feature stream before committing compute to a new ranker. No retraining, threshold changes or protected test access occurred.

## Five-day follow-through

The local branch already contained in-progress footprint work when the header task started. I verified its tests and regenerated its constructed queue receipt byte-for-byte. The archived six-crop footprint receipt matches 3/3 candidates with a 24-hour tolerance and 0/3 with two hours, because the inputs only provide a capture date. Neither tolerance matched any of the eight different-location controls. These are selected crops from one event. In constructed replays the footprint rule helps when an overlap is a true repeat and loses a timely observation when a nearby road overlaps the view or the condition changes rapidly. These failure cases keep the natural-traffic claim and protected holdout gated. The header trial commits exclude the pre-existing footprint work.

`powershell
python -m Evaluation.scheduler_trial_no_coco_audit 7a1e4b164360007642615b0705fd391b137b8da7 <new-audit-receipt.json>
python -m Evaluation.scheduler_trial_no_coco_replay 7a1e4b164360007642615b0705fd391b137b8da7 <new-replay-receipt.json>
python -m pytest Scheduler/tests/test_no_coco_scene_transport.py -q
```
