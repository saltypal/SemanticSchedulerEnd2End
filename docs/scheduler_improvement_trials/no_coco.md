# No-COCO scene header trial

The old scene block included slots for 80 standard COCO classes. The new opt-in block transmits the five project objects (building, bridge, utility pole, traffic sign and animal) and eight regions (road, water, debris, smoke, flame, snow, ground and grass). It also transmits the detector status and availability information. The receiver expands omitted slots as unavailable, which preserves the historical feature layout for a controlled inference comparison.

## Measured size

| Scene block | Minimum | Median | Mean | p95 | Observed maximum | Dense maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing compact v3 | 29 B | 33 B | 34.45 B | 45 B | 73 B | 307 B |
| No-COCO trial v4 | 8 B | 8 B | 8.16 B | 8 B | 16 B | 30 B |

The mean is 76.3% smaller, saving 26.29 bytes per cached scene block. This measures the actual encoders on 6,000 quantized records reconstructed from the tracked feature cache. Round-trip and retained-field checks passed for every record. The legacy byte counts also agree with the historical cache receipt. The audit uses a placeholder registry hash because the cache does not encode that hash and it contributes no bytes in either compact format. Both endpoints must configure the same producer and ordered vocabulary profile for actual use.

The scene block includes its own four-byte version, flags and profile prefix. Routing, SPI, relevance, novelty, timestamps, optional camera footprint, transport framing and codec payload are outside these numbers. The proposed 64-byte report layout describes a separate design. The no-COCO scene format can be tried independently without changing historical header readers, model weights or published runs. The original detector still produces the 93-slot record in this transport-only trial.

## What to rerun

1. **Transport and receiver checks.** Enable v4 at both endpoints of an isolated replay. Compare against v3 on the same arrived jobs. Existing v3 clients reject v4 explicitly. The v4 decoder expands into a 93-slot record with zero values and unavailable masks for COCO slots.
2. **Matched development ablation.** On the same trace identities and labels, compare the original model on the original header, the same model on v4 expanded headers, and the unchanged SPI-first control on both. Compare urgency penalty and class outcomes as well as tail waits and unfinished jobs. A model that accepts 387 inputs can still change its decisions when 320 COCO-related values and masks disappear.
3. **Train selectively if needed.** FIFO and SPI-first do not need retraining for scene-field removal. The SPI classifier and image codec have separate inputs and do not need retraining for this wire change. Scene-dependent learned rankers need a development replay before reuse. If their masked-input performance regresses, or if the model contract itself is reduced from 387 to 67 features, refit the affected scheduler ranker and its normalization on the reduced training contract. Start with the strongest prior learned comparator and the simple priority rule. Reuse cached retained scene measurements for this ablation.
4. **Measure detector compute as a separate change.** If the goal also includes fewer detector prompts at the sender, run the reduced detector on the same development images. Recompute the features and compare retained detections, since prompt and post-processing changes can affect results. No detector-time savings are inferred from this transport result.

The byte audit does not estimate scheduling quality. It does not read urgency labels or open the protected scheduler holdout.

```powershell
python -m Evaluation.scheduler_trial_no_coco_audit 7a1e4b164360007642615b0705fd391b137b8da7 <new-audit-receipt.json>
python -m pytest Scheduler/tests/test_no_coco_scene_transport.py -q
```
