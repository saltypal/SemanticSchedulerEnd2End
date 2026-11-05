# Selected scheduler codec: SQ-GAN digital payload

The active scheduler codec is configured in Codec/config/scheduler_codec.json and created through build_scheduler_codec() in Codec/integration/sqgan_payload.py.

Bundle:
SemanticLossless/Semantic_Lossless/Results/team_downloads/2/satyapaladuguplus__semantic-notebook-2/0b680269ce1a4e2c83d9fe5159b3af1c/published_output/models/teammate_2_models/SQ-GAN/20261028T130100Z-eea4f3aa

The full bundle was checksum-verified. The ten inference components were strictly restored from core.* in the fine-tuned last.pt, rather than left at their pretrained constructor weights.
last.pt SHA256:706396402d86e84276393ce3fe39f0b0b1b545ff024c0ecae3b8ab72d22f3d6d.

Transmitter: RGB1x3x256x512 + paired20-class semantic map -> SQ01 bytes.
The frozen segmentation teacher can generate the map using its state from the same fine-tuned bundle.
Receiver: complete SQ01 bytes -> reconstruction. No original RGB, original semantic map, SNR or analog I/Q tensor is supplied to decode.
The external inference implementation and experimental SQ01 format are reused from the existing Semantic_Lossless codebase. Codebooks/checkpoint are pre-shared setup assets.

Real cached image pilot:464 payload bytes (header accounting separate), finite reconstruction1x3x256x512, corrupt CRC rejected.
Single CPU measurements, two threads: teacher27.306s (includes cold teacher setup), encode22.687s, decode41.073s.
This is not a deployment latency distribution or image-quality/generalization benchmark.

Validation:12 passed in166.02s, exit0, no skips in the selected SQ-GAN + receiver lifecycle checks.
Tests include real strict checkpoint restoration, packet roundtrip, corruption rejection, opaque digital message dispatch, successful completion, failed completion and unchanged successful history on corruption.

Legacy SwinJSCC test source is preserved in legacy_swinjscc_integration.py.txt. SwinJSCC model artifacts were not deleted or modified.
Only extra inference dependency installed in existing WSL project environment:einops0.8.1; recorded in Codec/environment/sqgan_scheduler.txt.
No model training or Kaggle submission occurred. Scheduler simulation timing is not automatically replaced by the measured CPU decode time.

Limits: SQ-GAN is a lossy semantic/perceptual source codec; unchanged bytes do not mean exact pixels. The SQ01 framing is the existing experimental adapter, not the paper entropy coder. CRC detects errors; it does not correct them. No radio/fading/noise/retry test or lossless-reconstruction claim is made.
The selected export is a Cityscapes-oriented adaptation, with native convergence explicitly unestablished in its training metadata.
