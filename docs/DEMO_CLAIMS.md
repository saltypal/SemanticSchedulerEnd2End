# Faculty demo claim ledger

| Category | Defensible statement | Evidence or gate |
|---|---|---|
| Implemented and unit tested | Track-A feature records, success-only delivered history, 24-byte V2 header, deterministic whole-latent simulator, classical and learned policy code | `pytest SemParam/tests INFRA/tests Scheduler/tests` |
| Implemented and measured on one image | SwinJSCC checkpoint encode/decode and AWGN reconstruction at 10 dB in the compatible model environment | Local image `ladi-00806b9ef7807225`, checkpoint SHA-256 `8e9d53d8…`, 1024×1024/rate 96; `results/scheduler-codec-smoke/` records 196,608 complex channel uses, 27.44 dB PSNR and 0.8533 SSIM for **one image only** |
| Measured payload cache | All 60 draft images were encoded once with one checkpoint; every fixed-rate job measured 196,608 complex channel uses | `results/ladi-curation/v1/payload/payload_costs.csv`; SJF has no cost distinction on this cache |
| Implemented but not yet research-evaluated | Three-seed DQN training with a 40-minute cap per seed; same-run policy comparison and a separate evaluation notebook | Requires the prototype-fixed 60-image manifest, persistent artifacts and an actual run |
| Simulated | Five UEs, arrivals, non-preemptive service, deadlines, queue overflow, channel success | `INFRA/semantic_streams.py` and `INFRA/event_simulation.py` |
| Proposed | Robust over-air control header, real 5G MAC scheduling/resource blocks, distributed receiver knowledge synchronization | No physical-link claim |

The fixed prototype manifest deliberately bypasses human review and gives P7=1.
Do not present its CLIP-nominated pairs or its class priority rubric as universally correct ground
truth. Do not present DQN superiority, image-disjoint performance, or codec
quality aggregates until those experiments have actually run.
The privileged marginal-utility comparator sees rubric priority and evidence
groups; it is a one-step reference, not a deployable policy or mathematical
upper bound.
