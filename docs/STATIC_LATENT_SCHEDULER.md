# Whole image latent scheduling

This path selects complete image latent payloads. It has no UE identity, UE
queue, radio allocation or UE fairness objective. The previous UE simulation
remains a separate comparison track.

## Owners and flow

**EncDec** encodes once, saves tensors and measures channel demand; its existing
`integration/scheduler_replay.py` transports and reconstructs saved payloads.
**SemParam** extracts image/task CLIP cosine relevance. Novelty is
`(1 - max cosine to confirmed reconstructed embeddings) / 2`; absent history
remains unavailable. Task importance comes from an explicit application input
or separately validated prediction. Urgency comes from deadline slack.
**Scheduler** selects from immutable latent candidates. **INFRA** owns time,
eligibility, lifecycle, receiver history, logging and evaluation.

`run_latent_schedule(..., delivery_handler=transport)` invokes transport with
the selected complete `LatentJob` only after selection. Transport returns
`DeliveryEvidence`; identity and embedding shape are checked before feedback.
This supports actual payload execution without exposing future outcomes.
Use cached evidence instead for repeated fixed-channel comparisons.
`EncDec.integration.latent_transport.CachedSwinLatentTransport` implements the
live SwinJSCC callback: load the model once, fetch the selected cached tensor,
check checkpoint and demand, transmit, decode, persist reconstruction and
measured wall time. Optional receiver/task evaluators see the reconstruction.
No task score is invented when those evaluators are absent. CUDA/checkpoint
execution is separate from functional fixture validation.
`scripts/scheduler/replay_static_latent_schedule.py --help` exposes the existing codec's
offline replay for a saved order with demand/checkpoint identity checks. That
offline replay alone does not create online receiver feedback.

Static means every job arrives at zero. Confirmed receiver history can still
evolve after successful delivery. Dynamic mode accepts explicit job arrivals
through this same engine, without UEs. Only reconstructed receiver embeddings
from confirmed successful deliveries update history. Original source embeddings
are not substituted for missing reconstruction evidence.

## Policies and service

FIFO, shortest service first, earliest deadline first, priority FIFO, priority
EDF and semantic selection share the same inputs and feasibility rules.
Lower class integer means higher priority. Priority policies first gate on the
highest eligible supplied/predicted class. Probability inputs use the ordinal
posterior median. Sources are required; missing classes fail explicitly.

Within that gate, semantic selection divides relevance, novelty, importance and
urgency factors by service time. The first three factors have a 0.25 floor;
urgency is `1 + service / max(service, slack + service)`. Missing inputs stay
null in traces, with explicit selection defaults in `semantic_score`.
This is a transparent heuristic, not measured value of information or a novelty
claim.

Service is non-preemptive: payload demand divided by service rate, optionally
rounded to a quantum. A job that cannot finish before its deadline is marked
deadline-infeasible. A job that cannot fit the horizon is censored. Strict
priority can increase lower-class waiting; per-class wait metrics expose it.
Tensor storage bytes, digital bits and complex channel uses stay distinct.

## Run

From the repository root:

```powershell
.venv\Scripts\python.exe INFRA/latent/cli.py --batch Scheduler/config/static_latent_demo.json
.venv\Scripts\python.exe -m pytest INFRA/tests/test_latent_scheduler.py -q
.venv\Scripts\python.exe INFRA/latent/cli.py --cached-run downloaded/train-final-cli/outputs/20260929T021341Z-d3338ef4
```

The demo is a synthetic functional fixture. The cache adapter uses saved
classifier predictions, not evaluation classes. Use `--payload-costs` to supply
a matching measured codec profile. Simulated costs remain marked simulated.
Artifacts live in `results/Scheduler/<run-id>`: config, commit/dirty state,
source digest, batch hash, full candidate/decision traces, terminal outcomes,
metrics and flushed JSONL progress logs.

The portable notebook runs cache-only scheduling by default. Keep substantive
model training and GPU extraction/replay on Kaggle through its CLI. The existing
DQN remains a separate legacy experiment; it is not a validated latent-policy
replacement. Future RL must choose a masked latent candidate from this same
snapshot, use task-grounded reward, and beat these baselines on held-out tasks
and seeds.

## Evaluation limits

Reference classes are evaluator-only. Complete static schedules report nDCG
and nDCG@5; eligible-candidate pair agreement covers truncated/dynamic runs.
Predicted priority gate violations measure policy correctness, not classifier
accuracy. Deadline success, resource use, failures/censoring, class wait and
host policy compute time stay separate.

Without confirmed task quality, task utility remains unavailable. When quality
is explicitly supplied, evidence-quality gain and integrated coverage count
only positive improvements within an evidence group, preventing duplicate
coverage credit. This is a declared proxy, not universal downstream task loss.
PSNR/SSIM do not establish task accuracy. Existing prototype labels are not an
independently reviewed general-task benchmark. Cached outcomes assume a fixed
channel profile; changed SNR/rate/channel requires new decoder evidence.
