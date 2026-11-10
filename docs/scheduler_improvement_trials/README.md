# Scheduler improvement trials

This directory records changes tested against the original semantic scheduler. It does not redefine the scheduler task or replace its original benchmark.

## What changed and what the evidence says

- **Header:** the no-COCO transport format reduced the measured complete application header from 110.451 B to 84.163 B on the 6,000-record cache (23.8%). SPI-first produced the same dispatch orders and metrics in the 15 archived replay episodes. This is a wire-size result, not a scheduler-quality gain.
- **Simulator:** the trial replay separates message preparation, arrival, queue wait, decode/service, and decision outcomes; uses matched workloads and fixed deadlines; and reports late, failed, censored, and unfinished jobs. Arrivals remain controlled/synthetic because no receiver-arrival trace was supplied.
- **Policies:** SPI-first remains the supported trial baseline. The reduced-feature supervised ranker had waiting penalty 3.367 versus 3.374 for SPI-first, but met the proxy-Critical timing target for 55/75 jobs versus 67/75. It failed the predeclared guardrail and was not promoted. This was not an RL run.
- **SQ-GAN:** the existing codec was exercised on the LADI development images. At the tested settings, reader detections fell after reconstruction. The byte and decoding measurements are pilot evidence only; they do not show a scheduling gain.
- **Holdout:** no final holdout claim is supported by these development replays. No real receiver-arrival log or independently labeled Critical outcomes were available.

## Original scheduler context

The original project has a larger baseline evaluation with FIFO, deadline, semantic heuristic, learned ranking, and DQN policies. The development branch trials above do not replace those results. Static nDCG is a ranking metric for a complete, fixed ordering; it is not the dynamic scheduler's training reward. In live/staggered replay, report delivery utility, waiting, deadline outcomes, failures, and class-specific service separately.

The original 15-episode replay and 75 proxy-Critical offers are summarized in [the consolidated work record](work_record.md). See [the no-COCO wire audit](no_coco_wire_audit.json), [the full-header protocol](no_coco_full_header.md), [the matched replay receipt](no_coco_full_header_receipt.json), and [the SQ-GAN pilot](native_sqgan_pilot.md).

## Scope correction

CRASAR road-obstruction data, its reader, labels, and related ablations were an unrelated extension and have been removed from the current working tree. Their true-positive/false-report counts must not be cited as original-scheduler results. The original scheduler source, its existing benchmark data, policies, and RL reward code were not changed for this cleanup. Prior Git commits still preserve the historical trial files; the cleanup is currently an uncommitted change on `scheduler_improvement_trials`.
