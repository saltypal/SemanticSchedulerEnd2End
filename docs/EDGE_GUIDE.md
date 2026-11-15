# Edge decoder scheduler: guide

This guide covers the edge-scheduling study end to end: what the system does, how the code is organised, how the
results were produced, and where to find them. The paper is in `docs/paper/DRAFT_PAPER.md`; every number it reports is
in `docs/RESULTS_ALL.md`, generated from the raw files in `results/edge_final/`.

## 1. The problem

Generative semantic codecs such as SQ-GAN send an image as a few hundred bytes, but the receiver pays for the
generation. One decode takes about 0.74 s on a T4 GPU. In a disaster, many phones, drones and vehicles upload at once,
and an edge pool of GPU decoders cannot decode everything before the deadlines. The scheduler decides which received
image to decode next, and which to drop, from a 64-byte header alone.

The score is CTR@D: the fraction of human-labelled Critical images that are decoded before their Critical deadline.

## 2. What the system is made of

| Part | What it does | Where |
|---|---|---|
| UE payload | SQ-GAN v1 codec at masking fraction m = 0.25: 464 bytes per image | `Codec/edge/`, `parameters/edge/codec_table.py` |
| Header | 64 bytes: SPI urgency posterior (4 x u16), mission relevance, SimHash key, pHash key, STS descriptor, CRC | `parameters/edge/header64.py` (layout v2) |
| SPI | CLIP ViT-B/32 logistic head, class-balanced, temperature-calibrated | `parameters/edge/simtable.py`, `spi_head` |
| STS fusion | receiver model fusing SPI and STS into one tier belief (D19) | `parameters/edge/sts_fusion.py` |
| Simulator | event-driven edge with c decoder workers, measured batched latency, trace arrivals | `Evaluation/edge/env.py`, `run_sweep.py`, `arrivals.py` |
| Uplink (sensitivity) | ideal, AWGN, or Rayleigh block fading with HARQ | `Evaluation/edge/regimes.py` (`apply_link`) |
| Policies | header-aware value-rate batching, SV-DPP, baselines, learners | `Scheduler/edge/`, `Scheduler/training/edge/` |
| Live check | the same scheduler code driving real SQ-GAN decodes on two GPUs | `Evaluation/edge/live_check.py` |

Settings live in `config/edge/*.yaml`. The tuned SV-DPP values are in `config/edge/sim.yaml`
(V 0.1, purge 0.99, relevance beta 1.0, tie tolerance 0.5).

## 3. The schedulers

- **Value-rate batching (`spi_edf_ratio`)**: order by SPI Critical probability and soft-deadline slack. Jobs that
  cannot meet any likely hard deadline move to the back. For each profiled batch size B, compute the expected on-time
  value per second of decoding, then decode the best B. Code: `Scheduler/edge/batching.py`, `ratio_batch_size`.
- **Same rule without the header (`edf_ratio`)**: deadline order and a uniform tier prior. This control isolates the
  value of the header.
- **SV-DPP**: drift-plus-penalty with per-tier debt queues. Per decision it scores each job by a value term and a debt
  term, then applies renewal-ratio batching. Code: `Scheduler/edge/sv_dpp.py`.
- **Baselines**: FIFO, EDF on SPI-predicted deadlines, SPI-first + EDF, gen-c-mu/h with abandonment, Whittle with
  abandonment, and the true-tier SV-DPP reference (`genie_sv_dpp`, a reference, not an upper bound).
- **Learners**: RankNet and LambdaMART (listwise rankers), BC (imitation of SV-DPP), PPO and PPO-BC, and the SCOPE
  family (SCOPE-PPO, SCOPE-LTR, SCOPE-DQN). SCOPE is constrained RL with an imitation term toward the SV-DPP teacher,
  an adaptive teacher weight (TGRL-style) and dual updates of the per-tier penalties. Code: `Scheduler/training/edge/`.
- **Weak variants (appendix only)**: DQN, BCEdge, SemCover (its learned and analytic versions), Whittle.

All policies see only header quantities. The true tier and the duplicate group are used only for scoring, and by
the genie reference.

## 4. How a run works

1. The simulator takes the event split (train, validation or test), the arrival trace and the load rho. The CrisisMMD
   images arrive in tweet-time order, compressed so that the offered load equals rho. Everyday images are
   Poisson-interleaved. Each realisation uses a different permutation and seed.
2. Each worker, when free, asks the policy for a batch. The policy sees the queue headers, the ages and the latency
   profile. It returns the batch and any purges.
3. The simulator samples the decode time of that batch size from the measured T4 distribution and schedules the
   completion. Completions update the delivered-image memory and the debt queues.
4. CTR@D and the other metrics are computed against the true tiers. Confidence intervals come from a bootstrap over the
   realisations.

## 5. Splits and data

- Pool: 27,168 images (CrisisMMD v2 with human labels, Places365 validation sample, COCO people). 27,044 remain after
  removing 124 near-duplicates across splits (pHash <= 8 or SimHash <= 1).
- Event-disjoint splits: train = Harvey, Irma, Iraq-Iran, Sri Lanka; validation = California wildfires;
  test = Hurricane Maria and the Mexico earthquake.
- Published on Kaggle: `satyapaladuguplus/scheduler-dataset-v1-simtable` (the simulator table, latency profile and
  STS fusion model) and `satyapaladuguplus/scheduler-dataset-v1-images`.

## 6. Reproducing the results

The Kaggle notebooks are thin: each clones `main`, installs, and calls one package entry point. The builders are in
`scripts/edge/` (for example `push_sweeps_today.py`, `push_learned_sweeps.py`, `push_one.py`). They push through the
Windows Kaggle CLI.

Running a sweep locally, for example the validation main table at 2 decoders:

```bash
python -m Evaluation.edge.run_sweep --table <simtable_v1.jsonl> --profile <latency_v1.json> --split val \
  --policies sv_dpp spi_edf_ratio edf edf_ratio fifo --workers 1 2 4 8 --rhos 0.8 1.2 1.5 \
  --realisations 5 --arrival trace --out results/val_main.json
```

Decision traces and the replay files used for the audit:

```bash
python -m Evaluation.edge.decision_trace --table <simtable_v1.jsonl> --profile <latency_v1.json> --split val \
  --policies sv_dpp spi_edf_ratio --workers 2 --rho 1.2 --subsample 300 --seed 0 --arrival trace --out audit/
```

Tests (CPU only):

```bash
python -m pytest -q Scheduler/tests/test_edge_*.py
```

## 7. Where the results are

| What | File |
|---|---|
| Every result as tables | `docs/RESULTS_ALL.md` |
| Full-split sweeps (validation and test, Kaggle) | `results/edge_final/kaggle_full/*.json` |
| Subsample sweeps (1,000 and 2,000 images) | `results/edge_final/subsample/*.json` |
| Uplink sensitivity | `results/edge_final/link/*.json` |
| SV-DPP tuning grid | `results/edge_final/tuning/sv_dpp_beta_tie_grid.csv` |
| Semantic audit | `results/edge_final/audit/AUDIT_REPORT.md` |
| Paper draft | `docs/paper/DRAFT_PAPER.md` |
| Handover for co-authors | `docs/paper/HANDOVER_FOR_PAPER_WRITING.md` |

## 8. Decisions and history

- `docs/CHECKLIST.md`: locked decisions D1-D20 and the run log.
- `docs/reviews/FINAL_PLAN_V3.md`: the original design.
- `docs/reviews/FIX_PLAN_2026-11-13.md`: the fixes after the first learner wave.
- `docs/reviews/PLAN_2026-11-14_TO_8PM.md`: the final plan for the reporting round.
- `parameters/JOURNAL.md` and `parameters/edge_kaggle_runs.md`: dated records of runs.

## 9. Known limitations

- The codec was fine-tuned for under one epoch and is out of domain for disaster photos (mean LPIPS 0.466).
- SPI recall on the High tier is low (0.08). Label noise in CrisisMMD is visible in a few images.
- SV-DPP's result depends on the arrival trace on small subsets, while the value-rate rule is more stable.
- The live two-GPU check is implemented and smoke-tested with a fake decoder. Its hardware run is not part of the
  reported results.
