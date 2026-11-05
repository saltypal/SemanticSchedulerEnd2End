# Version 9 completed-training and development-results audit

No additional training submitted. At least one completed seed is sufficient under the user's latest decision.

Completion here means the run recorded completed training and development evaluation, not convergence or independent final evaluation.

## Completed learned methods

| Family | Algorithm | Completed seeds |
|---|---|---|
| D | linear_choice | 1729, 2718 |
| D | mlp_choice | 1729, 2718 |
| D | delay_huber | 1729, 2718 |
| D | ranknet | 1729, 2718 |
| D | lambdamart | 1729, 2718 |
| D | priority_classifier | 1729, 2718 |
| D | deepsets_ranknet | 1729, 2718 |
| D | set_transformer_ranknet | 1729, 2718 |
| D | listmle | 1729, 2718 |
| D | masked_ppo | 1729, 2718 |
| D | masked_double_dqn | 1729, 2718 |
| D | td_per | 1729 |
| D | ranknet_warmstart_ppo | 1729 |
| D | temporal_deepsets_ranknet | 1729 |
| feedback | vlm_replay | 1729, 2718, 31415 |

## Shared D validation: fixed first seed 1729

J_D is a normalized waiting penalty, not milliseconds. Lower is better. The saved validation score equals negative mean J_D. Fifteen validation traces; synthetic timing and silver priority references. No best-seed selection is used below.

| Method | Mean J_D |
|---|---:|
| spi_fifo | 3.374083 |
| spi_novelty | 3.434858 |
| ranknet | 3.630309 |
| deepsets_ranknet | 3.647492 |
| delay_huber | 3.671407 |
| lambdamart | 3.703640 |
| mlp_choice | 3.705926 |
| set_transformer_ranknet | 3.726756 |
| ranknet_warmstart_ppo | 3.746413 |
| temporal_deepsets_ranknet | 3.770175 |
| priority_classifier | 3.907849 |
| linear_choice | 4.159368 |
| listmle | 4.179957 |
| td_per | 4.713065 |
| relevance | 4.920339 |
| masked_double_dqn | 4.962876 |
| predicted_edf | 5.026631 |
| masked_ppo | 5.222276 |
| round_robin | 5.805900 |
| fifo | 5.822023 |
| random | 5.919610 |
| novelty | 6.191743 |

SPI-first cost 3.374083; FIFO 5.822023; RankNet 3.630309. RankNet reduces penalty versus FIFO by 37.65%, but has 7.59% greater penalty than SPI-first.
Seed 2718 LambdaMART score is -3.498117; it still does not beat SPI-first. This secondary result is not a final-test superiority claim.

## Blocked teacher methods

P rankers: 3 strict training and 4 strict validation preferences; required 16 and 8. VisionPrefer: 0 training and 2 validation strict preferences. No teacher-trained ranker completed.
RL-VLM-F-inspired learned-reward RL: blocked by insufficient qualified prefix support/reward validation. VLM-guided replay did complete 20,000 decisions for each of three seeds.
Feedback-run scores use different evaluation workloads from D; their numerical magnitudes cannot be ranked against D.

## Timeout omissions and retrieval boundary

Main seed 1729 has all 14 non-teacher methods completed; seed 2718 has 11 completed and TD-PER interrupted, with warmstarted PPO and temporal ranker not started. Main seed 31415 was not started.
The rate-study stage was after main training and was not reached. Master final report/evaluation aggregation did not run.
Raw status files and baseline summary recovered. Full checkpoint/artifact retrieval remains in progress; successful resume and portable inference have not been newly validated in this audit.
Published final independent emergency evaluation remains unavailable. No journal-level superiority conclusion follows from this development run.


Scientific follow-up: see diagnostic_audit/AUDIT.md. All24 replay judgments were neutral and replay guide coefficient norm was0. The optimized replay arm is50/50 uniform/TD replay with constant VLM multiplier, not evidence of informative VLM guidance. Supervised/RL delay data actually contain only CrisisMMD.
