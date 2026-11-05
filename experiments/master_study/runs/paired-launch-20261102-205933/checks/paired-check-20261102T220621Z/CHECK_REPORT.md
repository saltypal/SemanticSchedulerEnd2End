# Paired Kaggle run check — 3 November 2026

## Remote status

- WSL `satyapaladuguplus/semantic-decoder-scheduler-20261030`, version 12, run `wsl-20261102T162419Z`: Kaggle status **COMPLETE**. Its `RUN_STATUS.json` is **partial_or_blocked** because teacher and rate-study stages failed; the supervised methods completed development-only evaluation.
- Windows `satyapaladugu/semantic-decoder-scheduler-rl`, version 1, run `windows-20261102T154257Z`: Kaggle status **COMPLETE**. `RUN_STATUS.json` reports `complete_method_attempts`, with no incomplete jobs or stage failures. This is not a scientific final evaluation.
- Captured runtime totals: WSL 10,572 s (~2 h 56 m); Windows 9,933 s (~2 h 46 m). Neither hit the 12-hour limit.

## Methods

- **14 unique non-teacher models completed development evaluation:** ten WSL supervised/ranking methods (`linear_choice`, `mlp_choice`, `delay_huber`, `ranknet`, `lambdamart`, `priority_classifier`, `deepsets_ranknet`, `set_transformer_ranknet`, `listmle`, `temporal_deepsets_ranknet`) plus four Windows RL methods (`masked_double_dqn`, `masked_ppo`, `td_per`, `ranknet_warmstart_ppo`). Windows Deep Sets RankNet completed 50 epochs as PPO initialization; it duplicates WSL Deep Sets and is not a fifteenth unique model.
- WSL ledger: all ten supervised methods are `evaluated_development_only`, 64 episodes and 512 samples each. Final test was not opened.
- Windows logs: all four RL methods reached the configured 20,000 decision/step budget. No Windows stage failures. Teacher methods are not among the completed RL attempts.
- Nine WSL teacher-dependent methods are **blocked**, not trained: seven preference students plus two VLM-feedback RL arms.

## Failures and limits

- Qwen preference qualification failed: 8 contexts/pairs parsed, 4 consistent, agreement 0.50, `qualified=false`. Teacher collection stopped and labels were not promoted.
- WSL rate study failed importing `Evaluation.ue_scenarios` (`ModuleNotFoundError: No module named 'Evaluation'`). This did not stop the completed supervised suite.
- These are development-only results. Independent urgency and final-test claims are not established. WSL report: 3,915 assessable CrisisMMD references; Open Images and Places365 have zero assessable records for that reference.
- This check downloaded no weights, images, or archives. It retrieved only logs and small status/report/ledger JSONs. No notebook was resubmitted or restarted, and no fixes were made.

## Evidence

- `wsl-status.log`, `windows-status.log`: API status and exit codes.
- `wsl-v12-logs.txt`, `windows-v1-logs.txt`: captured per-version logs.
- `wsl-small/experiments/scheduler_pair/runs/wsl-20261102T162419Z/{RUN_STATUS.json,method_ledger.json}` and `jobs/supervised/D/REPORT.json`.
- `windows-small/experiments/scheduler_pair/runs/windows-20261102T154257Z/RUN_STATUS.json` and `jobs/{double_dqn,ppo_per}/D/REPORT.json`.
