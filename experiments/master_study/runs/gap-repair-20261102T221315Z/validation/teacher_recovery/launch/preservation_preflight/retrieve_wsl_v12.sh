#!/usr/bin/env bash
set -euo pipefail
/home/Satya/.venvs/semantic-kaggle/bin/kaggle kernels output satyapaladuguplus/semantic-decoder-scheduler-20261030/12 -p /mnt/c/Users/satya/Documents/SemanticSchedulerArtifacts/paired-20261102/historical_state/wsl_v12 --file-pattern '(^|/)(model\.json|resume\.pt|warmstart\.pt|warmstart_actor\.json|artifact_manifest\.json|input_manifest\.json|manifest\.json|run_config\.json|config\.json|RUN_STATUS\.json|shared_manifest\.json|status\.json)$'
