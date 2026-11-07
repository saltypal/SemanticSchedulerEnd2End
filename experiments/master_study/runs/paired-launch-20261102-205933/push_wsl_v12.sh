#!/usr/bin/env bash
set +e
ROOT="/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End"
KAGGLE_BIN="/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
LOG="$ROOT/experiments/master_study/runs/paired-launch-20261102-205933/push-wsl-v12.log"
EXIT="$ROOT/experiments/master_study/runs/paired-launch-20261102-205933/push-wsl-v12.exit"
"$KAGGLE_BIN" kernels push -p "$ROOT/experiments/master_study/paired_kaggle/wsl" >"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" > "$EXIT"
exit "$rc"