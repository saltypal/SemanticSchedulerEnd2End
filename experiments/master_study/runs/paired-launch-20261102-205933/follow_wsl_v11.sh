#!/usr/bin/env bash
set +e
ROOT="/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End"
KAGGLE_BIN="/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
LOG="$ROOT/experiments/master_study/runs/paired-launch-20261102-205933/wsl-v11-follow.log"
EXIT="$ROOT/experiments/master_study/runs/paired-launch-20261102-205933/wsl-v11-follow.exit"
timeout --signal=TERM 300s "$KAGGLE_BIN" kernels logs --follow satyapaladuguplus/semantic-decoder-scheduler-20261030 >"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" > "$EXIT"
exit "$rc"