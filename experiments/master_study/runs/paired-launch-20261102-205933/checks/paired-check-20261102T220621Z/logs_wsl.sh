#!/usr/bin/env bash
set +e
ROOT="/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End"
BIN="/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
DIR="$ROOT/experiments/master_study/runs/paired-launch-20261102-205933/checks/paired-check-20261102T220621Z"
"$BIN" kernels logs satyapaladuguplus/semantic-decoder-scheduler-20261030/12 >"$DIR/wsl-v12-logs.txt" 2>&1
rc=$?
printf '%s\n' "$rc" > "$DIR/wsl-logs.exit"
exit "$rc"