#!/usr/bin/env bash
set +e
ROOT="/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End"
BIN="/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
DIR="$ROOT/experiments/master_study/runs/gap-repair-20261102T221315Z"
"$BIN" kernels files satyapaladuguplus/semantic-decoder-scheduler-20261030/12 --page-size 200 --format json >"$DIR/wsl-v12-output-inventory.json" 2>"$DIR/wsl-v12-output-inventory.stderr"
rc=$?
printf '%s\n' "$rc" > "$DIR/wsl-v12-output-inventory.exit"
exit "$rc"