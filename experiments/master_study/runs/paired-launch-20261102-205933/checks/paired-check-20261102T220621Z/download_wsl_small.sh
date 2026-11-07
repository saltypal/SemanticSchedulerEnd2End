#!/usr/bin/env bash
set +e
ROOT="/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End"
BIN="/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
DIR="$ROOT/experiments/master_study/runs/paired-launch-20261102-205933/checks/paired-check-20261102T220621Z"
"$BIN" kernels output satyapaladuguplus/semantic-decoder-scheduler-20261030/12 --path "$DIR/wsl-small" --file-pattern '.*(RUN_STATUS|REPORT|method|stage|ledger).*\.json$' >"$DIR/wsl-small-download.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$DIR/wsl-small-download.exit"
exit "$rc"