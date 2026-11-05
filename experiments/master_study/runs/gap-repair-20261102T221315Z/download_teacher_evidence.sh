#!/usr/bin/env bash
set +e
ROOT="/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End"
BIN="/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
DIR="$ROOT/experiments/master_study/runs/gap-repair-20261102T221315Z"
DEST="$DIR/teacher-evidence"
mkdir -p "$DEST"
PATTERN='.*(requests\.jsonl|responses\.jsonl|preference_reliability\.json|teacher[^/]*\.(json|jsonl|txt)|[^/]*gate[^/]*\.(json|jsonl|txt)|[^/]*prompt[^/]*\.json|[^/]*context[^/]*\.json|[^/]*controls?[^/]*\.(json|jsonl|txt))$'
"$BIN" kernels output satyapaladuguplus/semantic-decoder-scheduler-20261030/12 --path "$DEST" --file-pattern "$PATTERN" >"$DIR/teacher-evidence-download.log" 2>"$DIR/teacher-evidence-download.stderr"
rc=$?
printf '%s\n' "$rc" > "$DIR/teacher-evidence-download.exit"
exit "$rc"