#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End
RUN="$ROOT/experiments/master_study/runs/repair-20261102/validation-final"
PY=/home/Satya/.venvs/semantic-scheduler/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT" || exit 90
{ find parameters Scheduler Test -type f -name '*.py' -print0; find experiments/stream_scheduler experiments/master_study -maxdepth 2 -type f -name '*.py' -print0; } | sort -zu | xargs -0 sha256sum > "$RUN/focused-source-before.sha256"
set +e
$PY -m pytest -q -ra experiments/master_study/tests/test_scientific_repairs.py experiments/master_study/tests/test_controller.py 2>&1 | tee "$RUN/pytest-focused.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN/exit_code-focused.txt"
{ find parameters Scheduler Test -type f -name '*.py' -print0; find experiments/stream_scheduler experiments/master_study -maxdepth 2 -type f -name '*.py' -print0; } | sort -zu | xargs -0 sha256sum > "$RUN/focused-source-after.sha256"
diff -u "$RUN/focused-source-before.sha256" "$RUN/focused-source-after.sha256" > "$RUN/focused-source-diff.patch" || true
printf 'focused_exit_code=%s\n' "$status"
printf 'focused_end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'focused_hash_diff_lines=%s\n' "$(wc -l < "$RUN/focused-source-diff.patch")"
exit "$status"
