#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End
RUN="$ROOT/experiments/master_study/runs/repair-20261102/validation-final"
PY=/home/Satya/.venvs/semantic-scheduler/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT" || exit 90
printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN/run-metadata.txt"
printf 'root=%s\npython=%s\n' "$ROOT" "$PY" | tee -a "$RUN/run-metadata.txt"
printf 'head=%s\n' "$(git rev-parse HEAD)" | tee -a "$RUN/run-metadata.txt"
printf 'thread_limits=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n' | tee -a "$RUN/run-metadata.txt"
printf 'python_version=%s\n' "$($PY -V 2>&1)" | tee -a "$RUN/run-metadata.txt"
printf 'pytest_version=%s\n' "$($PY -m pytest --version 2>&1)" | tee -a "$RUN/run-metadata.txt"
printf 'torch_info=' | tee -a "$RUN/run-metadata.txt"
$PY -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}")' 2>&1 | tee -a "$RUN/run-metadata.txt"
printf '\n' | tee -a "$RUN/run-metadata.txt"
find parameters Scheduler Test experiments/stream_scheduler experiments/master_study -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$RUN/source-before.sha256"
sha256sum pyproject.toml project_compat.py requirements.txt >> "$RUN/source-before.sha256"
set +e
$PY -m pytest -q -ra parameters/tests Scheduler/tests Test/tests experiments/stream_scheduler/tests experiments/master_study/tests 2>&1 | tee "$RUN/pytest.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN/exit_code.txt"
find parameters Scheduler Test experiments/stream_scheduler experiments/master_study -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$RUN/source-after.sha256"
sha256sum pyproject.toml project_compat.py requirements.txt >> "$RUN/source-after.sha256"
diff -u "$RUN/source-before.sha256" "$RUN/source-after.sha256" > "$RUN/source-hash-diff.patch" || true
printf 'exit_code=%s\n' "$status" | tee -a "$RUN/run-metadata.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN/run-metadata.txt"
printf 'hash_diff_lines=%s\n' "$(wc -l < "$RUN/source-hash-diff.patch")" | tee -a "$RUN/run-metadata.txt"
exit "$status"
