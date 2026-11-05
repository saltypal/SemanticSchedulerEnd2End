#!/usr/bin/env bash
set -euo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cli=/home/Satya/.venvs/semantic-kaggle/bin/kaggle
echo "WSL account target:satyapaladuguplus; source:$dir"
if [[ "${1:-}" != "--submit" ]]; then
  echo "Preview only. Attach private inputs and update dataset_sources before --submit."
  exit 0
fi
/home/Satya/.venvs/semantic-scheduler/bin/python - "$dir/kernel-metadata.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if not m.get('dataset_sources'):raise SystemExit('Attach the private shared-cache dataset and record its dataset_sources ID first.')
if not m['id'].startswith('satyapaladuguplus/'):raise SystemExit('Wrong account metadata')
PY
"$cli" kernels push -p "$dir" --accelerator NvidiaTeslaT4
