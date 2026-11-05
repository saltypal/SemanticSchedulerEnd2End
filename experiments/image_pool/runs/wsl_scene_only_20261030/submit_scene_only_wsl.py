from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi

stage = Path(__file__).resolve().parent / 'staging'
notebook = stage / 'Image_Pool_Caching.ipynb'
metadata = stage / 'kernel-metadata.json'
expected_sha = 'FDADE2A918375679C4EF05F77168A75903752AB2465235B096952D0A6E2297A2'
expected_slug = 'satyapaladuguplus/semantic-image-pool-cache-20261030'
expected_timeout = '24000'
raw = notebook.read_bytes()
sha = hashlib.sha256(raw).hexdigest().upper()
nb = nbformat.read(notebook, as_version=4)
nbformat.validate(nb)
md = json.loads(metadata.read_text(encoding='utf-8'))
first_code = next(cell for cell in nb.cells if cell.cell_type == 'code')
source = first_code.source
if sha != expected_sha or len(raw) != 336491 or len(nb.cells) != 36:
    raise SystemExit('staged notebook differs from approved source-ready snapshot')
if not source.startswith("os.environ['IMAGE_POOL_SCENE_ONLY']='1'"):
    raise SystemExit('scene-only setting is not the first setup operation')
if md.get('id') != expected_slug or md.get('is_private') is not True:
    raise SystemExit('staged Kaggle identity/privacy mismatch')
ops = Path(__file__).resolve().parent
provenance = {
    'kernel': expected_slug,
    'intended_version': 1,
    'kernel_title': md.get('title'),
    'notebook_sha256': sha,
    'notebook_bytes': len(raw),
    'cell_count': len(nb.cells),
    'scene_only': True,
    'training_root_required': False,
    'canonical_full_pipeline_default_preserved': True,
    'timeout_seconds': int(expected_timeout),
    'checked_utc': datetime.now(timezone.utc).isoformat(),
}
(ops / 'submission_preflight.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
api = KaggleApi()
api.authenticate()
try:
    response = api.kernels_push(str(stage), timeout=expected_timeout)
    receipt = {
        'kernel': expected_slug,
        'kernel_id': response.kernel_id,
        'version_number': response.version_number,
        'ref': response.ref,
        'url': response.url,
        'error': response.error,
        'invalid_tags': response.invalid_tags,
        'invalid_dataset_sources': response.invalid_dataset_sources,
        'invalid_competition_sources': response.invalid_competition_sources,
        'invalid_kernel_sources': response.invalid_kernel_sources,
        'invalid_model_sources': response.invalid_model_sources,
        'notebook_sha256': sha,
        'notebook_bytes': len(raw),
        'scene_only': True,
        'timeout_seconds': int(expected_timeout),
        'submitted_utc': datetime.now(timezone.utc).isoformat(),
        'execution_status': 'pending_status_verification',
    }
    (ops / 'submission_success.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('kernel','kernel_id','version_number','url','error','notebook_sha256','notebook_bytes','scene_only','timeout_seconds')}, indent=2))
except Exception as exc:
    receipt = {
        'kernel': expected_slug,
        'error_type': type(exc).__name__,
        'status_code': getattr(getattr(exc, 'response', None), 'status_code', None),
        'notebook_sha256': sha,
        'notebook_bytes': len(raw),
        'failed_utc': datetime.now(timezone.utc).isoformat(),
    }
    (ops / 'submission_error.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    raise
