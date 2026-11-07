"""Notebook hardware/provenance helpers. Source is always the repository."""
import os
from pathlib import Path
import platform

def hardware():
    import torch
    if os.environ.get('TPU_NAME') or os.environ.get('COLAB_TPU_ADDR'):
        raise RuntimeError('TPU/XLA training is not implemented; select a CPU or CUDA runtime')
    return {'environment':'kaggle' if Path('/kaggle').exists() else 'colab' if 'COLAB_RELEASE_TAG' in os.environ else 'local',
            'python':platform.python_version(), 'torch':str(torch.__version__),
            'device':'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu_names':[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}

def require_external_results(checkout,results):
    checkout,results=Path(checkout).resolve(),Path(results).resolve()
    if results == checkout or checkout in results.parents:
        raise ValueError('Persistent results must be outside the disposable source checkout')
    results.mkdir(parents=True,exist_ok=True)
    return results
