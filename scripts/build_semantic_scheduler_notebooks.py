"""Compatibility alias; canonical implementation: scripts.notebooks.build_semantic_scheduler_notebooks."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from runpy import run_module
    run_module('scripts.notebooks.build_semantic_scheduler_notebooks', run_name='__main__')
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module('scripts.notebooks.build_semantic_scheduler_notebooks')
