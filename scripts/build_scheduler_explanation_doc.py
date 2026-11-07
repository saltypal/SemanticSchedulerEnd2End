"""Compatibility alias; canonical implementation: scripts.reports.build_scheduler_explanation_doc."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from runpy import run_module
    run_module('scripts.reports.build_scheduler_explanation_doc', run_name='__main__')
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module('scripts.reports.build_scheduler_explanation_doc')
