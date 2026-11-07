"""Compatibility alias; canonical implementation: scripts.maintenance.refresh_graph."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from runpy import run_module
    run_module('scripts.maintenance.refresh_graph', run_name='__main__')
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module('scripts.maintenance.refresh_graph')
