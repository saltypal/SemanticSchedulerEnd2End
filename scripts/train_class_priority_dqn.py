"""Compatibility alias; canonical implementation: scripts.scheduler.train_class_priority_dqn."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from runpy import run_module
    run_module('scripts.scheduler.train_class_priority_dqn', run_name='__main__')
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module('scripts.scheduler.train_class_priority_dqn')
