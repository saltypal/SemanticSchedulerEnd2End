"""Compatibility entry point. Implementation: Scheduler/training/stream_suite/profile_exports.py."""
from pathlib import Path
import importlib, runpy, sys
_root=Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:sys.path.insert(0,str(_root))
if __name__ == '__main__':
    runpy.run_path(str(_root / 'Scheduler/training/stream_suite/profile_exports.py'), run_name='__main__')
else:
    sys.modules[__name__]=importlib.import_module('Scheduler.training.stream_suite.profile_exports')
