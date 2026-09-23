"""Compatibility loader; implementation moved to Scheduler/."""
import importlib.util as _util
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_spec = _util.spec_from_file_location(__name__, _root / 'Scheduler/__init__.py', submodule_search_locations=[str(_root / 'Scheduler')])
_module = _util.module_from_spec(_spec)
_sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
