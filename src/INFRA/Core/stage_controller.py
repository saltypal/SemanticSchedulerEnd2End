"""Compatibility import; use INFRA.Core.stage_controller."""
import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_target = _importlib.import_module('INFRA.Core.stage_controller')

_sys.modules[__name__] = _target
