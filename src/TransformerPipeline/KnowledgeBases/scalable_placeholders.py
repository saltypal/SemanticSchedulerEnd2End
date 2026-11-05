"""Compatibility import; use Archived.owners_before_simplification_20261027.SemParam.KnowledgeBases.scalable_placeholders."""
import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_target = _importlib.import_module('Archived.owners_before_simplification_20261027.SemParam.KnowledgeBases.scalable_placeholders')

_sys.modules[__name__] = _target
