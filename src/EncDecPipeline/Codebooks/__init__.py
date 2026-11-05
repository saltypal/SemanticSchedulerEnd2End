"""Compatibility import; use Codec.Codebooks."""
import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_target = _importlib.import_module('Codec.Codebooks')

def __getattr__(name):
    return getattr(_target, name)

__all__ = getattr(_target, "__all__", [name for name in vars(_target) if not name.startswith("_")])
