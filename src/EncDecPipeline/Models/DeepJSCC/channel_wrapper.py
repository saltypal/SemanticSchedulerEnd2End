"""Compatibility import; use Codec.Models.DeepJSCC.channel_wrapper."""
import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[4]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_target = _importlib.import_module('Codec.Models.DeepJSCC.channel_wrapper')

_sys.modules[__name__] = _target
