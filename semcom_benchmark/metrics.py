"""Compatibility import; implementation lives in EncDec.Research.metrics."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("EncDec.Research.metrics")
