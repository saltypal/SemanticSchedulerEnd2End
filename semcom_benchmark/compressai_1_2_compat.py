"""Compatibility import; implementation lives in EncDec.Research.compressai_1_2_compat."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("EncDec.Research.compressai_1_2_compat")
