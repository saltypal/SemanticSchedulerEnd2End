"""Compatibility import; implementation lives in EncDec.Research.digital_phy."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("EncDec.Research.digital_phy")
