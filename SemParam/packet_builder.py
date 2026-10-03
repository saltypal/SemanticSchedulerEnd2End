"""Compatibility alias; canonical implementation: SemParam.headers.packet."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('SemParam.headers.packet')
