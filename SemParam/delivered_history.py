"""Compatibility alias; canonical implementation: SemParam.history.delivered."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('SemParam.history.delivered')
