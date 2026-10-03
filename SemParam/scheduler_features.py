"""Compatibility alias; canonical implementation: SemParam.features.scheduler."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('SemParam.features.scheduler')
