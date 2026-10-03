"""Compatibility alias; canonical implementation: SemParam.features.extractors."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('SemParam.features.extractors')
