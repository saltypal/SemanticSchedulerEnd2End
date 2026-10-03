"""Compatibility alias; canonical implementation: SemParam.features.ladi_prediction."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('SemParam.features.ladi_prediction')
