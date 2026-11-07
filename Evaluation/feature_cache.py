"""Compatibility alias; canonical implementation: Evaluation.data.feature_cache."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.data.feature_cache')
