"""Compatibility alias; canonical implementation: INFRA.data.feature_cache."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('INFRA.data.feature_cache')
