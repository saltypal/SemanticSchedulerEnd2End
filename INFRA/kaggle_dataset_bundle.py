"""Compatibility alias; canonical implementation: INFRA.data.kaggle_bundle."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('INFRA.data.kaggle_bundle')
