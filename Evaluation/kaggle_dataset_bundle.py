"""Compatibility alias; canonical implementation: Evaluation.data.kaggle_bundle."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.data.kaggle_bundle')
