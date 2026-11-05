"""Compatibility alias; canonical implementation: Evaluation.scheduling.streams."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.scheduling.streams')
