"""Compatibility alias; canonical implementation: Evaluation.evaluation.semantic."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.evaluation.semantic')
