"""Compatibility alias; canonical implementation: Evaluation.scheduling.simulation."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.scheduling.simulation')
