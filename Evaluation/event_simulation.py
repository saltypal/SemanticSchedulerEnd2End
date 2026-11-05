"""Compatibility alias; canonical implementation: Evaluation.scheduling.event_simulation."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.scheduling.event_simulation')
