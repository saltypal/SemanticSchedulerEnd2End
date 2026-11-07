"""Compatibility alias; canonical implementation: Scheduler.semantic.features."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Scheduler.semantic.features')
