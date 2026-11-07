"""Compatibility alias; canonical implementation: Scheduler.baselines.fixtures."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Scheduler.baselines.fixtures')
